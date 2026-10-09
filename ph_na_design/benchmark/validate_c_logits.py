#!/usr/bin/env python
"""
使用NA-MPNN的概率筛选允许C突变的位点
===================================

新思路：
1. 对每个WT结构，mask掉可设计位置（非G-runs）
2. 获取NA-MPNN在每个可设计位点输出的C的probability
3. 筛选所有probability > threshold的位点作为"允许C"的位置
4. 输出可用于单位点/双位点AF筛选的候选位置

Usage:
    python validate_c_logits.py [--temperature 0.1] [--prob_threshold 0.0]
"""

import os
import sys
import csv
import json
import argparse
import urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
NAMPNN_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NAMPNN_ROOT))
sys.path.insert(0, str(NAMPNN_ROOT / "inference"))


def load_dataset(csv_path):
    """加载pH响应数据集"""
    dataset = {
        'wt': [],
        'success': [],
        'failed': []
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['pdb_id']:
                continue
            
            entry = {
                'pdb_id': row['pdb_id'],
                'name': row['name'],
                'sequence': row['sequence'],
                'ph_response': row['ph_response'],
                'c_positions': [int(x) for x in row['c_positions'].split(',') if x.strip()],
                'fixed_pattern': row.get('fixed_pattern', ''),
                'topology': row.get('topology', '')
            }
            
            if entry['name'] == 'WT':
                dataset['wt'].append(entry)
            elif entry['ph_response'] == 'Y':
                dataset['success'].append(entry)
            else:
                dataset['failed'].append(entry)
    
    return dataset


def parse_fixed_pattern(pattern):
    """解析固定位置pattern，返回固定位置列表 (1-indexed)"""
    if not pattern:
        return []
    fixed = []
    for i, c in enumerate(pattern, 1):
        if c != '.':
            fixed.append(i)
    return fixed


def download_pdb(pdb_id, output_dir):
    """从RCSB下载PDB文件"""
    pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, pdb_path)
        return pdb_path
    except Exception as e:
        print(f"    Failed to download {pdb_id}: {e}")
        return None


def get_c_logits_for_structure(pdb_path, fixed_positions, wt_sequence=None, temperature=0.1, device=None):
    """
    获取NA-MPNN在每个可设计位点对C的logits
    
    Args:
        pdb_path: PDB文件路径
        fixed_positions: 固定位置列表 (1-indexed)
        wt_sequence: WT序列字符串，用于识别已有C的位置
        temperature: 采样温度
        device: torch device
    
    Returns:
        dict: {
            'chain_id': str,
            'sequence_length': int,
            'designable_positions': list,  # 可设计位置 (1-indexed)
            'c_logits': dict,  # {position: logit} for C at each designable position
            'c_probs': dict,   # {position: prob} for C at each designable position
            'all_logits': np.array,  # [L, 33] full logits
            'restype_to_int': dict,
            'existing_c_positions': list,  # WT中已有C的位置
            'novel_c_logits': dict,  # 只包含非C位置的logits (用于选择新C位点)
            'novel_c_probs': dict
        }
    """
    from data_utils import parse_PDB, featurize
    from model_utils import ProteinMPNN
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # NA-MPNN配置
    atom_types = [
        'N', 'CA', 'C', 'O',
        'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"
    ]
    atom_dict = dict(zip(atom_types, range(len(atom_types))))
    
    polytypes = ['PP', 'DNA', 'RNA', 'UNK', 'MAS', 'PAD']
    polytype_to_int = dict(zip(polytypes, range(len(polytypes))))
    
    restypes = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'UNK', 'DA', 'DC', 'DG', 'DT', 'DX', 'A', 'C', 'G', 'U', 'RX', 'MAS', 'PAD'
    ]
    restype_to_int = dict(zip(restypes, range(len(restypes))))
    
    # DNA/RNA共享token
    restype_to_int["A"] = restype_to_int["DA"]
    restype_to_int["C"] = restype_to_int["DC"]
    restype_to_int["G"] = restype_to_int["DG"]
    restype_to_int["U"] = restype_to_int["DT"]
    restype_to_int["RX"] = restype_to_int["DX"]
    
    # DC的索引（这是我们要查询的C的logits）
    DC_IDX = restype_to_int["DC"]
    
    # 加载模型
    checkpoint_path = str(NAMPNN_ROOT / "models" / "design_model" / "s_19137.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=32,
        model_type="na_mpnn",
        vocab=33,
        num_letters=33,
        atom_dict=atom_dict,
        restype_to_int=restype_to_int,
        polytype_to_int=polytype_to_int
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 解析PDB
    macromolecule_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
        pdb_path,
        device=device,
        chains="",
        model_type="na_mpnn",
        parse_na_only=1,  # 只解析核酸
        na_shared_tokens=1,
        load_residues_with_missing_atoms=0
    )
    
    # 获取残基编码
    R_idx_list = list(macromolecule_dict["R_idx"].cpu().numpy())
    chain_letters_list = list(macromolecule_dict["chain_letters"])
    encoded_residues = []
    for i in range(len(R_idx_list)):
        tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
        encoded_residues.append(tmp)
    
    # 找出核酸链
    na_chain = chain_letters_list[0] if chain_letters_list else 'A'
    
    # 创建fixed_positions字符串（用于识别哪些是固定的）
    fixed_residue_strs = [f"{na_chain}{pos}" for pos in fixed_positions]
    
    # 创建chain_mask: 0=固定, 1=可设计
    chain_mask = torch.tensor(
        [float(item not in fixed_residue_strs) for item in encoded_residues],
        device=device
    )
    
    # 设置chain_mask到macromolecule_dict
    macromolecule_dict["chain_mask"] = chain_mask
    
    with torch.no_grad():
        feature_dict = featurize(macromolecule_dict)
        feature_dict["batch_size"] = 1
        B, L, _, _ = feature_dict["X"].shape
        
        # 设置bias和temperature
        feature_dict["temperature"] = temperature
        feature_dict["bias"] = torch.zeros([1, L, 33], device=device)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        feature_dict["randn"] = torch.randn([1, L], device=device)
        
        # 使用score方法获取完整的log_probs（不采样，直接看条件概率）
        # score方法会给出每个位置的log_probs，考虑了上下文
        output_dict = model.score(feature_dict)
        log_probs = output_dict["log_probs"]  # [B, L, 33]
        
        # 转换为numpy
        log_probs_np = log_probs[0].cpu().numpy()  # [L, 33]
        probs_np = np.exp(log_probs_np)  # [L, 33]
    
    # 识别WT中已有C的位置
    existing_c_positions = []
    if wt_sequence:
        for i, base in enumerate(wt_sequence, 1):
            if base == 'C':
                existing_c_positions.append(i)
    
    # 提取结果
    result = {
        'chain_id': na_chain,
        'sequence_length': L,
        'encoded_residues': encoded_residues,
        'designable_positions': [],
        'c_logits': {},
        'c_probs': {},
        'novel_c_logits': {},  # 只包含非C位置
        'novel_c_probs': {},
        'existing_c_positions': existing_c_positions,
        'all_logits': log_probs_np,
        'restype_to_int': restype_to_int,
        'DC_IDX': DC_IDX
    }
    
    # 提取每个可设计位置的C logits
    for i, res_str in enumerate(encoded_residues):
        # 从res_str中提取位置号 (e.g., "A1" -> 1)
        pos = int(''.join(filter(str.isdigit, res_str)))
        
        if pos not in fixed_positions:
            result['designable_positions'].append(pos)
            result['c_logits'][pos] = float(log_probs_np[i, DC_IDX])
            result['c_probs'][pos] = float(probs_np[i, DC_IDX])
            
            # 只有非C位置才加入novel_c_logits
            if pos not in existing_c_positions:
                result['novel_c_logits'][pos] = float(log_probs_np[i, DC_IDX])
                result['novel_c_probs'][pos] = float(probs_np[i, DC_IDX])
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Filter C-allowed positions using NA-MPNN probability")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--prob_threshold", type=float, default=0.0, help="Probability threshold for C-allowed positions")
    args = parser.parse_args()
    
    print("=" * 100)
    print("NA-MPNN C-Allowed Position Filtering for pH-Responsive Design")
    print("=" * 100)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Temperature: {args.temperature}")
    print(f"Probability Threshold: {args.prob_threshold} (positions with prob > {args.prob_threshold} are C-allowed)")
    
    # 1. 加载数据集
    csv_path = os.path.join(os.path.dirname(__file__), 'ph_response_dataset.csv')
    dataset = load_dataset(csv_path)
    
    print(f"\n[Dataset]")
    print(f"  Wild-types: {len(dataset['wt'])}")
    print(f"  Success (pH response=Y): {len(dataset['success'])}")
    print(f"  Failed (pH response=N): {len(dataset['failed'])}")
    
    # 2. 准备PDB缓存
    pdb_cache_dir = os.path.join(os.path.dirname(__file__), 'pdb_cache')
    os.makedirs(pdb_cache_dir, exist_ok=True)
    
    # 3. 为每个WT计算C logits
    print(f"\n[Computing C Logits for Each WT]")
    print("-" * 100)
    
    wt_logits = {}  # pdb_id -> result from get_c_logits_for_structure
    
    for wt in dataset['wt']:
        pdb_id = wt['pdb_id']
        fixed_pattern = wt['fixed_pattern']
        fixed_positions = parse_fixed_pattern(fixed_pattern)
        
        print(f"\n  {pdb_id}:")
        print(f"    Sequence: {wt['sequence']}")
        print(f"    Fixed positions: {len(fixed_positions)}")
        
        # 下载PDB
        pdb_path = download_pdb(pdb_id, pdb_cache_dir)
        if not pdb_path:
            continue
        
        try:
            result = get_c_logits_for_structure(
                pdb_path, fixed_positions,
                wt_sequence=wt['sequence'],
                temperature=args.temperature, 
                device=device
            )
            wt_logits[pdb_id] = result
            
            # 显示WT中已有C的位置
            existing_c = result['existing_c_positions']
            print(f"    Existing C in WT: {existing_c}")
            
            # 筛选probability > threshold的位置（排除已有C）
            c_allowed_positions = {
                pos: prob for pos, prob in result['novel_c_probs'].items()
                if prob > args.prob_threshold
            }
            sorted_allowed = sorted(c_allowed_positions.items(), key=lambda x: x[1], reverse=True)
            
            # 计算单位点和双位点组合数
            n_allowed = len(c_allowed_positions)
            n_single = n_allowed
            n_double = n_allowed * (n_allowed - 1) // 2
            
            print(f"    Designable positions: {len(result['designable_positions'])} (novel: {len(result['novel_c_logits'])})")
            print(f"    C-allowed positions (prob>{args.prob_threshold}): {n_allowed}")
            print(f"    Mutation combinations: {n_single} single + {n_double} double = {n_single + n_double} total")
            if sorted_allowed:
                positions_str = ', '.join([f"{pos}({prob:.3f})" for pos, prob in sorted_allowed])
                print(f"    Positions: {positions_str}")
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 验证正样本和负样本
    print(f"\n[Validation Results]")
    print("=" * 100)
    
    success_results = []
    failed_results = []
    
    # 处理所有突变（成功和失败）
    for entry in dataset['success'] + dataset['failed']:
        pdb_id = entry['pdb_id']
        c_positions = entry['c_positions']
        
        if pdb_id not in wt_logits or not c_positions:
            continue
        
        result = wt_logits[pdb_id]
        existing_c = set(result['existing_c_positions'])
        
        # 区分：实验引入的是"新C"还是"已有C"
        novel_c_positions = [pos for pos in c_positions if pos not in existing_c]
        existing_c_in_mutation = [pos for pos in c_positions if pos in existing_c]
        
        # 计算新C位点的概率之和（使用novel_c_probs）
        novel_c_probs_sum = sum(result['novel_c_probs'].get(pos, 0) for pos in novel_c_positions)
        
        # 筛选C-allowed位置（probability > threshold）
        c_allowed_positions = {
            pos: prob for pos, prob in result['novel_c_probs'].items()
            if prob > args.prob_threshold
        }
        sorted_allowed = sorted(c_allowed_positions.items(), key=lambda x: x[1], reverse=True)
        c_allowed_set = set(c_allowed_positions.keys())
        
        # 计算有多少新C位点在C-allowed中
        novel_in_allowed = sum(1 for pos in novel_c_positions if pos in c_allowed_set)
        
        # 获取每个新C位点的概率
        novel_c_probs_detail = {pos: result['novel_c_probs'].get(pos, 0) for pos in novel_c_positions}
        
        # 计算组合数
        n_allowed = len(c_allowed_positions)
        n_single = n_allowed
        n_double = n_allowed * (n_allowed - 1) // 2
        
        entry_result = {
            'pdb_id': pdb_id,
            'name': entry['name'],
            'c_positions': c_positions,
            'novel_c_positions': novel_c_positions,
            'novel_c_probs_detail': novel_c_probs_detail,
            'existing_c_in_mutation': existing_c_in_mutation,
            'novel_c_probs_sum': novel_c_probs_sum,
            'novel_in_allowed': novel_in_allowed,
            'total_novel_c': len(novel_c_positions),
            'c_allowed_positions': sorted_allowed,
            'n_allowed': n_allowed,
            'n_single': n_single,
            'n_double': n_double,
            'ph_response': entry['ph_response']
        }
        
        if entry['ph_response'] == 'Y':
            success_results.append(entry_result)
        else:
            failed_results.append(entry_result)
    
    # 5. 显示结果
    print(f"\n[Success Cases (pH response = Y)] - C-Allowed Analysis")
    print("=" * 140)
    
    for r in success_results:
        pdb_id = r['pdb_id']
        result = wt_logits[pdb_id]
        existing_c = result['existing_c_positions']
        
        # 获取C-allowed位点
        allowed_str = ', '.join([f"{pos}({prob:.3f})" for pos, prob in r['c_allowed_positions']])
        
        # 获取真正新C的概率
        novel_c_probs_str = [f"{pos}({r['novel_c_probs_detail'].get(pos, 0):.3f})" for pos in r['novel_c_positions']]
        
        # 检查novel C是否在C-allowed中
        c_allowed_set = set([pos for pos, _ in r['c_allowed_positions']])
        novel_c_set = set(r['novel_c_positions'])
        matched = novel_c_set & c_allowed_set
        not_matched = novel_c_set - c_allowed_set
        
        print(f"\n  {pdb_id} {r['name']}:")
        print(f"    原C位点:      {','.join(map(str, r['c_positions']))}")
        print(f"    WT已有C:      {','.join(map(str, existing_c)) if existing_c else '无'}")
        print(f"    真正新C:      {', '.join(novel_c_probs_str) if novel_c_probs_str else '无'}")
        print(f"    C-allowed位点: {allowed_str if allowed_str else '无'} ({r['n_allowed']}个)")
        print(f"    组合数:       {r['n_single']}单 + {r['n_double']}双 = {r['n_single'] + r['n_double']}总")
        if r['novel_in_allowed'] == r['total_novel_c']:
            print(f"    ✓ 覆盖:       所有新C都在C-allowed中")
        elif r['novel_in_allowed'] > 0:
            print(f"    △ 部分覆盖:   {','.join(map(str, sorted(matched)))}在内, {','.join(map(str, sorted(not_matched)))}不在")
        else:
            print(f"    ✗ 未覆盖:     所有新C都不在C-allowed中")
    
    print(f"\n\n[Failed Cases (pH response = N)] - Top 10 by novel C probability sum")
    print("=" * 140)
    failed_results.sort(key=lambda x: x['novel_c_probs_sum'], reverse=True)
    
    for r in failed_results[:10]:
        pdb_id = r['pdb_id']
        if pdb_id not in wt_logits:
            continue
        result = wt_logits[pdb_id]
        existing_c = result['existing_c_positions']
        
        allowed_str = ', '.join([f"{pos}({prob:.3f})" for pos, prob in r['c_allowed_positions']])
        novel_c_probs_str = [f"{pos}({r['novel_c_probs_detail'].get(pos, 0):.3f})" for pos in r['novel_c_positions']]
        
        c_allowed_set = set([pos for pos, _ in r['c_allowed_positions']])
        novel_c_set = set(r['novel_c_positions'])
        matched = novel_c_set & c_allowed_set
        
        print(f"\n  {pdb_id} {r['name']}:")
        print(f"    原C位点:      {','.join(map(str, r['c_positions']))}")
        print(f"    WT已有C:      {','.join(map(str, existing_c)) if existing_c else '无'}")
        print(f"    真正新C:      {', '.join(novel_c_probs_str) if novel_c_probs_str else '无'}")
        print(f"    C-allowed位点: {allowed_str if allowed_str else '无'} ({r['n_allowed']}个)")
        print(f"    组合数:       {r['n_single']}单 + {r['n_double']}双 = {r['n_single'] + r['n_double']}总")
        if r['novel_in_allowed'] == r['total_novel_c']:
            print(f"    ✓ 覆盖")
        elif r['novel_in_allowed'] > 0:
            print(f"    △ 部分覆盖 ({r['novel_in_allowed']}/{r['total_novel_c']})")
        else:
            print(f"    ✗ 未覆盖")
    
    # 6. 统计分析
    print(f"\n[Statistical Analysis] - C-Allowed Coverage (prob > {args.prob_threshold})")
    print("=" * 100)
    
    # 过滤掉没有novel C的样本
    success_with_novel = [r for r in success_results if r['total_novel_c'] > 0]
    failed_with_novel = [r for r in failed_results if r['total_novel_c'] > 0]
    
    if success_with_novel:
        success_coverage = [r['novel_in_allowed'] / r['total_novel_c'] for r in success_with_novel]
        success_n_allowed = [r['n_allowed'] for r in success_with_novel]
        success_total_combos = [r['n_single'] + r['n_double'] for r in success_with_novel]
        
        print(f"\n  Success Cases (n={len(success_with_novel)}):")
        print(f"    Novel C coverage rate: {np.mean(success_coverage)*100:.1f}% (新C在C-allowed中的比例)")
        print(f"    Average C-allowed positions: {np.mean(success_n_allowed):.1f}")
        print(f"    Average total combinations: {np.mean(success_total_combos):.1f}")
        
        # 统计完全覆盖、部分覆盖、未覆盖
        full_cover = sum(1 for r in success_with_novel if r['novel_in_allowed'] == r['total_novel_c'])
        partial_cover = sum(1 for r in success_with_novel if 0 < r['novel_in_allowed'] < r['total_novel_c'])
        no_cover = sum(1 for r in success_with_novel if r['novel_in_allowed'] == 0)
        
        print(f"    Full coverage: {full_cover}/{len(success_with_novel)} ({100*full_cover/len(success_with_novel):.1f}%)")
        print(f"    Partial coverage: {partial_cover}/{len(success_with_novel)} ({100*partial_cover/len(success_with_novel):.1f}%)")
        print(f"    No coverage: {no_cover}/{len(success_with_novel)} ({100*no_cover/len(success_with_novel):.1f}%)")
    
    if failed_with_novel:
        failed_coverage = [r['novel_in_allowed'] / r['total_novel_c'] for r in failed_with_novel]
        failed_n_allowed = [r['n_allowed'] for r in failed_with_novel]
        failed_total_combos = [r['n_single'] + r['n_double'] for r in failed_with_novel]
        
        print(f"\n  Failed Cases (n={len(failed_with_novel)}):")
        print(f"    Novel C coverage rate: {np.mean(failed_coverage)*100:.1f}%")
        print(f"    Average C-allowed positions: {np.mean(failed_n_allowed):.1f}")
        print(f"    Average total combinations: {np.mean(failed_total_combos):.1f}")
    
    # 总体统计
    all_results = success_results + failed_results
    if all_results:
        print(f"\n  Overall ({len(all_results)} mutations):")
        total_allowed = sum(r['n_allowed'] for r in all_results)
        total_single = sum(r['n_single'] for r in all_results)
        total_double = sum(r['n_double'] for r in all_results)
        print(f"    Total C-allowed positions: {total_allowed}")
        print(f"    Total combinations: {total_single} single + {total_double} double = {total_single + total_double}")
    
    # 7. 关键问题验证
    print(f"\n[Key Validation Questions] - C-Allowed Analysis (prob > {args.prob_threshold})")
    print("=" * 100)
    
    # Q1: 正样本的新C位点是否在C-allowed中？
    success_with_novel = [r for r in success_results if r['total_novel_c'] > 0]
    
    if success_with_novel:
        success_perfect = sum(1 for r in success_with_novel if r['novel_in_allowed'] == r['total_novel_c'])
        success_partial = sum(1 for r in success_with_novel if 0 < r['novel_in_allowed'] < r['total_novel_c'])
        success_none = sum(1 for r in success_with_novel if r['novel_in_allowed'] == 0)
        
        print(f"\n  Q1: Are successful NOVEL C positions in NA-MPNN's C-allowed set?")
        print(f"      Full coverage: {success_perfect}/{len(success_with_novel)} ({100*success_perfect/len(success_with_novel):.1f}%)")
        print(f"      Partial coverage: {success_partial}/{len(success_with_novel)} ({100*success_partial/len(success_with_novel):.1f}%)")
        print(f"      No coverage: {success_none}/{len(success_with_novel)} ({100*success_none/len(success_with_novel):.1f}%)")
        
        # Q2: 详细列出每个成功案例的覆盖情况
        print(f"\n  Q2: Success case coverage details:")
        for r in success_with_novel:
            if r['novel_in_allowed'] == r['total_novel_c']:
                match = "✓"
            elif r['novel_in_allowed'] > 0:
                match = "△"
            else:
                match = "✗"
            novel_c_str = ','.join(map(str, r['novel_c_positions']))
            allowed_str = ','.join([str(pos) for pos, _ in r['c_allowed_positions']])
            print(f"      {match} {r['pdb_id']} {r['name']}: NovelC={novel_c_str}")
            print(f"         C-allowed: {allowed_str} ({r['n_allowed']}个, {r['n_single'] + r['n_double']}组合)")
    else:
        print(f"\n  No success cases with novel C positions")

    # 8. 全序列衍生规模分析 (26个WT)
    print(f"\n[Full Sequence Derivative Analysis] - 26 Wild-Types")
    print("=" * 100)
    print(f"{'PDB ID':<8} | {'Seq Length':<10} | {'Allowed Pos':<12} | {'+1 C':<8} | {'+2 C':<8} | {'Total Designs':<12}")
    print("-" * 100)
    
    total_designs_all = 0
    for pdb_id, result in wt_logits.items():
        n_allowed = len(result['novel_c_probs']) # 所有 prob > 0 的新位置
        n_single = n_allowed
        n_double = n_allowed * (n_allowed - 1) // 2
        total = n_single + n_double
        total_designs_all += total
        print(f"{pdb_id:<8} | {result['sequence_length']:<10} | {n_allowed:<12} | {n_single:<8} | {n_double:<8} | {total:<12}")
    
    print("-" * 100)
    print(f"{'TOTAL':<8} | {'-':<10} | {'-':<12} | {'-':<8} | {'-':<8} | {total_designs_all:<12}")
    print(f"\nConclusion: Average {total_designs_all/len(wt_logits):.1f} designs per WT for AF3 validation.")

    
    # 8. 保存结果
    output_path = os.path.join(os.path.dirname(__file__), 'c_allowed_positions.json')
    
    # 计算统计信息（如果有数据）
    stats = {}
    if success_with_novel:
        stats['success_coverage_rate'] = float(np.mean([r['novel_in_allowed'] / r['total_novel_c'] for r in success_with_novel]))
        stats['success_avg_allowed'] = float(np.mean([r['n_allowed'] for r in success_with_novel]))
    if failed_with_novel:
        stats['failed_coverage_rate'] = float(np.mean([r['novel_in_allowed'] / r['total_novel_c'] for r in failed_with_novel]))
        stats['failed_avg_allowed'] = float(np.mean([r['n_allowed'] for r in failed_with_novel]))
    
    # 清理不可序列化的数据
    def clean_result(r):
        cleaned = dict(r)
        # 转换c_allowed_positions为可序列化格式
        cleaned['c_allowed_positions'] = [{'pos': pos, 'prob': prob} for pos, prob in r['c_allowed_positions']]
        return cleaned
    
    save_data = {
        'config': {
            'temperature': args.temperature,
            'prob_threshold': args.prob_threshold
        },
        'success_results': [clean_result(r) for r in success_results],
        'failed_results': [clean_result(r) for r in failed_results],
        'statistics': stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n[Results saved to {output_path}]")
    print("=" * 100)


if __name__ == "__main__":
    main()
