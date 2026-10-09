#!/usr/bin/env python
"""
pH-Responsive Nucleic Acid Design - Full Pipeline
==================================================

完整的pH敏感核酸设计流程，用于验证26个G4 WT的C-rich衍生物筛选。

English Summary:
This pipeline implements a three-stage workflow for designing pH-responsive nucleic acids:
1. Library Generation: Physics-based (CC distance) and Statistics-based (NA-MPNN) filtering.
2. Structure Prediction: High-throughput folding using AlphaFold3.
3. Hierarchical Screening: Layered verification using DSSR (G4 detection) and 3DNA parameters.
   - Key Innovation (2025-01): Strict Stagger filter (< 1.0Å) and BPO Euclidean distance ranking.

Workflow (Three-Stage Pipeline):
  1. Library: CC距离筛选(≤15Å) → NA-MPNN概率筛选(prob>0.01) → 生成+1C/+2C衍生物库
  2. Fold:    Local AF3结构预测（多采样 num_diffusion_samples=5）
  3. Screen:  结构质量控制 → DSSR验证G4 → 几何参数筛选 → BPO欧式距离排序

Screening Strategy (2025-01 优化版):
  - Layer 1 (Structure Quality): 
    - 结构合理性检查 (min_clash_dist >= 1.4Å)
    - 核心前置筛选：首先排除AF3预测失败产生的碱基变形或重叠的 artifact 结构
  - Layer 2 (G4 Detection): DSSR 检测 G-quadruplex 核心 (≥ 2 tetrads)
  - Layer 3 (Geometry Filter): 
    - |Stagger| < 1.0Å (确保处于同一平面高度)
    - 8.5Å ≤ C1'-C1' Dist ≤ 12.5Å (标准碱基配对距离范围)
  - Ranking: BPO 欧式距离 = sqrt(B² + P² + O²)，越小越好

核心逻辑:
  - Clash检测作为第一层，排除碱基重叠/变形的artifact（节省后续计算）
  - Stagger 决定两个 C 是否在同一层高度（质子化难以改变）
  - BPO 角度反映质子化后成键潜力（质子化可纠正角度偏差）

Filtering Strategy (Library Stage):
  - Stage 1 (Physical): CC distance constraint
    - +1C: distance to nearest existing C ≤ 15Å
    - +2C: distance between two new C ≤ 15Å
  - Stage 2 (Statistical): NA-MPNN probability filter
    - Keep positions where P(C) > 0.01

Usage:
  python full_pipeline.py --stage all              # 运行全部阶段
  python full_pipeline.py --stage library          # 仅生成衍生物库
  python full_pipeline.py --stage fold             # 仅AF3折叠
  python full_pipeline.py --stage screen           # 仅筛选
  python full_pipeline.py --stage library,fold     # 组合运行
"""

import os
import sys
import csv
import json
import time
import argparse
import subprocess
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Any, Optional
import numpy as np

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
NAMPNN_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NAMPNN_ROOT))
sys.path.insert(0, str(NAMPNN_ROOT / "inference"))

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # 数据集路径
    "dataset_csv": PROJECT_ROOT / "benchmark" / "ph_response_dataset.csv",
    "pdb_cache": PROJECT_ROOT / "benchmark" / "pdb_cache",
    
    # 输出目录
    "output_root": PROJECT_ROOT / "benchmark" / "pipeline_output",
    "library_dir": "01_library",
    "af3_dir": "02_af3_structures", 
    "screen_dir": "03_screening_results",
    
    # NA-MPNN参数
    "nampnn_temperature": 0.1,
    "prob_threshold": 0.01,  # P(C) > 0.01 保证100%阳性样本覆盖
    
    # AF3参数。路径不写死，投稿/复现时用环境变量指定。
    "af3_conda_env": os.environ.get("AF3_CONDA_ENV", "AF3"),
    "af3_dir_path": os.environ.get("AF3_DIR", ""),
    "af3_model_dir": os.environ.get("AF3_MODEL_DIR", ""),
    "af3_db_dir": os.environ.get("AF3_DB_DIR", ""),
    "skip_msa": True,
    "num_diffusion_samples": 5,  # 与 AF3 Server 一致，增加 G4 检测成功率
    
    # 筛选参数
    "require_g4": True,
    "cc_close_threshold": 4.0,
    "cc_medium_threshold": 6.0,
    
    # 两阶段筛选参数
    # Stage 1: 物理约束 - CC距离
    "cc_pair_dist_threshold": 15.0,  # C1'-C1' 距离阈值 (Å)
    # Stage 2: 统计约束 - NA-MPNN概率 (prob_threshold已在上面定义)
}


# ============================================================================
# Stage 1: Library Generation
# ============================================================================

def load_wt_dataset(csv_path: Path) -> List[Dict]:
    """加载WT数据集"""
    wt_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['name'] == 'WT' and row['pdb_id']:
                wt_list.append({
                    'pdb_id': row['pdb_id'],
                    'sequence': row['sequence'],
                    'fixed_pattern': row['fixed_pattern'],
                    'topology': row['topology'],
                })
    return wt_list


def get_c1_coords(pdb_path: str) -> Dict[int, np.ndarray]:
    """
    获取PDB结构中所有残基的C1'原子坐标
    
    用于空间约束筛选：判断两个位置是否足够近以形成CC配对
    
    Returns:
        Dict[residue_id, coord_array]
    """
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', pdb_path)
        chain = list(structure[0].get_chains())[0]
        coords = {}
        for res in chain:
            res_id = res.get_id()[1]
            if "C1'" in res:
                coords[res_id] = res["C1'"].get_coord()
        return coords
    except Exception as e:
        print(f"  Warning: Could not read C1' coords from {pdb_path}: {e}")
        return {}


def get_spatial_distance(coords: Dict[int, np.ndarray], pos1: int, pos2: int) -> Optional[float]:
    """计算两个残基C1'原子之间的距离"""
    if pos1 in coords and pos2 in coords:
        return float(np.linalg.norm(coords[pos1] - coords[pos2]))
    return None


def parse_fixed_pattern(pattern: str) -> List[int]:
    """解析固定位置pattern，返回固定位置列表 (1-indexed)"""
    if not pattern:
        return []
    fixed = []
    for i, c in enumerate(pattern, 1):
        if c != '.':
            fixed.append(i)
    return fixed


def get_c_logits_for_structure(pdb_path: str, fixed_positions: List[int], 
                                wt_sequence: str, temperature: float = 0.1):
    """
    获取NA-MPNN在每个可设计位点对C的logits以及完整概率分布
    
    新增功能：
    - 返回原始碱基的概率 (original_base_probs)
    - 返回突变代价 delta_log_p = log P(C) - log P(original)
    - 支持计算组合突变的总代价
    """
    import torch
    from data_utils import parse_PDB, featurize
    from model_utils import ProteinMPNN
    
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
    restype_to_int["A"] = restype_to_int["DA"]
    restype_to_int["C"] = restype_to_int["DC"]
    restype_to_int["G"] = restype_to_int["DG"]
    restype_to_int["U"] = restype_to_int["DT"]
    restype_to_int["RX"] = restype_to_int["DX"]
    
    # 碱基名到索引的映射（用于查找原始碱基的概率）
    base_to_idx = {
        'A': restype_to_int["DA"],
        'C': restype_to_int["DC"],
        'G': restype_to_int["DG"],
        'T': restype_to_int["DT"],
        'U': restype_to_int["DT"],
    }
    DC_IDX = restype_to_int["DC"]
    
    # 加载模型
    checkpoint_path = str(NAMPNN_ROOT / "models" / "design_model" / "s_19137.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32,
        model_type="na_mpnn", vocab=33, num_letters=33,
        atom_dict=atom_dict, restype_to_int=restype_to_int, polytype_to_int=polytype_to_int
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 解析PDB
    macromolecule_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
        pdb_path, device=device, chains="", model_type="na_mpnn",
        parse_na_only=1, na_shared_tokens=1, load_residues_with_missing_atoms=0
    )
    
    R_idx_list = list(macromolecule_dict["R_idx"].cpu().numpy())
    chain_letters_list = list(macromolecule_dict["chain_letters"])
    encoded_residues = []
    for i in range(len(R_idx_list)):
        tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
        encoded_residues.append(tmp)
    
    na_chain = chain_letters_list[0] if chain_letters_list else 'A'
    fixed_residue_strs = [f"{na_chain}{pos}" for pos in fixed_positions]
    
    chain_mask = torch.tensor(
        [float(item not in fixed_residue_strs) for item in encoded_residues],
        device=device
    )
    macromolecule_dict["chain_mask"] = chain_mask
    
    with torch.no_grad():
        feature_dict = featurize(macromolecule_dict)
        feature_dict["batch_size"] = 1
        B, L, _, _ = feature_dict["X"].shape
        feature_dict["temperature"] = temperature
        feature_dict["bias"] = torch.zeros([1, L, 33], device=device)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        feature_dict["randn"] = torch.randn([1, L], device=device)
        
        output_dict = model.score(feature_dict)
        log_probs = output_dict["log_probs"]
        log_probs_np = log_probs[0].cpu().numpy()  # [L, 33]
        probs_np = np.exp(log_probs_np)
    
    # 识别WT中已有C的位置
    existing_c_positions = [i for i, base in enumerate(wt_sequence, 1) if base == 'C']
    
    # 提取可设计位置的信息
    novel_c_probs = {}       # C的概率
    novel_c_log_probs = {}   # C的log概率
    original_base_probs = {} # 原始碱基的概率
    original_base_log_probs = {}  # 原始碱基的log概率
    delta_log_p = {}         # 突变代价: log P(C) - log P(original)
    position_info = {}       # 每个位置的详细信息
    
    for i, res_str in enumerate(encoded_residues):
        pos = int(''.join(filter(str.isdigit, res_str)))
        if pos not in fixed_positions and pos not in existing_c_positions:
            # 获取C的概率
            c_prob = float(probs_np[i, DC_IDX])
            c_log_prob = float(log_probs_np[i, DC_IDX])
            
            # 获取原始碱基的概率
            if pos <= len(wt_sequence):
                orig_base = wt_sequence[pos - 1]
                orig_idx = base_to_idx.get(orig_base)
                if orig_idx is not None:
                    orig_prob = float(probs_np[i, orig_idx])
                    orig_log_prob = float(log_probs_np[i, orig_idx])
                else:
                    orig_prob = 0.0
                    orig_log_prob = -float('inf')
            else:
                orig_base = '?'
                orig_prob = 0.0
                orig_log_prob = -float('inf')
            
            # 计算突变代价 (delta_log_p = log P(C) - log P(original))
            # 负值表示C的概率低于原始碱基（不利突变）
            # 正值表示C的概率高于原始碱基（有利突变）
            if orig_log_prob > -float('inf'):
                cost = c_log_prob - orig_log_prob
            else:
                cost = 0.0  # 如果原始碱基概率未知，不计算代价
            
            novel_c_probs[pos] = c_prob
            novel_c_log_probs[pos] = c_log_prob
            original_base_probs[pos] = orig_prob
            original_base_log_probs[pos] = orig_log_prob
            delta_log_p[pos] = cost
            
            position_info[pos] = {
                'original_base': orig_base,
                'c_prob': c_prob,
                'c_log_prob': c_log_prob,
                'original_prob': orig_prob,
                'original_log_prob': orig_log_prob,
                'delta_log_p': cost,
            }
    
    return {
        'novel_c_probs': novel_c_probs,
        'novel_c_log_probs': novel_c_log_probs,
        'original_base_probs': original_base_probs,
        'original_base_log_probs': original_base_log_probs,
        'delta_log_p': delta_log_p,
        'position_info': position_info,
        'existing_c_positions': existing_c_positions,
        'sequence_length': L,
    }


def generate_library_for_wt(wt: Dict, output_dir: Path, prob_threshold: float = 0.0) -> Dict:
    """
    为单个WT生成C-rich衍生物库
    
    新流程（CC距离优先）：
    1. 先用 CC 距离筛选位点（物理约束）
       - +1C: 突变位置到最近原有C的距离 <= 阈值
       - +2C: 两个突变位置之间的距离 <= 阈值
    2. 再用 NA-MPNN 计算 delta_log_p（能量评估）
       - 作为排序指标，不作为过滤条件
    """
    pdb_id = wt['pdb_id']
    pdb_path = CONFIG["pdb_cache"] / f"{pdb_id}.pdb"
    
    if not pdb_path.exists():
        # 尝试下载
        import urllib.request
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, pdb_path)
        except:
            return {"error": f"Could not download {pdb_id}"}
    
    fixed_positions = parse_fixed_pattern(wt['fixed_pattern'])
    original_seq = wt['sequence']
    
    # ===== 第一步：CC距离筛选（物理约束优先）=====
    dist_threshold = CONFIG.get("cc_pair_dist_threshold", 15.0)
    c1_coords = get_c1_coords(str(pdb_path))
    
    # 获取WT中原有C的位置
    existing_c_positions = [i+1 for i, b in enumerate(original_seq) if b == 'C']
    
    # 可设计位点 = 所有位点 - 固定位点
    all_positions = list(range(1, len(original_seq) + 1))
    designable_positions = [p for p in all_positions if p not in fixed_positions]
    
    # 可突变位点 = 可设计位点 - 原有C
    mutable_positions = [p for p in designable_positions if p not in existing_c_positions]
    
    # 筛选 +1C 候选位点（到最近原有C的距离 <= 阈值）
    valid_single_positions = []
    single_dist_info = {}  # 记录每个位点到最近C的距离
    
    for pos in mutable_positions:
        if pos not in c1_coords:
            continue
        dists = []
        for ec in existing_c_positions:
            if ec in c1_coords:
                d = float(np.linalg.norm(c1_coords[pos] - c1_coords[ec]))
                dists.append((ec, d))
        if dists:
            nearest_c, min_dist = min(dists, key=lambda x: x[1])
            if min_dist <= dist_threshold:
                valid_single_positions.append(pos)
                single_dist_info[pos] = {'nearest_c': nearest_c, 'dist': min_dist}
    
    # 筛选 +2C 候选组合（两个位点之间距离 <= 阈值）
    valid_double_pairs = []
    double_dist_info = {}  # 记录每对位点之间的距离
    
    for p1, p2 in combinations(mutable_positions, 2):
        if p1 not in c1_coords or p2 not in c1_coords:
            continue
        dist = float(np.linalg.norm(c1_coords[p1] - c1_coords[p2]))
        if dist <= dist_threshold:
            valid_double_pairs.append((p1, p2))
            double_dist_info[(p1, p2)] = dist
    
    # ===== 第二步：NA-MPNN 概率筛选 + 评分 =====
    # 计算所有通过距离筛选位点的 C 概率和 delta_log_p
    all_cc_filtered_positions = set(valid_single_positions)
    for p1, p2 in valid_double_pairs:
        all_cc_filtered_positions.add(p1)
        all_cc_filtered_positions.add(p2)
    
    delta_log_p = {}
    position_info = {}
    c_probs = {}
    
    if all_cc_filtered_positions:
        try:
            result = get_c_logits_for_structure(
                str(pdb_path), fixed_positions, original_seq, CONFIG["nampnn_temperature"]
            )
            delta_log_p = result.get('delta_log_p', {})
            position_info = result.get('position_info', {})
            # 提取 C 概率
            for pos in all_cc_filtered_positions:
                c_probs[pos] = position_info.get(pos, {}).get('c_prob', 0)
        except Exception as e:
            print(f"  Warning: NA-MPNN evaluation failed: {e}")
    
    # 应用概率阈值筛选 (Stage 2)
    prob_threshold = CONFIG.get("prob_threshold", 0.01)
    
    # 记录筛选前的数量
    n_cc_single = len(valid_single_positions)
    n_cc_double = len(valid_double_pairs)
    
    # +1C: 筛选 prob > threshold
    valid_single_positions = [p for p in valid_single_positions if c_probs.get(p, 0) > prob_threshold]
    
    # +2C: 两个位点的 prob 都 > threshold
    valid_double_pairs = [(p1, p2) for p1, p2 in valid_double_pairs 
                          if c_probs.get(p1, 0) > prob_threshold and c_probs.get(p2, 0) > prob_threshold]
    
    # 更新有效位点集合
    all_relevant_positions = set(valid_single_positions)
    for p1, p2 in valid_double_pairs:
        all_relevant_positions.add(p1)
        all_relevant_positions.add(p2)
    
    # 为了兼容性，构建 c_allowed 和 sorted_allowed
    c_allowed = {pos: c_probs.get(pos, 0) for pos in all_relevant_positions}
    sorted_allowed = sorted(c_allowed.items(), key=lambda x: x[1], reverse=True)
    allowed_positions = list(all_relevant_positions)
    
    # ===== 第三步：生成衍生物 =====
    derivatives = []
    
    # 统计信息
    n_total_mutable = len(mutable_positions)
    n_sat_single = n_total_mutable
    n_sat_double = n_total_mutable * (n_total_mutable - 1) // 2
    
    # +1C（已通过距离筛选）
    for pos in valid_single_positions:
        new_seq = list(original_seq)
        new_seq[pos - 1] = 'C'
        new_seq = ''.join(new_seq)
        
        pos_delta = delta_log_p.get(pos, 0.0)
        pos_info = position_info.get(pos, {})
        dist_info = single_dist_info.get(pos, {})
        
        derivatives.append({
            'sequence': new_seq,
            'mutation': f"+1C@{pos}",
            'c_positions': [pos],
            'type': '+1C',
            'c_prob': c_allowed.get(pos, 0),
            # Log-P评估字段
            'delta_log_p': pos_delta,
            'original_base': pos_info.get('original_base', '?'),
            'original_prob': pos_info.get('original_prob', 0),
            # 空间约束字段
            'nearest_c': dist_info.get('nearest_c'),
            'nearest_c_dist': dist_info.get('dist'),
        })
    
    # +2C（已通过距离筛选）
    for pos1, pos2 in valid_double_pairs:
        new_seq = list(original_seq)
        new_seq[pos1 - 1] = 'C'
        new_seq[pos2 - 1] = 'C'
        new_seq = ''.join(new_seq)
        
        delta1 = delta_log_p.get(pos1, 0.0)
        delta2 = delta_log_p.get(pos2, 0.0)
        combined_delta = delta1 + delta2
        
        pos1_info = position_info.get(pos1, {})
        pos2_info = position_info.get(pos2, {})
        pair_dist = double_dist_info.get((pos1, pos2))
        
        derivatives.append({
            'sequence': new_seq,
            'mutation': f"+2C@{pos1},{pos2}",
            'c_positions': [pos1, pos2],
            'type': '+2C',
            'c_prob_sum': c_allowed.get(pos1, 0) + c_allowed.get(pos2, 0),
            # Log-P评估字段
            'delta_log_p': combined_delta,
            'delta_log_p_detail': {
                str(pos1): delta1,
                str(pos2): delta2,
            },
            'original_bases': {
                str(pos1): pos1_info.get('original_base', '?'),
                str(pos2): pos2_info.get('original_base', '?'),
            },
            # 空间约束字段
            'pair_dist': pair_dist,
        })
    
    # 按delta_log_p排序（从高到低，高分=更有利的突变）
    derivatives.sort(key=lambda x: x['delta_log_p'], reverse=True)
    
    # 保存
    wt_output_dir = output_dir / pdb_id
    wt_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建位点详细信息
    position_details = []
    for pos in sorted(all_relevant_positions):
        info = position_info.get(pos, {})
        dist_info = single_dist_info.get(pos, {})
        position_details.append({
            'pos': pos,
            'c_prob': info.get('c_prob', 0),
            'original_base': info.get('original_base', '?'),
            'original_prob': info.get('original_prob', 0),
            'delta_log_p': info.get('delta_log_p', 0),
            'nearest_c_dist': dist_info.get('dist'),
        })
    
    # 统计信息 (两阶段筛选)
    n_single = len(valid_single_positions)
    n_double = len(valid_double_pairs)
    # Stage 1: CC距离筛选
    n_cc_filtered_single = n_sat_single - n_cc_single
    n_cc_filtered_double = n_sat_double - n_cc_double
    # Stage 2: 概率筛选
    n_prob_filtered_single = n_cc_single - n_single
    n_prob_filtered_double = n_cc_double - n_double
    # 总筛选
    n_filtered_single = n_sat_single - n_single
    n_filtered_double = n_sat_double - n_double
    
    library_data = {
        'pdb_id': pdb_id,
        'original_sequence': original_seq,
        'fixed_positions': fixed_positions,
        'existing_c_positions': existing_c_positions,
        'mutable_positions': mutable_positions,
        'position_details': position_details,
        'n_mutable': n_total_mutable,
        'n_single': n_single,
        'n_double': n_double,
        'n_total': len(derivatives),
        # 两阶段筛选统计
        'filtering_stats': {
            # 饱和突变
            'saturated_single': n_sat_single,
            'saturated_double': n_sat_double,
            'saturated_total': n_sat_single + n_sat_double,
            # Stage 1: CC距离筛选
            'cc_dist_threshold': dist_threshold,
            'after_cc_single': n_cc_single,
            'after_cc_double': n_cc_double,
            'after_cc_total': n_cc_single + n_cc_double,
            # Stage 2: 概率筛选
            'prob_threshold': prob_threshold,
            'after_prob_single': n_single,
            'after_prob_double': n_double,
            'after_prob_total': n_single + n_double,
            # 总减少率
            'reduction_rate': (n_filtered_single + n_filtered_double) / (n_sat_single + n_sat_double) if (n_sat_single + n_sat_double) > 0 else 0,
        },
        # Log-P统计信息
        'delta_log_p_stats': {
            'mean': float(np.mean([d['delta_log_p'] for d in derivatives])) if derivatives else 0,
            'min': float(np.min([d['delta_log_p'] for d in derivatives])) if derivatives else 0,
            'max': float(np.max([d['delta_log_p'] for d in derivatives])) if derivatives else 0,
        },
        'derivatives': derivatives,
    }
    
    with open(wt_output_dir / "library.json", 'w') as f:
        json.dump(library_data, f, indent=2)
    
    # 生成AF3 JSON
    af3_jobs_dir = wt_output_dir / "af3_jobs"
    af3_jobs_dir.mkdir(exist_ok=True)
    
    for d in derivatives:
        job_name = f"{pdb_id}_{d['mutation'].replace(',', '_').replace('@', '')}"
        af3_json = [{
            "name": job_name,
            "modelSeeds": [42],
            "sequences": [{"dnaSequence": {"sequence": d['sequence']}}],
            "dialect": "alphafoldserver",
            "version": 1
        }]
        with open(af3_jobs_dir / f"{job_name}.json", 'w') as f:
            json.dump(af3_json, f, indent=2)
    
    return library_data


def run_library_stage():
    """运行Library生成阶段"""
    print("=" * 80)
    print("STAGE 1: C-Rich Library Generation (Two-Stage Filtering)")
    print(f"  Stage 1 (Physical):    CC Distance ≤ {CONFIG.get('cc_pair_dist_threshold', 15.0)}Å")
    print(f"  Stage 2 (Statistical): P(C) > {CONFIG.get('prob_threshold', 0.01)}")
    print("=" * 80)
    
    wt_list = load_wt_dataset(CONFIG["dataset_csv"])
    print(f"Loaded {len(wt_list)} WT structures\n")
    
    output_dir = CONFIG["output_root"] / CONFIG["library_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_derivatives = 0
    total_saturated = 0
    total_filtered = 0
    all_results = []
    all_delta_log_p = []
    
    for i, wt in enumerate(wt_list, 1):
        print(f"[{i}/{len(wt_list)}] {wt['pdb_id']}: {wt['sequence'][:30]}...")
        
        result = generate_library_for_wt(wt, output_dir, CONFIG["prob_threshold"])
        
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
        else:
            # 显示统计信息
            stats = result.get('delta_log_p_stats', {})
            filter_stats = result.get('filtering_stats', {})
            n_sat = filter_stats.get('saturated_total', 0)
            n_after_prob = filter_stats.get('after_prob_total', 0)
            reduction = filter_stats.get('reduction_rate', 0) * 100
            
            delta_info = f"ΔLogP: [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}]"
            
            print(f"  ✓ {result['n_single']} +1C + {result['n_double']} +2C = {result['n_total']} (sat:{n_sat}, -{reduction:.0f}%) | {delta_info}")
            total_derivatives += result['n_total']
            total_saturated += n_sat
            total_filtered += (n_sat - n_after_prob)  # 计算总减少量
            all_results.append(result)
            
            # 收集所有delta_log_p
            for d in result.get('derivatives', []):
                all_delta_log_p.append(d.get('delta_log_p', 0))
    
    # 保存汇总
    overall_reduction = (total_filtered / total_saturated * 100) if total_saturated > 0 else 0
    
    # 计算两阶段筛选的详细统计
    total_after_cc = sum(r.get('filtering_stats', {}).get('after_cc_total', 0) for r in all_results)
    
    summary = {
        'n_wt': len(wt_list),
        'total_derivatives': total_derivatives,
        'total_saturated': total_saturated,
        'total_after_cc': total_after_cc,
        'overall_reduction_rate': overall_reduction,
        # 两阶段筛选配置
        'filtering_config': {
            'cc_dist_threshold': CONFIG.get('cc_pair_dist_threshold', 15.0),
            'prob_threshold': CONFIG.get('prob_threshold', 0.01),
        },
        # 全局delta_log_p统计
        'global_delta_log_p_stats': {
            'mean': float(np.mean(all_delta_log_p)) if all_delta_log_p else 0,
            'std': float(np.std(all_delta_log_p)) if all_delta_log_p else 0,
            'min': float(np.min(all_delta_log_p)) if all_delta_log_p else 0,
            'max': float(np.max(all_delta_log_p)) if all_delta_log_p else 0,
            'median': float(np.median(all_delta_log_p)) if all_delta_log_p else 0,
        },
        'results': [{k: v for k, v in r.items() if k != 'derivatives'} for r in all_results]
    }
    with open(output_dir / "library_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    cc_reduction = (1 - total_after_cc / total_saturated) * 100 if total_saturated > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Library Generation Complete!")
    print(f"  Total WT: {len(wt_list)}")
    print(f"\n  Two-Stage Filtering Results:")
    print(f"    Saturated Mutations:     {total_saturated}")
    print(f"    After CC Distance (≤{CONFIG.get('cc_pair_dist_threshold', 15.0)}Å): {total_after_cc} (−{cc_reduction:.1f}%)")
    print(f"    After P(C) > {CONFIG.get('prob_threshold', 0.01)}:       {total_derivatives} (−{overall_reduction:.1f}% total)")
    if all_delta_log_p:
        gstats = summary['global_delta_log_p_stats']
        print(f"\n  Delta Log-P Statistics (NA-MPNN Energy Evaluation):")
        print(f"    Mean:   {gstats['mean']:.3f}")
        print(f"    Std:    {gstats['std']:.3f}")
        print(f"    Range:  [{gstats['min']:.3f}, {gstats['max']:.3f}]")
        print(f"    Median: {gstats['median']:.3f}")
    print(f"\n  Output: {output_dir}")
    print("=" * 80)
    
    return summary


# ============================================================================
# Stage 2: AF3 Folding
# ============================================================================

def _require_af3_path(key: str, env_name: str) -> str:
    value = str(CONFIG.get(key, "")).strip()
    if not value:
        raise RuntimeError(
            f"Missing {env_name}. Export {env_name} before the fold stage "
            "(AlphaFold3 source, model parameters, and databases)."
        )
    return value


def run_af3_prediction(json_path: Path, output_dir: Path) -> Dict:
    """运行单个AF3预测"""
    conda_env = CONFIG["af3_conda_env"]
    af3_dir = _require_af3_path("af3_dir_path", "AF3_DIR")
    model_dir = _require_af3_path("af3_model_dir", "AF3_MODEL_DIR")
    db_dir = _require_af3_path("af3_db_dir", "AF3_DB_DIR")
    conda_base = os.environ.get("CONDA_BASE", "").strip() or "$HOME/miniconda3"

    cmd = f"""
    if [ -z "${{CONDA_BASE:-}}" ]; then
        CONDA_BASE="$(conda info --base 2>/dev/null || echo {conda_base})"
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh" && \
    conda activate {conda_env} && \
    cd {af3_dir} && \
    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" python run_alphafold.py \
        --json_path={json_path} \
        --output_dir={output_dir} \
        --model_dir={model_dir} \
        --db_dir={db_dir} \
        --run_data_pipeline=False \
        --flash_attention_implementation=xla \
        --num_diffusion_samples={CONFIG['num_diffusion_samples']}
    """
    
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        return {"success": result.returncode == 0, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_fold_stage():
    """运行AF3折叠阶段"""
    print("=" * 80)
    print("STAGE 2: AlphaFold3 Structure Prediction")
    print("=" * 80)
    
    library_dir = CONFIG["output_root"] / CONFIG["library_dir"]
    af3_output_dir = CONFIG["output_root"] / CONFIG["af3_dir"]
    af3_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有待预测的任务
    jobs = []
    for wt_dir in library_dir.iterdir():
        if wt_dir.is_dir():
            af3_jobs = wt_dir / "af3_jobs"
            if af3_jobs.exists():
                for json_file in af3_jobs.glob("*.json"):
                    jobs.append({
                        'json_path': json_file,
                        'pdb_id': wt_dir.name,
                        'job_name': json_file.stem
                    })
    
    print(f"Found {len(jobs)} jobs to predict\n")
    
    success_count = 0
    skip_count = 0
    for i, job in enumerate(jobs, 1):
        output_path = af3_output_dir / job['pdb_id'] / job['job_name']
        
        # 跳过已完成的 (AF3 输出在子目录中)
        existing_cifs = list(output_path.rglob("*_model.cif"))
        if existing_cifs:
            skip_count += 1
            success_count += 1
            if skip_count <= 10 or skip_count % 50 == 0:
                print(f"[{i}/{len(jobs)}] {job['job_name']} - SKIP (exists)")
            continue
        
        print(f"[{i}/{len(jobs)}] {job['job_name']} - Predicting...")
        output_path.mkdir(parents=True, exist_ok=True)
        
        result = run_af3_prediction(job['json_path'], output_path)
        
        if result.get("success"):
            print(f"  ✓ Complete")
            success_count += 1
        else:
            print(f"  ✗ Failed: {result.get('error', result.get('stderr', '')[:100])}")
    
    print(f"\n{'='*80}")
    print(f"AF3 Folding Complete!")
    print(f"  Success: {success_count}/{len(jobs)}")
    print(f"  Output: {af3_output_dir}")
    print("=" * 80)


# ============================================================================
# Stage 3: Screening
# ============================================================================

def count_structure_clashes(cif_path: str, clash_threshold: float = 2.0) -> dict:
    """
    检测结构中的原子重叠（close contacts/clashes）
    
    用于排除AF3预测失败产生的artifact结构（碱基变形、重叠）。
    
    Args:
        cif_path: CIF文件路径
        clash_threshold: 距离阈值(Å)，小于此值的非键合原子对被视为clash
        
    Returns:
        dict: {
            'n_clashes': int,       # clash数量
            'n_severe': int,        # 严重clash数量 (< 1.5Å)
            'min_dist': float,      # 最小clash距离
            'has_severe': bool,     # 是否有严重clash (min_dist < 1.5Å)
        }
    """
    try:
        from Bio.PDB import MMCIFParser
        from Bio.PDB.NeighborSearch import NeighborSearch
        
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("model", cif_path)
        atoms = list(structure.get_atoms())
        
        if len(atoms) == 0:
            return {'n_clashes': 0, 'n_severe': 0, 'min_dist': 999.0, 'has_severe': False}
        
        ns = NeighborSearch(atoms)
        
        clashes = []
        checked = set()
        
        for atom in atoms:
            neighbors = ns.search(atom.get_coord(), clash_threshold)
            for neighbor in neighbors:
                if neighbor == atom:
                    continue
                # 跳过同一残基内的原子（键合原子）
                if atom.get_parent() == neighbor.get_parent():
                    continue
                
                res1 = atom.get_parent()
                res2 = neighbor.get_parent()
                res1_id = res1.get_id()[1]
                res2_id = res2.get_id()[1]
                
                # 跳过相邻残基的骨架连接原子
                if abs(res1_id - res2_id) == 1:
                    backbone_atoms = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"}
                    if atom.get_name() in backbone_atoms and neighbor.get_name() in backbone_atoms:
                        continue
                
                # 避免重复计数
                pair_key = tuple(sorted([id(atom), id(neighbor)]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                
                dist = np.linalg.norm(atom.get_coord() - neighbor.get_coord())
                if dist < clash_threshold:
                    clashes.append(dist)
        
        n_severe = sum(1 for d in clashes if d < 1.5)
        min_dist = min(clashes) if clashes else 999.0
        
        return {
            'n_clashes': len(clashes),
            'n_severe': n_severe,
            'min_dist': min_dist,
            'has_severe': min_dist < 1.5
        }
    except Exception as e:
        # 如果检测失败，假设没有clash
        return {'n_clashes': -1, 'n_severe': -1, 'min_dist': 999.0, 'has_severe': False}


def run_screen_stage():
    """
    运行筛选阶段
    
    筛选顺序（重要！）：
    1. Layer 1 (Structure Quality): 首先排除 artifact 结构 (min_clash >= 1.4Å)
    2. Layer 2 (G4 Detection): DSSR 检测 G4 核心
    3. Layer 3 (Geometry Filter): CC 几何条件筛选
    4. Ranking: BPO 欧式距离
    
    这样的顺序确保：
    - artifact 结构不会浪费 DSSR 分析时间
    - 只有高质量结构才进入后续筛选
    """
    print("=" * 80)
    print("STAGE 3: Quality → G4 → Geometry Hierarchical Screening")
    print("=" * 80)
    
    from modules.dssr_analyzer import DSSRAnalyzer
    
    af3_dir = CONFIG["output_root"] / CONFIG["af3_dir"]
    screen_dir = CONFIG["output_root"] / CONFIG["screen_dir"]
    screen_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = DSSRAnalyzer()
    
    # ========================================================================
    # Step 1: 收集所有结构文件（遍历所有 AF3 samples）
    # ========================================================================
    structures = []
    for cif_file in af3_dir.rglob("*_model.cif"):
        # 只选择 seed-*_sample-* 目录下的文件（独立样本）
        if "sample-" not in cif_file.parent.name:
            continue  # 跳过副本
        
        sample_dir = cif_file.parent
        job_name_lower_dir = sample_dir.parent
        job_dir = job_name_lower_dir.parent
        pdb_dir = job_dir.parent
        
        # 读取 pLDDT 数据
        conf_file = cif_file.with_name(cif_file.stem.replace('_model', '_confidences') + '.json')
        plddt_mean = 0.0
        ranking_score = 0.0
        if conf_file.exists():
            try:
                with open(conf_file) as f:
                    conf_data = json.load(f)
                    if 'atom_plddts' in conf_data:
                        plddt_mean = float(np.mean(conf_data['atom_plddts']))
            except: pass
        
        # 读取 ranking_score
        summ_file = cif_file.with_name(cif_file.stem.replace('_model', '_summary_confidences') + '.json')
        if summ_file.exists():
            try:
                with open(summ_file) as f:
                    summ_data = json.load(f)
                    ranking_score = summ_data.get('ranking_score', 0.0)
            except: pass
        
        structures.append({
            'path': cif_file,
            'pdb_id': pdb_dir.name,
            'job_name': job_dir.name,
            'sample': sample_dir.name,
            'plddt_mean': plddt_mean,
            'ranking_score': ranking_score,
        })
    
    print(f"Found {len(structures)} structures (all AF3 samples) to screen\n")
    
    # ========================================================================
    # Layer 1: Structure Quality Check (Clash Detection) - 首先执行！
    # ========================================================================
    print("Layer 1: Structure Quality Check (Clash Detection)...")
    quality_passed = []
    quality_failed = []
    
    for i, struct in enumerate(structures, 1):
        if i % 100 == 0:
            print(f"  Checked quality {i}/{len(structures)}...")
        
        clash_info = count_structure_clashes(str(struct['path']))
        struct['clash_info'] = clash_info
        
        if clash_info['min_dist'] >= 1.4:
            quality_passed.append(struct)
        else:
            quality_failed.append(struct)
    
    print(f"  ✓ Quality passed: {len(quality_passed)} / {len(structures)}")
    print(f"  ✗ Quality failed (artifact): {len(quality_failed)}")
    
    # ========================================================================
    # Layer 2 & 3: 只对通过质量检测的结构进行 DSSR 分析
    # ========================================================================
    print(f"\nLayer 2 & 3: DSSR Analysis (G4 + Geometry) on quality-passed structures...")
    
    all_results = []
    for i, struct in enumerate(quality_passed, 1):
        if i % 50 == 0:
            print(f"  Analyzed {i}/{len(quality_passed)}...")
        
        try:
            result = analyzer.analyze_with_frames(str(struct['path']))
            result['pdb_id'] = struct['pdb_id']
            result['job_name'] = struct['job_name']
            result['sample_name'] = struct['sample']
            result['structure_path'] = str(struct['path'])
            result['plddt_mean'] = struct.get('plddt_mean', 0.0)
            result['ranking_score'] = struct.get('ranking_score', 0.0)
            result['clash_info'] = struct['clash_info']
            all_results.append(result)
        except Exception as e:
            print(f"  ⚠ Error analyzing {struct['job_name']}: {e}")
    
    # 按序列聚合（取最高 pLDDT 的结构代表该序列）
    seq_data = defaultdict(list)
    for r in all_results:
        seq = r.get('sequence', 'unknown')
        seq_data[seq].append(r)
    
    # ========================================================================
    # 层层筛选流程（注意：Layer 1 已在上面完成）
    # Layer 1: 结构质量 - 已完成（只有 quality_passed 进入这里）
    # Layer 2: G4 存在 (≥ 2 tetrads)
    # Layer 3: C-C 几何条件 (Stagger < 1.0Å, C1'-C1' 8.5-12.5Å)
    # 最终: 按 BPO 欧式距离排序
    # ========================================================================
    
    all_candidates = []
    for seq, rs in seq_data.items():
        rs = [r for r in rs if r is not None]
        if not rs:
            continue
        
        # === 核心筛选逻辑：在质量已通过的结构中，检查 G4 和几何条件 ===
        valid_models_for_seq = []
        
        for r in rs:
            # 此时所有结构都已通过 Layer 1 (质量检测)
            clash_info = r.get('clash_info', {})
            min_clash_dist = clash_info.get('min_dist', 999.0)
            
            # Layer 2: 检查当前结构的 G4 (要求至少 2 层)
            g4_info = r.get('g_quadruplex', {})
            n_tetrads = g4_info.get('n_tetrads', 0)
            has_g4 = n_tetrads >= 2 # 必须有稳定的G4核心
            
            # Layer 3: 检查当前结构的 CC 对 (排除堆叠和相邻，Stagger < 1.0)
            frame_analysis = r.get('cc_frame_analysis', {})
            cc_pairs = frame_analysis.get('cc_frame_pairs', [])
            
            best_pair_in_model = None
            for p in cc_pairs:
                # 使用 has_pairing_potential 字段（已排除堆叠和相邻的配对）
                if p.get('has_pairing_potential', False):
                    bpo_score = p.get('bpo_score', 999)
                    if best_pair_in_model is None or bpo_score < best_pair_in_model.get('bpo_score', 999):
                        best_pair_in_model = p.copy()
            
            # 只有同时满足 G4 + 几何 的结构才算有效（质量已在 Layer 1 过滤）
            if has_g4 and best_pair_in_model:
                r_copy = r.copy()
                r_copy['best_planar_pair'] = best_pair_in_model
                r_copy['n_tetrads'] = n_tetrads
                r_copy['min_clash_dist'] = min_clash_dist
                valid_models_for_seq.append(r_copy)
        
        # ====================================================================
        # 结果汇总
        # ====================================================================
        # 该序列是否通过筛选（至少有一个模型全通过 Layer 2 & 3）
        seq_passed = len(valid_models_for_seq) > 0
        
        if seq_passed:
            # 从所有有效模型中选出 BPO 距离最小的那个
            best_model = min(valid_models_for_seq, key=lambda x: x['best_planar_pair']['bpo_score'])
            
            all_candidates.append({
                'sequence': seq,
                'pdb_id': best_model.get('pdb_id', 'unknown'),
                'job_name': best_model.get('job_name', 'unknown'),
                'sample_name': best_model.get('sample_name', 'unknown'),
                'g4_valid': True,
                'n_tetrads': best_model['n_tetrads'],
                'has_planar_pair': True,
                'best_planar_pair': best_model['best_planar_pair'],
                'min_clash_dist': best_model.get('min_clash_dist', 999.0),
                'plddt_mean': best_model.get('plddt_mean', 0.0),
                'n_cytosines': best_model.get('cc_analysis', {}).get('n_cytosines', 0),
                'c_positions': best_model.get('cc_analysis', {}).get('c_positions', []),
                'min_cc_distance': best_model.get('cc_analysis', {}).get('min_distance', 99.0),
                'n_models': len(rs),
                'n_valid_models': len(valid_models_for_seq)
            })
        else:
            # 记录未通过的原因，用于统计
            # 注意：这里的 rs 都是已通过 Layer 1 (质量检测) 的结构
            any_g4 = any(r.get('g_quadruplex', {}).get('detected') for r in rs)
            any_geometry = any(
                any(p.get('has_pairing_potential', False) for p in r.get('cc_frame_analysis', {}).get('cc_frame_pairs', []))
                for r in rs
            )
            all_candidates.append({
                'sequence': seq,
                'pdb_id': rs[0].get('pdb_id', 'unknown'),
                'job_name': rs[0].get('job_name', 'unknown'),
                'g4_valid': any_g4,
                'has_planar_pair': any_geometry,
                'n_models': len(rs)
            })
    
    # ========================================================================
    # 层层筛选统计
    # ========================================================================
    total_structures = len(structures)
    total_sequences = len(all_candidates)
    
    # 严格筛选：只有真正通过 seq_passed (即 valid_models_for_seq > 0) 的才算 layer3_pass
    # 这样可以排除那些 G4 和 CC 对不在同一个模型中的无效数据
    layer2_pass = [c for c in all_candidates if c.get('g4_valid') and c.get('n_valid_models', 0) > 0]
    layer3_pass = [c for c in layer2_pass if c.get('has_planar_pair', False)]
    
    # 最终候选按 BPO 欧式距离排序（越小越好）
    final_candidates = sorted(layer3_pass, key=lambda x: x.get('best_planar_pair', {}).get('bpo_score', 999))
    
    # 显示结果
    print(f"\n{'='*140}")
    print("HIERARCHICAL SCREENING RESULTS")
    print("=" * 140)
    print(f"  Layer 1 (Structure Quality):  {len(quality_passed):>4} / {total_structures} samples passed")
    print(f"    Criteria: min_clash_dist >= 1.4Å (排除artifact结构)")
    print(f"  Layer 2 (G4 Detection):       {len(layer2_pass):>4} / {total_sequences} sequences passed")
    print(f"    Criteria: ≥ 2 G-tetrads detected by DSSR")
    print(f"  Layer 3 (Geometry Filter):    {len(layer3_pass):>4} / {len(layer2_pass)} passed")
    print(f"    Criteria: |Stagger| < 1.0Å, C1'-C1' 8.5-12.5Å")
    print(f"\n  FINAL CANDIDATES: {len(final_candidates)} (sorted by BPO Euclidean Distance, lower is better)")
    print("=" * 140)
    
    if final_candidates:
        print(f"\n{'Rank':<5} {'Design':<20} {'Sequence':<24} {'Stagger':<8} {'BPO_Dist':<10} {'B/P/O':<22} {'Clash':<8} {'pLDDT':<8}")
        print("-" * 140)
        
        for i, c in enumerate(final_candidates[:50], 1):
            seq_disp = c['sequence'][:21] + "..." if len(c['sequence']) > 21 else c['sequence']
            design_name = c['job_name'][:18] + ".." if len(c['job_name']) > 18 else c['job_name']
            
            bp = c.get('best_planar_pair', {})
            if bp:
                b, p, o = bp.get('buckle', 0), bp.get('propeller', 0), bp.get('opening', 0)
                bpo_dist = bp.get('bpo_score', 999)
                stagger = bp.get('stagger', 0)
                angle_str = f"{b:.1f}/{p:.1f}/{o:.1f}"
            else:
                bpo_dist = 999
                stagger = 0
                angle_str = "-"
            
            min_clash = c.get('min_clash_dist', 999.0)
            clash_str = "clean" if min_clash >= 999 else f"{min_clash:.2f}Å"
            
            print(f"{i:<5} {design_name:<20} {seq_disp:<24} {stagger:>6.2f}Å  {bpo_dist:>8.1f}  {angle_str:<22} {clash_str:<8} {c['plddt_mean']:>6.1f}")
    else:
        print("\n  ⚠ No candidates passed all filters!")
        print(f"\n  Candidates that passed G4 but failed Stagger filter (top 10 by pLDDT):")
        g4_only = sorted(layer1_pass, key=lambda x: x['plddt_mean'], reverse=True)[:10]
        for i, c in enumerate(g4_only, 1):
            bp = c.get('best_frame_pair', {})
            if bp:
                b, p, o = bp.get('buckle', 0), bp.get('propeller', 0), bp.get('opening', 0)
                angle_str = f"B={b:.0f}, P={p:.0f}, O={o:.0f}"
            else:
                angle_str = "No C-C pair"
            print(f"    {i}. {c['job_name']}: pLDDT={c['plddt_mean']:.1f}, {angle_str}")
    
    # 统计
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total AF3 samples: {total_structures}")
    print(f"  Unique sequences: {total_sequences}")
    print(f"  ├─ Layer 1 (Quality): {len(quality_passed)} samples passed ({len(quality_passed)/total_structures*100:.1f}%)")
    print(f"  ├─ Layer 2 (G4): {len(layer2_pass)} sequences passed ({len(layer2_pass)/total_sequences*100:.1f}%)" if total_sequences > 0 else "")
    print(f"  └─ Layer 3 (Geometry): {len(layer3_pass)} passed ({len(layer3_pass)/len(layer2_pass)*100:.1f}% of L2)" if layer2_pass else "")
    
    if final_candidates:
        best = final_candidates[0]
        bp = best.get('best_planar_pair', {})
        bpo_dist = bp.get('bpo_score', 999) if bp else 999
        min_clash = best.get('min_clash_dist', 999.0)
        print(f"\n  🏆 BEST CANDIDATE (by lowest BPO Distance):")
        print(f"     Design: {best['job_name']} (PDB: {best['pdb_id']}, Sample: {best.get('sample_name', '-')})")
        print(f"     Sequence: {best['sequence']}")
        print(f"     G4: {best.get('n_tetrads', 0)} tetrads, pLDDT: {best['plddt_mean']:.2f}")
        print(f"     Structure Quality: {'clean' if min_clash >= 999 else f'min_clash={min_clash:.2f}Å'}")
        if bp:
            print(f"     Stagger: {bp.get('stagger', 0):.2f}Å")
            print(f"     BPO Distance: {bpo_dist:.1f} (B={bp['buckle']:.1f}°, P={bp['propeller']:.1f}°, O={bp['opening']:.1f}°)")
            print(f"     N3-N3 distance: {bp.get('n3_distance', 0):.2f}Å")
    
    # 更新 candidates 和 valid_candidates 变量以便后续保存
    candidates = all_candidates
    valid_candidates = final_candidates
    
    # 保存结果
    results_path = screen_dir / "screening_results.json"
    with open(results_path, 'w') as f:
        json.dump(candidates, f, indent=2, default=str)
    
    # 保存Top候选
    top_path = screen_dir / "top_candidates.json"
    with open(top_path, 'w') as f:
        json.dump(valid_candidates[:100], f, indent=2, default=str)
    
    # 保存为CSV (层层筛选后的最终候选，按 BPO 欧式距离排序)
    csv_path = screen_dir / "screening_results.csv"
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Rank', 'Design', 'PDB', 'Sample', 'Sequence', 'CC_Pair', 'C1p_Dist', 'Stagger', 'BPO_Dist',
                'Buckle', 'Propeller', 'Opening', 'N3_Dist', 'Is_Stacking', 'Is_Adjacent',
                'Min_Clash_Dist', 'pLDDT', 'G4', 'Tetrads'
            ])
            for i, c in enumerate(valid_candidates, 1):
                bp = c.get('best_planar_pair', {})
                bpo_dist = bp.get('bpo_score', 999) if bp else 999
                c1p_dist = bp.get('c1p_distance') if bp else None
                min_clash = c.get('min_clash_dist', 999.0)
                
                # 获取配对的C位置
                if bp:
                    r1, r2 = bp.get('residue1', ('?', '?')), bp.get('residue2', ('?', '?'))
                    cc_pair = f"C{r1[1]}-C{r2[1]}"
                else:
                    cc_pair = "-"
                
                writer.writerow([
                    i, c.get('job_name', 'unknown'), c.get('pdb_id', 'unknown'), c.get('sample_name', '-'), c.get('sequence', ''),
                    cc_pair,
                    f"{c1p_dist:.2f}" if c1p_dist else "-",
                    f"{bp.get('stagger', 0):.2f}" if bp else "-",
                    f"{bpo_dist:.1f}",
                    f"{bp.get('buckle', 0):.1f}" if bp else "-",
                    f"{bp.get('propeller', 0):.1f}" if bp else "-",
                    f"{bp.get('opening', 0):.1f}" if bp else "-",
                    f"{bp.get('n3_distance', c.get('min_cc_distance', 99)):.2f}" if bp else f"{c.get('min_cc_distance', 99):.2f}",
                    'Y' if bp and bp.get('is_stacking', False) else 'N',
                    'Y' if bp and bp.get('is_adjacent', False) else 'N',
                    f"{min_clash:.2f}" if min_clash < 999 else "clean",
                    f"{c.get('plddt_mean', 0.0):.2f}",
                    'Y' if c.get('g4_valid', False) else 'N',
                    c.get('n_tetrads', 0)
                ])
        print(f"  CSV summary saved: {csv_path}")
    except Exception as e:
        print(f"  Warning: Could not save CSV: {e}")
    
    print(f"\n  Results saved: {results_path}")
    print(f"  Top candidates: {top_path}")
    print("=" * 80)
    
    return candidates


# ============================================================================
# Main
# ============================================================================

def run_status_stage():
    """显示流水线当前状态"""
    print("=" * 80)
    print("PIPELINE STATUS")
    print("=" * 80)
    
    library_dir = CONFIG["output_root"] / CONFIG["library_dir"]
    af3_dir = CONFIG["output_root"] / CONFIG["af3_dir"]
    screen_dir = CONFIG["output_root"] / CONFIG["screen_dir"]
    
    # 1. Library 状态
    total_designs = 0
    wt_count = 0
    if library_dir.exists():
        for wt_dir in library_dir.iterdir():
            if wt_dir.is_dir():
                lib_json = wt_dir / "library.json"
                if lib_json.exists():
                    with open(lib_json, 'r') as f:
                        data = json.load(f)
                        total_designs += data.get('n_total', 0)
                        wt_count += 1
    
    print(f"Library Status:")
    print(f"  - WT Structures: {wt_count}")
    print(f"  - Total Designs: {total_designs}")
    
    # 2. Folding 状态
    folded_count = 0
    if af3_dir.exists():
        for cif in af3_dir.rglob("*_model.cif"):
            folded_count += 1
    
    fold_pct = (folded_count / total_designs * 100) if total_designs > 0 else 0
    print(f"\nFolding Status (AF3):")
    print(f"  - Completed: {folded_count} / {total_designs} ({fold_pct:.1f}%)")
    
    # 3. Screening 状态
    screened_count = 0
    results_json = screen_dir / "screening_results.json"
    if results_json.exists():
        with open(results_json, 'r') as f:
            data = json.load(f)
            screened_count = len(data)
    
    print(f"\nScreening Status (G4 + Stagger):")
    print(f"  - Analyzed Unique Designs: {screened_count}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="pH-Responsive NA Design Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage", 
        default="all",
        help="Stages to run: all, library, fold, screen, status, or comma-separated"
    )
    parser.add_argument(
        "--output",
        help="Override output directory"
    )
    
    args = parser.parse_args()
    
    if args.output:
        CONFIG["output_root"] = Path(args.output)
    
    CONFIG["output_root"].mkdir(parents=True, exist_ok=True)
    
    stages = args.stage.lower().split(',')
    
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("pH-Responsive Nucleic Acid Design - Full Pipeline")
    print("=" * 80)
    print(f"Output directory: {CONFIG['output_root']}")
    print(f"Stages to run: {stages}")
    print("=" * 80 + "\n")
    
    if "all" in stages or "library" in stages:
        run_library_stage()
        print()
    
    if "all" in stages or "fold" in stages:
        run_fold_stage()
        print()
    
    if "all" in stages or "screen" in stages:
        run_screen_stage()
        print()
    
    if "status" in stages:
        run_status_stage()
        print()
    
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
