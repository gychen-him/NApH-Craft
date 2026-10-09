#!/usr/bin/env python
"""
pH-Responsive Nucleic Acid Design - Unified Entry Point
========================================================

pH敏感核酸设计流程，用于筛选在弱酸条件下能形成C·C+配对的G4突变体。

工作流（完整流程）：
  1. library: NA-MPNN获取C概率 → 生成C-rich衍生物库（+1C, +2C组合）
  2. fold:    使用AF3折叠序列（跳过MSA加速）
  3. screen:  使用DSSR分析G4结构 + 测量CC距离 → 筛选pH响应候选

核心原理：
  Stage 1 - Library Generation (NA-MPNN):
    - 对WT结构运行NA-MPNN的score方法，获取每个可设计位置对C的条件概率
    - 筛选probability > 0的位置作为"允许C突变"的位置
    - 系统生成+1C（单位点）和+2C（双位点）组合，构建候选库
    
  Stage 2 - Structure Prediction (AlphaFold3):
    - 对候选序列进行结构预测
    - 跳过MSA搜索以加速（纯核酸预测不依赖MSA）
    
  Stage 3 - Screening (DSSR + CC Distance):
    - 使用DSSR验证G-quadruplex结构完整性（G-tetrad检测）
    - 测量C-C距离：<4Å可形成C·C+ H-bond，4-6Å具有潜力
    - GCNS评分用于排序候选序列

Usage:
  # 生成C-rich衍生物库
  python run.py library --pdb xxx.pdb --fixed "2,3,4,7,8,9"
  
  # 或使用旧的design命令（兼容）
  python run.py design --pdb xxx.pdb --fixed "2,3,4,7,8,9" --n_c 2

  # AF3折叠（跳过MSA搜索加速）
  python run.py fold --sequences design_results.json --output af3_results/
  
  # 筛选AF3结构（DSSR + CC距离分析）
  python run.py screen --cif af3_results/
  
  # 完整流程
  python run.py pipeline --pdb xxx.pdb --fixed "2,3,4"
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def parse_fixed_pattern(pattern):
    """
    解析固定位置pattern
    
    支持两种格式：
    1. 逗号分隔的数字: "2,3,4,7,8,9"
    2. Pattern格式: ".GGG..GGG.GG." (G=固定, .=可设计)
    
    Returns:
        set: 固定位置的集合 (1-indexed)
    """
    if not pattern:
        return set()
    
    # 检查是否是pattern格式
    if '.' in pattern or pattern.isalpha():
        fixed = set()
        for i, c in enumerate(pattern, 1):
            if c != '.':
                fixed.add(i)
        return fixed
    else:
        # 逗号分隔格式
        return set(int(x.strip()) for x in pattern.split(',') if x.strip())


def cmd_library(args):
    """
    生成C-rich衍生物库：NA-MPNN获取C概率 → 筛选允许位点 → 生成+1C/+2C组合
    
    基于validate_c_logits.py的逻辑实现。
    """
    from pathlib import Path
    from itertools import combinations
    import torch
    
    print("=" * 80)
    print("C-Rich Derivative Library Generation")
    print("  Method: NA-MPNN C-probability scoring + combinatorial expansion")
    print("=" * 80)
    
    pdb_path = args.pdb
    pdb_name = Path(pdb_path).stem
    
    # 输出目录
    if hasattr(args, 'output') and args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join("library_output", pdb_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析固定位置（G-runs等需要保持不变的位置）
    fixed_positions = parse_fixed_pattern(args.fixed) if args.fixed else set()
    
    print(f"\n[Input]")
    print(f"  PDB: {pdb_path}")
    print(f"  Fixed positions: {sorted(fixed_positions) if fixed_positions else 'auto-detect'}")
    print(f"  Temperature: {args.temp}")
    print(f"  Probability threshold: {args.prob_threshold}")
    
    # 运行NA-MPNN获取C概率
    print(f"\n[1] Running NA-MPNN to get C probabilities...")
    
    try:
        # 导入benchmark中的函数
        sys.path.insert(0, str(Path(__file__).parent / "benchmark"))
        from validate_c_logits import get_c_logits_for_structure, parse_fixed_pattern as parse_fp
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        
        # 获取WT序列（如果提供）
        wt_sequence = args.wt_seq if hasattr(args, 'wt_seq') and args.wt_seq else None
        
        result = get_c_logits_for_structure(
            pdb_path,
            list(fixed_positions),
            wt_sequence=wt_sequence,
            temperature=args.temp,
            device=device
        )
        
        print(f"  Sequence length: {result['sequence_length']}")
        print(f"  Designable positions: {len(result['designable_positions'])}")
        print(f"  Existing C positions: {result['existing_c_positions']}")
        
    except Exception as e:
        print(f"  ✗ Error running NA-MPNN: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 筛选C-allowed位置（probability > threshold）
    print(f"\n[2] Filtering C-allowed positions (prob > {args.prob_threshold})...")
    
    c_allowed = {}
    for pos, prob in result['novel_c_probs'].items():
        if prob > args.prob_threshold:
            c_allowed[pos] = prob
    
    sorted_allowed = sorted(c_allowed.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  C-allowed positions: {len(c_allowed)}")
    if sorted_allowed:
        positions_str = ', '.join([f"{pos}({prob:.3f})" for pos, prob in sorted_allowed[:10]])
        if len(sorted_allowed) > 10:
            positions_str += f"... (+{len(sorted_allowed)-10} more)"
        print(f"  Top positions: {positions_str}")
    
    # 生成衍生物组合
    print(f"\n[3] Generating C-rich derivatives...")
    
    allowed_positions = [pos for pos, _ in sorted_allowed]
    n_allowed = len(allowed_positions)
    
    # 获取原始序列
    # 从result或PDB中提取序列
    original_seq = None
    if wt_sequence:
        original_seq = wt_sequence
    else:
        # 尝试从PDB提取
        try:
            from utils.pdb_utils import parse_pdb
            pdb_info = parse_pdb(pdb_path)
            for chain_id, chain_data in pdb_info.items():
                if isinstance(chain_data, dict) and 'sequence' in chain_data:
                    original_seq = chain_data['sequence']
                    break
        except:
            pass
    
    if not original_seq:
        print("  ⚠ Could not extract original sequence, generating position-based library")
        original_seq = "N" * result['sequence_length']  # Placeholder
    
    print(f"  Original sequence: {original_seq}")
    
    derivatives = []
    
    # +1C: 单位点突变
    for pos in allowed_positions:
        if pos <= len(original_seq):
            new_seq = list(original_seq)
            new_seq[pos - 1] = 'C'
            new_seq = ''.join(new_seq)
            derivatives.append({
                'sequence': new_seq,
                'mutation': f"+1C@{pos}",
                'c_positions': [pos],
                'type': '+1C',
                'prob_sum': c_allowed[pos]
            })
    
    # +2C: 双位点突变（可选限制数量）
    max_double = args.max_double if hasattr(args, 'max_double') else 100
    double_count = 0
    
    for pos1, pos2 in combinations(allowed_positions, 2):
        if double_count >= max_double:
            break
        if pos1 <= len(original_seq) and pos2 <= len(original_seq):
            new_seq = list(original_seq)
            new_seq[pos1 - 1] = 'C'
            new_seq[pos2 - 1] = 'C'
            new_seq = ''.join(new_seq)
            derivatives.append({
                'sequence': new_seq,
                'mutation': f"+2C@{pos1},{pos2}",
                'c_positions': [pos1, pos2],
                'type': '+2C',
                'prob_sum': c_allowed[pos1] + c_allowed[pos2]
            })
            double_count += 1
    
    n_single = len([d for d in derivatives if d['type'] == '+1C'])
    n_double = len([d for d in derivatives if d['type'] == '+2C'])
    
    print(f"  Generated: {n_single} +1C + {n_double} +2C = {len(derivatives)} total")
    
    # 按概率排序
    derivatives.sort(key=lambda x: x['prob_sum'], reverse=True)
    
    # 保存结果
    print(f"\n[4] Saving results...")
    
    # 保存详细JSON
    library_data = {
        'pdb': pdb_path,
        'pdb_name': pdb_name,
        'original_sequence': original_seq,
        'fixed_positions': list(fixed_positions),
        'existing_c_positions': result['existing_c_positions'],
        'c_allowed_positions': [{'pos': pos, 'prob': prob} for pos, prob in sorted_allowed],
        'config': {
            'temperature': args.temp,
            'prob_threshold': args.prob_threshold,
        },
        'statistics': {
            'n_allowed': n_allowed,
            'n_single': n_single,
            'n_double': n_double,
            'n_total': len(derivatives),
        },
        'derivatives': derivatives
    }
    
    json_path = os.path.join(output_dir, "library.json")
    with open(json_path, 'w') as f:
        json.dump(library_data, f, indent=2)
    
    # 保存FASTA格式（供AF3使用）
    fasta_path = os.path.join(output_dir, "library.fasta")
    with open(fasta_path, 'w') as f:
        f.write(f">WT_{pdb_name}\n{original_seq}\n")
        for i, d in enumerate(derivatives, 1):
            f.write(f">{pdb_name}_{d['mutation']}\n{d['sequence']}\n")
    
    # 保存AF3 JSON格式
    af3_dir = os.path.join(output_dir, "af3_jobs")
    os.makedirs(af3_dir, exist_ok=True)
    
    for i, d in enumerate(derivatives, 1):
        job_name = f"{pdb_name}_{d['mutation'].replace(',', '_').replace('@', '')}"
        af3_json = [{
            "name": job_name,
            "modelSeeds": [42],
            "sequences": [{"dnaSequence": {"sequence": d['sequence'], "count": 1}}],
            "dialect": "alphafold3",
            "version": 1
        }]
        af3_path = os.path.join(af3_dir, f"{job_name}.json")
        with open(af3_path, 'w') as f:
            json.dump(af3_json, f, indent=2)
    
    print(f"\n[Output]")
    print(f"  Library JSON: {json_path}")
    print(f"  Library FASTA: {fasta_path}")
    print(f"  AF3 Jobs: {af3_dir}/ ({len(derivatives)} files)")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print(f"  1. Fold: python run.py fold --sequences {json_path} --output af3_results/")
    print(f"  2. Screen: python run.py screen --cif af3_results/")
    print("=" * 80)
    
    return library_data


def cmd_design(args):
    """设计序列：分析结构 → 选C位点 → NA-MPNN → 输出序列（兼容旧接口）"""
    
    from config import PATHS, ensure_output_dirs
    from modules.site_selector import SwitchSiteSelector
    from utils.pdb_utils import parse_pdb, modify_residue_name
    import shutil
    import subprocess
    
    ensure_output_dirs()
    
    pdb_path = args.pdb
    pdb_name = Path(pdb_path).stem
    
    # 支持自定义输出目录
    if hasattr(args, 'output') and args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(PATHS["OUTPUT_NAMPNN"], pdb_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("pH-Responsive NA Design")
    print("=" * 60)
    
    # 1. 解析用户指定的固定位置 (支持pattern格式)
    fixed_positions = parse_fixed_pattern(args.fixed) if args.fixed else set()
    
    # 2. 解析用户指定的强制C位置
    force_c = []
    if args.force_c:
        force_c = [int(x.strip()) for x in args.force_c.split(',')]
    
    # 3. 运行site_selector（静默模式）
    print("\n[1] Analyzing structure...")
    selector = SwitchSiteSelector(mode='auto')
    
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = selector.select_sites(
            pdb_path, 
            fixed_positions=fixed_positions if fixed_positions else None,
            force_c_positions=force_c if force_c else None
        )
    
    chain_id = result["chain_id"]
    seq_info = result["sequence_info"]
    original_seq = seq_info["sequence"]
    
    if not fixed_positions:
        fixed_positions = set(result.get('fixed_residues', []))
    
    if not force_c:
        candidates = [p for p in result.get('force_c_residues', []) if p not in fixed_positions]
        force_c = candidates[:args.n_c]
    
    print(f"\n[2] Design Strategy:")
    print(f"    Sequence: {original_seq} ({len(original_seq)} nt)")
    print(f"    Fixed:    {sorted(fixed_positions)} (不设计)")
    print(f"    Force C:  {force_c} (强制C，用于pH响应)")
    
    designable = [i for i in range(1, len(original_seq)+1) 
                  if i not in fixed_positions and i not in force_c]
    print(f"    NA-MPNN:  {designable} (自由设计)")
    
    # 4. 准备PDB
    print("\n[3] Preparing structure...")
    prepared_pdb = os.path.join(output_dir, f"{pdb_name}_prepared.pdb")
    shutil.copy(pdb_path, prepared_pdb)
    
    is_dna = any(r["resname"].startswith("D") for r in seq_info["residues"])
    c_name = "DC" if is_dna else "C"
    
    for pos in force_c:
        modify_residue_name(prepared_pdb, chain_id, pos, c_name, prepared_pdb)
    
    # 5. 运行NA-MPNN
    print("\n[4] Running NA-MPNN...")
    
    all_fixed = sorted(set(fixed_positions) | set(force_c))
    fixed_str = ' '.join([f'{chain_id}{p}' for p in all_fixed])
    
    design_script = os.path.join(PATHS["NA_MPNN_DIR"], "design_na_with_fixed_positions.py")
    cmd = [
        "python", design_script,
        "--cif_path", prepared_pdb,
        "--out_folder", output_dir,
        "--fixed_positions", fixed_str,
        "--num_sequences", str(args.n_seq),
        "--temperature", str(args.temp),
        "--batch_size", "5",
        "--output_pdbs", "0"
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    # 6. 提取序列
    print("\n[5] Designed Sequences:")
    
    fasta_files = list(Path(output_dir).rglob("*.fa"))
    sequences = []
    
    for fasta_path in fasta_files:
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        current_seq = []
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    full_seq = ''.join(current_seq)
                    na_seq = full_seq.split('/')[-1].upper() if '/' in full_seq else full_seq.upper()
                    if na_seq not in sequences:
                        sequences.append(na_seq)
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            full_seq = ''.join(current_seq)
            na_seq = full_seq.split('/')[-1].upper() if '/' in full_seq else full_seq.upper()
            if na_seq not in sequences:
                sequences.append(na_seq)
    
    for i, seq in enumerate(sequences):
        c_ok = all(seq[p-1] == 'C' for p in force_c if p <= len(seq))
        mark = "✓" if c_ok else "✗"
        print(f"    {i+1}. {seq} {mark}")
    
    # 7. 保存结果
    output_path = os.path.join(output_dir, "designed_sequences.txt")
    with open(output_path, 'w') as f:
        f.write(f"# Original: {original_seq}\n")
        f.write(f"# Fixed: {sorted(fixed_positions)}\n")
        f.write(f"# Force C: {force_c}\n\n")
        for i, seq in enumerate(sequences):
            f.write(f">design_{i+1}\n{seq}\n")
    
    info_path = os.path.join(output_dir, "design_info.json")
    with open(info_path, 'w') as f:
        json.dump({
            "pdb": pdb_path,
            "chain": chain_id,
            "original": original_seq,
            "fixed": sorted(fixed_positions),
            "force_c": force_c,
            "sequences": sequences
        }, f, indent=2)
    
    # 8. 生成AF3 JSON（支持本地和服务器两种格式）
    af3_dir = os.path.join(output_dir, "af3_jobs")
    os.makedirs(af3_dir, exist_ok=True)
    
    for i, seq in enumerate(sequences, 1):
        job_name = f"{pdb_name}_design_{i:03d}"
        
        # 本地AF3格式（用于run_alphafold.py）
        af3_local = [{
            "name": job_name,
            "modelSeeds": [42],
            "sequences": [{"dnaSequence": {"sequence": seq, "count": 1}}],
            "dialect": "alphafold3",
            "version": 1
        }]
        af3_path = os.path.join(af3_dir, f"{job_name}.json")
        with open(af3_path, 'w') as f:
            json.dump(af3_local, f, indent=2)
    
    # 保存设计结果JSON（供pipeline脚本使用）
    design_results_path = os.path.join(output_dir, "design_results.json")
    with open(design_results_path, 'w') as f:
        json.dump({
            "pdb": pdb_path,
            "chain": chain_id,
            "original": original_seq,
            "fixed": sorted(fixed_positions),
            "force_c": force_c,
            "sequences": sequences
        }, f, indent=2)
    
    print(f"\n[Output]")
    print(f"    Sequences ({len(sequences)}): {output_path}")
    print(f"    Design JSON: {design_results_path}")
    print(f"    AF3 JSONs: {af3_dir}/")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print(f"  本地AF3: python run_alphafold.py --json_path={af3_dir}/<name>.json ...")
    print(f"  或上传 {af3_dir}/*.json 到 AF3 Server")
    print(f"  完成后: python run.py screen --cif <af3_results_folder>")
    print("=" * 60)


def extract_sequence_from_cif(cif_path):
    """从CIF文件提取DNA序列"""
    try:
        import Bio.PDB.MMCIF2Dict as M2D
        mmcif_dict = M2D.MMCIF2Dict(cif_path)
        if '_entity_poly_seq.mon_id' in mmcif_dict:
            monids = mmcif_dict['_entity_poly_seq.mon_id']
            seq = ''.join([m[1] if m.startswith('D') else m[0] for m in monids])
            return seq
    except:
        pass
    return None




def cmd_fold(args):
    """使用AF3折叠设计的序列（跳过MSA加速）"""
    from modules.local_af3 import LocalAF3Runner
    
    print("=" * 70)
    print("AlphaFold3 Structure Prediction (No MSA)")
    print("=" * 70)
    
    # 加载序列
    sequences = []
    names = []
    
    if args.sequences.endswith('.json'):
        with open(args.sequences, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # 直接是序列列表
            sequences = data
            names = [f"seq_{i+1}" for i in range(len(sequences))]
        elif isinstance(data, dict):
            # design_results.json 格式
            if 'sequences' in data:
                sequences = data['sequences']
                pdb_name = Path(data.get('pdb', 'design')).stem
                names = [f"{pdb_name}_design_{i+1}" for i in range(len(sequences))]
    else:
        # 假设是 FASTA 格式
        with open(args.sequences, 'r') as f:
            current_name = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_name and current_seq:
                        names.append(current_name)
                        sequences.append(''.join(current_seq))
                    current_name = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_name and current_seq:
                names.append(current_name)
                sequences.append(''.join(current_seq))
    
    if not sequences:
        print("Error: No sequences found in input file")
        return
    
    print(f"\nLoaded {len(sequences)} sequences")
    
    # 创建输出目录
    output_dir = args.output if args.output else "af3_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化AF3 Runner
    runner = LocalAF3Runner(use_conda_env="AF3")
    
    # 检查设置
    checks = runner._check_setup()
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        print(f"\n⚠ AF3 setup issues: {failed}")
        print("Please check README_AF3_SETUP.md for configuration.")
        return
    
    print(f"\nOutput directory: {output_dir}")
    print(f"MSA search: DISABLED (for speed)")
    print("\n" + "-" * 70)
    
    # 逐个预测
    results = []
    for i, (seq, name) in enumerate(zip(sequences, names), 1):
        print(f"\n[{i}/{len(sequences)}] Folding: {name}")
        print(f"  Sequence: {seq[:40]}{'...' if len(seq) > 40 else ''}")
        
        seq_output = os.path.join(output_dir, name)
        
        try:
            result = runner.predict(
                sequence=seq,
                name=name,
                output_dir=seq_output,
                molecule_type="dna",
                run_data_pipeline=False,  # 跳过MSA搜索
                num_diffusion_samples=args.n_samples if hasattr(args, 'n_samples') else 1,
            )
            
            if result.get("success"):
                print(f"  ✓ Completed: {len(result.get('cif_files', []))} structure(s)")
                results.append({
                    "name": name,
                    "sequence": seq,
                    "success": True,
                    "output_dir": seq_output,
                    "cif_files": result.get("cif_files", []),
                })
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                results.append({
                    "name": name,
                    "sequence": seq,
                    "success": False,
                    "error": result.get("error"),
                })
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append({
                "name": name,
                "sequence": seq,
                "success": False,
                "error": str(e),
            })
    
    # 保存汇总
    summary_path = os.path.join(output_dir, "fold_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    n_success = sum(1 for r in results if r.get("success"))
    print("\n" + "=" * 70)
    print(f"Folding complete: {n_success}/{len(results)} succeeded")
    print(f"Results saved: {summary_path}")
    print(f"\nNext step: python run.py screen --cif {output_dir}")
    print("=" * 70)


def cmd_screen(args):
    """筛选AF3结构的pH响应潜力（DSSR验证G4 + CC距离分析）"""
    import glob
    from collections import defaultdict
    
    print("=" * 90)
    print("pH-Responsive NA Screen")
    print("  Methods: DSSR Structure Analysis + CC Pairing Potential")
    print("=" * 90)
    
    # 检查DSSR分析器
    try:
        from modules.dssr_analyzer import DSSRAnalyzer
        dssr_analyzer = DSSRAnalyzer()
        print("\n  ✓ DSSR initialized for G-quadruplex validation")
    except Exception as e:
        print(f"\n  ✗ Error: DSSR is required for screening. Please install it first. ({e})")
        return
    
    print(f"\n  Distance thresholds for C·C+ pairing:")
    print(f"    Close (Pairing): ≤ 4.0 Å (High potential)")
    print(f"    Medium (Network): 4.0 - 6.0 Å (Secondary potential)")
    
    # 获取结构文件
    all_files = []
    for ext in ["*.cif", "*.pdb"]:
        all_files.extend(glob.glob(os.path.join(args.cif, ext)))
        all_files.extend(glob.glob(os.path.join(args.cif, "**", ext), recursive=True))
    
    if not all_files:
        print(f"\nError: No structure files found in {args.cif}")
        return
    
    print(f"\n  Analyzing {len(all_files)} structures...")
    
    # 1. 运行分析
    all_results = []
    for struct_path in sorted(all_files):
        try:
            r = dssr_analyzer.analyze(struct_path)
            if r.get('success'):
                r['file'] = struct_path
                r['filename'] = os.path.basename(struct_path)
                all_results.append(r)
        except Exception as e:
            print(f"    ⚠ Error analyzing {os.path.basename(struct_path)}: {e}")
    
    # 2. 按序列聚合统计
    seq_data = defaultdict(list)
    for r in all_results:
        seq = r.get('sequence', 'unknown')
        seq_data[seq].append(r)
    
    seq_summary = []
    for seq, rs in seq_data.items():
        # 过滤掉 None 结果
        rs = [r for r in rs if r is not None]
        if not rs:
            continue
            
        # 获取基础信息
        cc = rs[0].get('cc_analysis', {})
        if cc is None: cc = {}
        n_c = cc.get('n_cytosines', 0)
        c_positions = cc.get('c_positions', [])
        
        # 统计 G4 和 CC
        g4_detected = any(r.get('g_quadruplex', {}).get('detected') if r.get('g_quadruplex') else False for r in rs)
        max_tetrads = max(r.get('g_quadruplex', {}).get('n_tetrads', 0) if r.get('g_quadruplex') else 0 for r in rs)
        
        avg_close = np.mean([r.get('cc_analysis', {}).get('n_close_pairs', 0) if r.get('cc_analysis') else 0 for r in rs])
        avg_medium = np.mean([r.get('cc_analysis', {}).get('n_medium_pairs', 0) if r.get('cc_analysis') else 0 for r in rs])
        min_dist = min([r.get('cc_analysis', {}).get('min_distance', 99) if r.get('cc_analysis') else 99 for r in rs])
        
        # 计算综合筛选分 (Screening Score)
        # 基础分来自 CC 距离网络 (原GCNS)
        cc_score = max(r.get('cc_analysis', {}).get('gcns_score', 0) if r.get('cc_analysis') else 0 for r in rs)
        
        # 奖励项
        g4_bonus = 5.0 if g4_detected else 0.0
        pairing_bonus = avg_close * 3.0  # 强化近距离配对权重
        
        total_score = cc_score + g4_bonus + pairing_bonus
        
        seq_summary.append({
            'sequence': seq,
            'n_cytosines': n_c,
            'c_positions': c_positions,
            'g4_valid': g4_detected,
            'n_tetrads': max_tetrads,
            'close_pairs': avg_close,
            'medium_pairs': avg_medium,
            'min_dist': min_dist,
            'score': total_score,
            'models': len(rs)
        })
    
    # 3. 排序并显示
    seq_summary.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "=" * 115)
    print(f"{'Rank':<5} {'Sequence':<24} {'G4':<4} {'#C':<4} {'Close':<6} {'Med':<6} {'MinDist':<9} {'Score':<10}")
    print("-" * 115)
    
    for i, s in enumerate(seq_summary, 1):
        seq_disp = s['sequence'][:21] + "..." if len(s['sequence']) > 21 else s['sequence']
        g4_mark = "✓" if s['g4_valid'] else "-"
        print(f"{i:<5} {seq_disp:<24} {g4_mark:<4} {s['n_cytosines']:<4} {s['close_pairs']:>4.1f} {s['medium_pairs']:>5.1f} {s['min_dist']:>7.2f}Å {s['score']:>8.2f}")
    
    # 4. 保存结果
    results_path = os.path.join(args.cif, "screening_results.json")
    with open(results_path, 'w') as f:
        json.dump(seq_summary, f, indent=2)
    
    print("\n" + "=" * 90)
    print(f"Screening complete. Results saved to: {results_path}")
    print("=" * 90)


def cmd_pipeline(args):
    """完整的pH响应核酸设计流程"""
    import time
    
    print("=" * 70)
    print("pH-Responsive NA Design Pipeline")
    print("=" * 70)
    print("\nStages:")
    print("  1. Library: NA-MPNN获取C概率 → 生成+1C/+2C衍生物库")
    print("  2. Fold:    AlphaFold3折叠 (跳过MSA)")
    print("  3. Screen:  DSSR验证G4 + CC距离筛选")
    print("=" * 70)
    
    start_time = time.time()
    
    # 准备输出目录
    pdb_name = Path(args.pdb).stem
    if args.output:
        pipeline_dir = args.output
    else:
        pipeline_dir = os.path.join(os.path.dirname(args.pdb), f"{pdb_name}_pipeline")
    os.makedirs(pipeline_dir, exist_ok=True)
    
    print(f"\nPipeline output: {pipeline_dir}")
    
    # Stage 1: Library Generation
    print("\n" + "=" * 70)
    print("STAGE 1: C-Rich Library Generation (NA-MPNN)")
    print("=" * 70)
    
    library_args = argparse.Namespace(
        pdb=args.pdb,
        fixed=args.fixed,
        wt_seq=getattr(args, 'wt_seq', None),
        temp=getattr(args, 'temp', 0.1),
        prob_threshold=getattr(args, 'prob_threshold', 0.0),
        max_double=getattr(args, 'max_double', 100),
        output=os.path.join(pipeline_dir, "01_library"),
    )
    
    library_result = cmd_library(library_args)
    
    library_json_path = os.path.join(pipeline_dir, "01_library", "library.json")
    if not os.path.exists(library_json_path):
        print("\n✗ Library generation failed - no results file found")
        return
    
    # Stage 2: Fold (optional - if AF3 is available)
    print("\n" + "=" * 70)
    print("STAGE 2: Structure Prediction (AlphaFold3)")
    print("=" * 70)
    
    af3_output = os.path.join(pipeline_dir, "02_af3")
    
    if args.skip_fold:
        print("\n  Skipping fold stage (--skip_fold)")
        print(f"  Please fold sequences manually and place results in: {af3_output}")
    else:
        try:
            from modules.local_af3 import LocalAF3Runner
            runner = LocalAF3Runner(use_conda_env="AF3")
            checks = runner._check_setup()
            
            if all(checks.values()):
                fold_args = argparse.Namespace(
                    sequences=library_json_path,
                    output=af3_output,
                    n_samples=getattr(args, 'n_samples', 1),
                )
                cmd_fold(fold_args)
            else:
                print("\n  ⚠ AF3 not properly configured, skipping fold stage")
                print(f"  Please fold sequences manually and place results in: {af3_output}")
        except Exception as e:
            print(f"\n  ⚠ AF3 initialization failed: {e}")
            print(f"  Please fold sequences manually and place results in: {af3_output}")
    
    # Stage 3: Screen
    print("\n" + "=" * 70)
    print("STAGE 3: Structure Screening (DSSR + CC Distance)")
    print("=" * 70)
    
    # 检查是否有AF3结果
    if os.path.exists(af3_output) and os.listdir(af3_output):
        screen_args = argparse.Namespace(cif=af3_output)
        cmd_screen(screen_args)
    else:
        print(f"\n  Waiting for AF3 results in: {af3_output}")
        print(f"  After folding, run: python run.py screen --cif {af3_output}")
    
    # 汇总
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Pipeline completed in {elapsed:.1f}s")
    print(f"Output directory: {pipeline_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="pH-Responsive NA Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 生成C-rich衍生物库（推荐）
  python run.py library --pdb 8PSB.pdb --fixed ".GGG..GGG..GGG..GGG."
  
  # 旧接口（兼容）
  python run.py design --pdb 8PSB.pdb --fixed "2,3,4" --force_c "1,5"

  # AF3折叠（跳过MSA加速）
  python run.py fold --sequences library.json --output af3_results/

  # 筛选AF3结构（DSSR + CC配对分析）
  python run.py screen --cif af3_results/

  # 完整流程
  python run.py pipeline --pdb 8PSB.pdb --fixed ".GGG..GGG..GGG..GGG."

Workflow:
  1. library: NA-MPNN获取C概率 → 生成+1C/+2C组合库
  2. fold:    AF3折叠（跳过MSA）
  3. screen:  DSSR验证G4 + CC距离筛选
        """
    )
    
    subparsers = parser.add_subparsers(dest="cmd", help="Commands")
    
    # library 命令（新的主要命令）
    p_library = subparsers.add_parser("library", help="生成C-rich衍生物库 (NA-MPNN概率)")
    p_library.add_argument("--pdb", required=True, help="输入PDB文件")
    p_library.add_argument("--fixed", help="固定位置: pattern格式 '.GGG..GGG.' 或逗号分隔 '2,3,4'")
    p_library.add_argument("--wt_seq", help="WT序列（用于识别已有C位置）")
    p_library.add_argument("--temp", type=float, default=0.1, help="NA-MPNN采样温度 (默认0.1)")
    p_library.add_argument("--prob_threshold", type=float, default=0.0, help="C概率阈值 (默认0.0)")
    p_library.add_argument("--max_double", type=int, default=100, help="最大+2C组合数 (默认100)")
    p_library.add_argument("--output", "-o", help="输出目录")
    
    # design 命令（兼容旧接口）
    p_design = subparsers.add_parser("design", help="设计序列（旧接口，建议用library）")
    p_design.add_argument("--pdb", required=True, help="输入PDB文件")
    p_design.add_argument("--chain", help="核酸链ID (默认自动检测)")
    p_design.add_argument("--fixed", help="固定位置: 逗号分隔 '2,3,4' 或 pattern '.GGG..GGG.'")
    p_design.add_argument("--force_c", help="强制C位置，逗号分隔")
    p_design.add_argument("--n_c", type=int, default=2, help="自动选择C位点数量 (默认2)")
    p_design.add_argument("--n_seq", type=int, default=10, help="生成序列数量 (默认10)")
    p_design.add_argument("--temperature", "--temp", type=float, default=0.3, dest="temp", help="采样温度 (默认0.3)")
    p_design.add_argument("--output", "-o", help="输出目录 (默认自动生成)")
    
    # fold 命令 - AF3折叠
    p_fold = subparsers.add_parser("fold", help="AF3结构预测 (跳过MSA)")
    p_fold.add_argument("--sequences", "-s", required=True, help="序列文件 (JSON/FASTA)")
    p_fold.add_argument("--output", "-o", help="输出目录 (默认: af3_results)")
    p_fold.add_argument("--n_samples", type=int, default=1, help="每个序列的diffusion样本数 (默认1)")
    
    # screen 命令 - DSSR + CC距离分析
    p_screen = subparsers.add_parser("screen", help="筛选结构 (DSSR + CC距离)")
    p_screen.add_argument("--cif", required=True, help="结构文件夹 (CIF/PDB)")
    
    # pipeline 命令 - 完整流程
    p_pipeline = subparsers.add_parser("pipeline", help="完整流程 (library → fold → screen)")
    p_pipeline.add_argument("--pdb", required=True, help="输入PDB文件")
    p_pipeline.add_argument("--fixed", help="固定位置: pattern格式 '.GGG..GGG.' 或逗号分隔")
    p_pipeline.add_argument("--wt_seq", help="WT序列")
    p_pipeline.add_argument("--temp", type=float, default=0.1, help="NA-MPNN温度 (默认0.1)")
    p_pipeline.add_argument("--prob_threshold", type=float, default=0.0, help="C概率阈值")
    p_pipeline.add_argument("--max_double", type=int, default=100, help="最大+2C组合数")
    p_pipeline.add_argument("--output", "-o", help="输出目录")
    p_pipeline.add_argument("--skip_fold", action="store_true", help="跳过AF3折叠步骤")
    p_pipeline.add_argument("--n_samples", type=int, default=1, help="AF3样本数")
    
    args = parser.parse_args()
    
    if args.cmd == "library":
        cmd_library(args)
    elif args.cmd == "design":
        cmd_design(args)
    elif args.cmd == "fold":
        cmd_fold(args)
    elif args.cmd == "screen":
        cmd_screen(args)
    elif args.cmd == "pipeline":
        cmd_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
