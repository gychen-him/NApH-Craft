#!/bin/bash
# =============================================================================
# pH-Responsive Nucleic Acid Design Pipeline
# 完整工作流：位点定位 -> NA-MPNN设计 -> AF3折叠 -> GCNS打分
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置区域 - 请根据实际情况修改
# =============================================================================

# Conda环境名称
NAMPNN_ENV="NA-MPNN"
AF3_ENV="AF3"

# 路径配置。仓库根目录由脚本位置推出；AF3 安装位置用环境变量指定。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMPNN_DIR="${NAMPNN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${AF3_DIR:?Set AF3_DIR to the AlphaFold3 source directory}"
: "${AF3_MODEL_DIR:?Set AF3_MODEL_DIR to the AlphaFold3 model directory}"
: "${AF3_DB_DIR:?Set AF3_DB_DIR to the AlphaFold3 database directory}"

# 默认参数
DEFAULT_N_SEQ=20
DEFAULT_TEMPERATURE=0.2
DEFAULT_N_C=2

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    cat << EOF
用法: $(basename $0) [选项]

pH-Responsive Nucleic Acid Design Pipeline
完整工作流：位点定位 -> NA-MPNN设计 -> AF3折叠 -> GCNS打分

必需参数:
  -p, --pdb PATH          输入PDB文件路径
  -f, --fixed PATTERN     固定位置模式 (例如: ".GGG..GGG.GG.GGG....." 或 "2,3,4,7,8,9")
  -o, --output DIR        输出目录

可选参数:
  -n, --n-seq NUM         生成序列数量 (默认: ${DEFAULT_N_SEQ})
  -t, --temperature TEMP  采样温度 (默认: ${DEFAULT_TEMPERATURE})
  --n-c NUM               强制C位点数量 (默认: ${DEFAULT_N_C})
  --skip-nampnn           跳过NA-MPNN设计步骤
  --skip-af3              跳过AF3折叠步骤
  --skip-score            跳过GCNS打分步骤
  -h, --help              显示帮助信息

示例:
  $(basename $0) -p 8PSB.pdb -f ".GGG..GGG.GG.GGG....." -o ./output
  $(basename $0) -p 8PSB.pdb -f "2,3,4,7,8,9,11,12,14,15,16" -o ./output

环境要求:
  - Conda环境: ${NAMPNN_ENV} (用于NA-MPNN)
  - Conda环境: ${AF3_ENV} (用于AlphaFold3)

EOF
}

# =============================================================================
# 参数解析
# =============================================================================
PDB_PATH=""
FIXED_PATTERN=""
OUTPUT_DIR=""
N_SEQ=${DEFAULT_N_SEQ}
TEMPERATURE=${DEFAULT_TEMPERATURE}
N_C=${DEFAULT_N_C}
SKIP_NAMPNN=false
SKIP_AF3=false
SKIP_SCORE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pdb)
            PDB_PATH="$2"
            shift 2
            ;;
        -f|--fixed)
            FIXED_PATTERN="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--n-seq)
            N_SEQ="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --n-c)
            N_C="$2"
            shift 2
            ;;
        --skip-nampnn)
            SKIP_NAMPNN=true
            shift
            ;;
        --skip-af3)
            SKIP_AF3=true
            shift
            ;;
        --skip-score)
            SKIP_SCORE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证必需参数
if [[ -z "$PDB_PATH" ]] || [[ -z "$FIXED_PATTERN" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 缺少必需参数 (--pdb, --fixed, --output)"
    show_help
    exit 1
fi

if [[ ! -f "$PDB_PATH" ]]; then
    echo "错误: PDB文件不存在: $PDB_PATH"
    exit 1
fi

# =============================================================================
# 初始化
# =============================================================================
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 创建输出目录结构
mkdir -p "${OUTPUT_DIR}/01_nampnn_designs"
mkdir -p "${OUTPUT_DIR}/02_af3_inputs"
mkdir -p "${OUTPUT_DIR}/03_af3_outputs"
mkdir -p "${OUTPUT_DIR}/04_gcns_results"

# 日志文件
LOG_FILE="${OUTPUT_DIR}/pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================================="
echo "pH-Responsive Nucleic Acid Design Pipeline"
echo "开始时间: $(date)"
echo "=============================================================================="
echo ""
echo "配置参数:"
echo "  PDB文件: $PDB_PATH"
echo "  固定模式: $FIXED_PATTERN"
echo "  输出目录: $OUTPUT_DIR"
echo "  序列数量: $N_SEQ"
echo "  采样温度: $TEMPERATURE"
echo "  强制C位点: $N_C"
echo ""

# =============================================================================
# Step 1: NA-MPNN 序列设计
# =============================================================================
if [[ "$SKIP_NAMPNN" == false ]]; then
    echo "=============================================================================="
    echo "[Step 1/3] NA-MPNN 序列设计"
    echo "=============================================================================="
    
    conda activate ${NAMPNN_ENV}
    echo "已激活环境: ${NAMPNN_ENV}"
    
    cd "${NAMPNN_DIR}"
    
    # 运行设计
    python ph_na_design/run.py design \
        --pdb "$PDB_PATH" \
        --fixed "$FIXED_PATTERN" \
        --n_seq "$N_SEQ" \
        --temp "$TEMPERATURE" \
        --n_c "$N_C" \
        --output "${OUTPUT_DIR}/01_nampnn_designs"
    
    echo ""
    echo "✓ NA-MPNN设计完成"
    echo "  输出目录: ${OUTPUT_DIR}/01_nampnn_designs"
    
    conda deactivate
else
    echo "[Step 1/3] 跳过 NA-MPNN 设计"
fi

# =============================================================================
# Step 2: 生成AF3输入并运行预测
# =============================================================================
if [[ "$SKIP_AF3" == false ]]; then
    echo ""
    echo "=============================================================================="
    echo "[Step 2/3] AlphaFold3 结构预测"
    echo "=============================================================================="
    
    # 2.1 生成AF3输入JSON
    echo "[2.1] 生成AF3输入文件..."
    
    conda activate ${NAMPNN_ENV}
    
    # 读取设计的序列并生成AF3 JSON
    export OUTPUT_DIR
    python << 'PYTHON_SCRIPT'
import os
import json
import glob
import shutil

output_dir = os.environ.get('OUTPUT_DIR')
nampnn_output = os.path.join(output_dir, '01_nampnn_designs')
af3_input_dir = os.path.join(output_dir, '02_af3_inputs')

# 方法1: 直接读取design_results.json
design_results = os.path.join(nampnn_output, 'design_results.json')
sequences = []

if os.path.exists(design_results):
    with open(design_results, 'r') as f:
        data = json.load(f)
        sequences = data.get('sequences', [])
    print(f"从 design_results.json 读取到 {len(sequences)} 个序列")

# 方法2: 读取FASTA文件
if not sequences:
    fasta_files = glob.glob(os.path.join(nampnn_output, '**/*.fa'), recursive=True)
    for fasta_file in fasta_files:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>') and not line.startswith('#'):
                    # 提取核酸序列（可能包含蛋白/核酸分隔符）
                    na_seq = line.split('/')[-1].upper() if '/' in line else line.upper()
                    if na_seq and na_seq not in sequences:
                        sequences.append(na_seq)
    print(f"从 FASTA 文件读取到 {len(sequences)} 个序列")

if not sequences:
    print("警告: 未找到设计的序列，请检查NA-MPNN输出")
    exit(1)

# 方法3: 复制已有的AF3 JSON (如果存在)
existing_af3_dir = os.path.join(nampnn_output, 'af3_jobs')
if os.path.isdir(existing_af3_dir):
    for json_file in glob.glob(os.path.join(existing_af3_dir, '*.json')):
        shutil.copy(json_file, af3_input_dir)
    print(f"复制了 {len(glob.glob(os.path.join(existing_af3_dir, '*.json')))} 个AF3 JSON文件")
else:
    # 生成AF3 JSON文件
    for i, seq in enumerate(sequences, 1):
        job_name = f"design_{i:03d}"
        job = [{
            "name": job_name,
            "modelSeeds": [42],
            "sequences": [
                {"dnaSequence": {"sequence": seq, "count": 1}}
            ],
            "dialect": "alphafold3",
            "version": 1
        }]
        
        json_path = os.path.join(af3_input_dir, f"{job_name}.json")
        with open(json_path, 'w') as f:
            json.dump(job, f, indent=2)
        
        print(f"  生成: {json_path}")

print(f"\n✓ 完成AF3输入准备")
PYTHON_SCRIPT
    
    conda deactivate
    
    # 2.2 运行AF3预测
    echo ""
    echo "[2.2] 运行AlphaFold3预测..."
    
    conda activate ${AF3_ENV}
    echo "已激活环境: ${AF3_ENV}"
    
    AF3_OUTPUT="${OUTPUT_DIR}/03_af3_outputs"
    AF3_INPUTS="${OUTPUT_DIR}/02_af3_inputs"
    
    # 遍历所有JSON文件进行预测
    for json_file in "${AF3_INPUTS}"/*.json; do
        if [[ -f "$json_file" ]]; then
            job_name=$(basename "$json_file" .json)
            job_output="${AF3_OUTPUT}/${job_name}"
            
            if [[ -d "$job_output" ]] && [[ -n "$(ls -A "$job_output" 2>/dev/null)" ]]; then
                echo "  跳过已完成: $job_name"
                continue
            fi
            
            echo "  预测: $job_name"
            
            python "${AF3_DIR}/run_alphafold.py" \
                --json_path="$json_file" \
                --output_dir="$job_output" \
                --model_dir="${AF3_MODEL_DIR}" \
                --db_dir="${AF3_DB_DIR}" \
                --flash_attention_implementation=xla \
                2>&1 | tee -a "${OUTPUT_DIR}/af3_${job_name}.log"
            
            echo "    ✓ 完成"
        fi
    done
    
    echo ""
    echo "✓ AlphaFold3预测完成"
    echo "  输出目录: ${AF3_OUTPUT}"
    
    conda deactivate
else
    echo "[Step 2/3] 跳过 AlphaFold3 预测"
fi

# =============================================================================
# Step 3: GCNS 打分排序
# =============================================================================
if [[ "$SKIP_SCORE" == false ]]; then
    echo ""
    echo "=============================================================================="
    echo "[Step 3/3] GCNS 打分排序"
    echo "=============================================================================="
    
    conda activate ${NAMPNN_ENV}
    echo "已激活环境: ${NAMPNN_ENV}"
    
    cd "${NAMPNN_DIR}"
    
    # 运行GCNS打分
    python ph_na_design/run.py screen \
        --cif "${OUTPUT_DIR}/03_af3_outputs"
    
    # 复制结果到最终目录
    if [[ -f "${OUTPUT_DIR}/03_af3_outputs/gcns_results.json" ]]; then
        cp "${OUTPUT_DIR}/03_af3_outputs/gcns_results.json" "${OUTPUT_DIR}/04_gcns_results/"
    fi
    
    echo ""
    echo "✓ GCNS打分完成"
    echo "  结果目录: ${OUTPUT_DIR}/04_gcns_results"
    
    conda deactivate
else
    echo "[Step 3/3] 跳过 GCNS 打分"
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "=============================================================================="
echo "Pipeline 完成!"
echo "结束时间: $(date)"
echo "=============================================================================="
echo ""
echo "输出目录结构:"
echo "  ${OUTPUT_DIR}/"
echo "  ├── 01_nampnn_designs/  # NA-MPNN设计的序列"
echo "  ├── 02_af3_inputs/      # AF3输入JSON文件"
echo "  ├── 03_af3_outputs/     # AF3预测结构"
echo "  ├── 04_gcns_results/    # GCNS打分结果"
echo "  └── pipeline.log        # 运行日志"
echo ""
echo "查看最终排名:"
echo "  cat ${OUTPUT_DIR}/04_gcns_results/gcns_results.json"
echo ""

