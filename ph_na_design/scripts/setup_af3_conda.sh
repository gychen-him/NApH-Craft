#!/bin/bash
# AlphaFold3 Conda Environment Setup Script
# 为pH-Responsive NA Design Pipeline配置本地AF3环境

set -e

echo "=========================================="
echo "AlphaFold3 Conda Environment Setup"
echo "=========================================="

# 配置路径
: "${AF3_DIR:?Set AF3_DIR to the AlphaFold3 source directory}"
: "${AF3_MODEL_DIR:?Set AF3_MODEL_DIR to the AlphaFold3 model directory}"
: "${AF3_DB_DIR:?Set AF3_DB_DIR to the AlphaFold3 database directory}"
AF3_SOURCE="$AF3_DIR"
MODEL_DIR="$AF3_MODEL_DIR"
DB_DIR="$AF3_DB_DIR"

# 检查是否已存在AF3环境
if conda info --envs | grep -q "AF3"; then
    echo "AF3 conda environment already exists."
    echo "To recreate, first run: conda env remove -n AF3"
    read -p "Continue with existing environment? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Creating new AF3 conda environment..."
    
    # 创建Python 3.12环境（AlphaFold3官方推荐）
    conda create -n AF3 python=3.12 -y
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AF3

echo "Installing dependencies..."

# 安装基础依赖
pip install --upgrade pip setuptools wheel

# 安装HMMER（如果系统没有）
if ! command -v jackhmmer &> /dev/null; then
    echo "Installing HMMER via conda..."
    conda install -c bioconda hmmer -y
fi

# 安装AlphaFold3
echo "Installing AlphaFold3 from source..."
cd "$AF3_SOURCE"
pip install -e .

# 构建数据文件
echo "Building chemical components database..."
build_data || echo "Note: build_data may need sudo permissions"

# 测试安装
echo ""
echo "Testing installation..."

if python -c "import alphafold3; print('✓ AlphaFold3 import OK')" 2>/dev/null; then
    echo "✓ AlphaFold3 installed successfully!"
else
    echo "✗ AlphaFold3 import failed"
    echo "  Try running with: pip install ."
fi

# 测试CUDA
python -c "import jax; print(f'✓ JAX devices: {jax.devices()}')" 2>/dev/null || echo "Note: JAX may need CUDA setup"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use AlphaFold3:"
echo "  conda activate AF3"
echo "  python ${AF3_SOURCE}/run_alphafold.py \\"
echo "    --json_path=input.json \\"
echo "    --output_dir=./output \\"
echo "    --model_dir=${MODEL_DIR} \\"
echo "    --db_dir=${DB_DIR}"
echo ""

