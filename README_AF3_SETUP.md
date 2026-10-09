# AlphaFold3 Conda Environment Setup & MSA Strategy

This guide provides detailed instructions for setting up the AlphaFold3 (AF3) environment, specifically optimized for servers with potential CUDA version mismatches and for users who want to avoid downloading huge MSA databases.

## 1. Conda Environment Installation

Follow these steps to create a clean environment for AF3:

```bash
# 1. Create a Python 3.12 environment (AF3 requirement)
conda create -n AF3 python=3.12 -y
conda activate AF3

# 2. Install JAX with CUDA support
# Match this to your system's CUDA capability. For CUDA 12.x:
pip install jax[cuda12]==0.4.34

# 3. Clone and Install AlphaFold3
git clone https://github.com/google-deepmind/alphafold3.git ~/alphafold3
cd ~/alphafold3
pip install -e .

# 4. Install additional utilities
pip install biopython pandas
```

## 2. Resolving CUDA/PTX Compatibility Issues

If you see errors like `ptxas too old` or `XlaRuntimeError`, it's usually due to a mismatch between the JAX-compiled kernels and your system's CUDA driver. 

**Fix:** Disable the Triton GEMM kernel and use XLA's native implementation.

In your execution script or `~/.bashrc`:
```bash
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
```

When running `run_alphafold.py`, always include:
```bash
--flash_attention_implementation=xla
```

## 3. MSA (Multiple Sequence Alignment) Strategy

AlphaFold3's full database is ~10TB. You can avoid this by using different strategies for different molecule types.

### A. Nucleic Acids (DNA/RNA)
For pure nucleic acid design or folding, MSA is often less critical than for proteins. You can skip the data pipeline entirely:

**Configuration:**
- Set `--run_data_pipeline=False`
- No database download required (only model parameters `af3.bin` are needed).

### B. Proteins
For proteins, accuracy drops significantly without MSA. 

**Recommended Approaches:**

1. **User-Provided MSA**: 
   If you have `.a3m` or `.sto` files from other runs or tools, you can include them directly in the input JSON under the `unpairedMsa` and `pairedMsa` fields.

2. **ColabFold API Integration**:
   Use [ColabFold](https://github.com/sokrypton/ColabFold) to search for MSAs using their public API.
   ```bash
   # Example: using colabfold_search
   colabfold_search sequence.fasta /path/to/databases result_dir
   ```
   Then take the generated `.a3m` and provide it to the AF3 pipeline.

3. **Hybrid Mode**:
   Our `local_af3.py` wrapper is designed to switch between these modes based on the input molecule type.

## 4. Model Parameters

You still need the trained model weights (~10GB):
1. Request access from Google DeepMind.
2. Download `af3.bin` and associated files.
3. Set the directory path:
   ```bash
   export AF3_MODEL_DIR="/path/to/models"
   ```

## 5. Summary of Environment Variables

Ensure these are set before running the pipeline:

```bash
# Path to AF3 source code
export AF3_DIR="$HOME/alphafold3"
# Path to model parameters
export AF3_MODEL_DIR="$HOME/models"

# Conda environment name
export AF3_ENV="AF3"
# JAX Fix
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
```

