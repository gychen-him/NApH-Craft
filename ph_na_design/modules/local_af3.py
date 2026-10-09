"""
Local AlphaFold3 Integration Module
====================================

Provides interface to run AlphaFold3 predictions locally for pH-responsive NA design.

Requirements:
- AlphaFold3 installed (either via conda or system)
- AF3_DIR, AF3_MODEL_DIR, and AF3_DB_DIR set to the local install,
  model parameters, and databases

Usage:
    from modules.local_af3 import LocalAF3Runner
    
    runner = LocalAF3Runner()
    output_dir = runner.predict("AGGGTAGGGCGGCGGG", name="test")
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any


class LocalAF3Runner:
    """Run AlphaFold3 predictions locally."""
    
    # Paths come from arguments or AF3_DIR / AF3_MODEL_DIR / AF3_DB_DIR.
    DEFAULT_PYTHON = "python3.12"  # AF3 needs Python 3.12
    
    def __init__(self,
                 af3_dir: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 db_dir: Optional[str] = None,
                 python_cmd: Optional[str] = None,
                 use_conda_env: Optional[str] = None):
        """
        Initialize LocalAF3Runner.
        
        Args:
            af3_dir: AlphaFold3 source directory
            model_dir: Model parameters directory
            db_dir: Databases directory
            python_cmd: Python command (e.g., 'python3.12')
            use_conda_env: Conda environment name to use (e.g., 'AF3')
        """
        self.af3_dir = af3_dir or os.environ.get('AF3_DIR', '')
        self.model_dir = model_dir or os.environ.get('AF3_MODEL_DIR', '')
        self.db_dir = db_dir or os.environ.get('AF3_DB_DIR', '')
        self.python_cmd = python_cmd or os.environ.get('AF3_PYTHON', self.DEFAULT_PYTHON)
        self.use_conda_env = use_conda_env
        
        self.run_script = os.path.join(self.af3_dir, "run_alphafold.py")
        
    def _check_setup(self) -> Dict[str, bool]:
        """Check if AF3 is properly set up."""
        checks = {
            "af3_dir_exists": os.path.isdir(self.af3_dir),
            "run_script_exists": os.path.isfile(self.run_script),
            "model_dir_exists": os.path.isdir(self.model_dir),
            "db_dir_exists": os.path.isdir(self.db_dir),
        }
        
        # Check for model file
        model_file = os.path.join(self.model_dir, "af3.bin")
        checks["model_file_exists"] = os.path.isfile(model_file)
        
        return checks
    
    def create_input_json(self,
                          sequence: str,
                          name: str = "prediction",
                          molecule_type: str = "dna",
                          msa_data: Optional[Dict[str, str]] = None,
                          model_seeds: List[int] = None,
                          output_path: Optional[str] = None) -> str:
        """
        Create input JSON file for AlphaFold3.
        
        Args:
            sequence: DNA/RNA/Protein sequence
            name: Job name
            molecule_type: 'dna', 'rna', or 'protein'
            msa_data: Optional dict with 'unpairedMsa' and/or 'pairedMsa' content (Stockholm/A3M)
            model_seeds: List of random seeds (default: [1])
            output_path: Where to save the JSON
            
        Returns:
            Path to the created JSON file
        """
        if model_seeds is None:
            model_seeds = [1]
        
        # Determine sequence key based on molecule type
        mol_type = molecule_type.lower()
        if mol_type == "dna":
            seq_key = "dna"
        elif mol_type == "rna":
            seq_key = "rna"
        else:
            seq_key = "protein"
        
        # Build sequence object
        seq_obj = {
            "id": "A",  # Default Chain ID
            "sequence": sequence.upper()
        }
        
        # Add MSA if provided
        if msa_data:
            if "unpairedMsa" in msa_data:
                seq_obj["unpairedMsa"] = msa_data["unpairedMsa"]
            if "pairedMsa" in msa_data:
                seq_obj["pairedMsa"] = msa_data["pairedMsa"]
        
        # AF3 JSON 格式需要包含 dialect, version, id 字段
        input_data = {
            "dialect": "alphafold3",
            "version": 1,
            "name": name,
            "sequences": [
                {
                    seq_key: seq_obj
                }
            ],
            "modelSeeds": model_seeds
        }
        
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"{name}_input.json")
        
        with open(output_path, 'w') as f:
            json.dump(input_data, f, indent=2)
        
        return output_path
    
    def predict(self,
                sequence: str,
                name: str = "prediction",
                output_dir: Optional[str] = None,
                molecule_type: str = "dna",
                msa_data: Optional[Dict[str, str]] = None,
                num_diffusion_samples: int = 1,
                model_seeds: List[int] = None,
                run_data_pipeline: bool = False,  # 默认跳过MSA搜索，纯核酸预测不需要
                timeout: int = 3600) -> Dict[str, Any]:
        """
        Run AlphaFold3 prediction on a sequence.
        
        Args:
            sequence: DNA/RNA/Protein sequence
            name: Job name
            output_dir: Output directory
            molecule_type: 'dna', 'rna', or 'protein'
            msa_data: Optional MSA content
            num_diffusion_samples: Number of diffusion samples
            model_seeds: Random seeds
            run_data_pipeline: Whether to run MSA search (default False for NA-only)
            timeout: Timeout in seconds
            
        Returns:
            Dict with prediction results and paths
        """
        # Check setup
        checks = self._check_setup()
        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            raise RuntimeError(f"AF3 setup check failed: {failed}")
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), f"af3_{name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create input JSON
        input_json = self.create_input_json(
            sequence=sequence,
            name=name,
            molecule_type=molecule_type,
            msa_data=msa_data,
            model_seeds=model_seeds,
            output_path=os.path.join(output_dir, f"{name}_input.json")
        )
        
        # Build command
        cmd = [
            self.python_cmd,
            self.run_script,
            f"--json_path={input_json}",
            f"--output_dir={output_dir}",
            f"--model_dir={self.model_dir}",
            f"--db_dir={self.db_dir}",
            f"--num_diffusion_samples={num_diffusion_samples}",
        ]
        
        if not run_data_pipeline:
            cmd.append("--run_data_pipeline=False")
        
        # 使用 XLA 实现来避免 CUDA 版本兼容性问题
        cmd.append("--flash_attention_implementation=xla")
        
        # Wrap with conda activation if specified
        if self.use_conda_env:
            # 获取 conda 基础路径
            conda_base = os.environ.get('CONDA_PREFIX_1') or os.environ.get('CONDA_PREFIX', '').rsplit('/envs/', 1)[0]
            if not conda_base:
                # 尝试从 CONDA_EXE 推断
                conda_exe = os.environ.get('CONDA_EXE', '')
                if conda_exe:
                    conda_base = os.path.dirname(os.path.dirname(conda_exe))
            
            # 构建完整的 bash 命令
            full_cmd = " ".join(cmd)
            cmd = [
                "bash", "-c",
                f"source {conda_base}/etc/profile.d/conda.sh && "
                f"conda activate {self.use_conda_env} && " 
                f"cd {self.af3_dir} && "
                f"{full_cmd}"
            ]
        
        print(f"Running AlphaFold3 prediction...")
        print(f"  Sequence: {sequence[:30]}...")
        print(f"  Output: {output_dir}")
        
        # 清理可能干扰 AF3 的环境变量 (如 NA-MPNN 设置的 CUDA 11.8 变量)
        clean_env = os.environ.copy()
        # 移除 NA-MPNN 可能设置的 CUDA/LD_LIBRARY_PATH 变量
        cuda_vars_to_remove = [
            'CUDA_HOME', 'CUDA_VERSION', 'CUDA_PATH',
            'LD_LIBRARY_PATH', 'LIBRARY_PATH',
            'NV_CUDA_COMPAT_PACKAGE', 'NV_CUDA_CUDART_VERSION',
            'NV_CUDA_LIB_VERSION', 'NVIDIA_REQUIRE_CUDA'
        ]
        for var in cuda_vars_to_remove:
            clean_env.pop(var, None)
        # 移除所有 NV_ 开头的变量
        for key in list(clean_env.keys()):
            if key.startswith('NV_'):
                clean_env.pop(key, None)
        
        # 设置 XLA_FLAGS 来禁用 triton，避免 CUDA ptxas 版本问题
        clean_env['XLA_FLAGS'] = "--xla_gpu_enable_triton_gemm=false"
        
        # Run prediction
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.af3_dir,
                env=clean_env  # 使用清理后的环境
            )
            
            success = result.returncode == 0
            
            if not success:
                print(f"  ✗ Prediction failed!")
                print(f"  Stderr: {result.stderr[:500]}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "output_dir": output_dir
                }
            
            print(f"  ✓ Prediction completed!")
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout after {timeout} seconds",
                "output_dir": output_dir
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir
            }
        
        # Find output files - AF3 outputs to output_dir/name/name_model.cif
        cif_files = list(Path(output_dir).glob("**/*_model.cif"))
        
        if not cif_files:
            # 也尝试直接在输出目录查找
            cif_files = list(Path(output_dir).glob("*_model.cif"))
        
        return {
            "success": True,
            "output_dir": output_dir,
            "cif_files": [str(f) for f in cif_files],
            "input_json": input_json,
            "sequence": sequence,
            "name": name
        }
    
    def predict_batch(self,
                      sequences: List[str],
                      names: Optional[List[str]] = None,
                      output_dir: Optional[str] = None,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Run predictions on multiple sequences.
        
        Args:
            sequences: List of sequences
            names: List of names (optional)
            output_dir: Base output directory
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of prediction results
        """
        if names is None:
            names = [f"seq_{i+1}" for i in range(len(sequences))]
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="af3_batch_")
        
        results = []
        for i, (seq, name) in enumerate(zip(sequences, names)):
            print(f"\n[{i+1}/{len(sequences)}] Predicting {name}...")
            
            seq_output = os.path.join(output_dir, name)
            result = self.predict(
                sequence=seq,
                name=name,
                output_dir=seq_output,
                **kwargs
            )
            results.append(result)
        
        return results


def test_af3_setup():
    """Quick test to check if AF3 is properly configured."""
    print("Testing AlphaFold3 setup...")
    
    runner = LocalAF3Runner()
    checks = runner._check_setup()
    
    print("\nSetup checks:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    if all(checks.values()):
        print("\n✓ AlphaFold3 is properly configured!")
        return True
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    test_af3_setup()

