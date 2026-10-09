"""
[Module 2] NA-MPNN Wrapper
===========================
Interface for calling NA-MPNN to generate nucleic acid sequences.

Features:
- Force specific positions to be cytosine (for pH sensitivity)
- Fix selected switch sites while designing other positions
- Batch sequence generation
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHS, NAMPNN_PARAMS
from utils.pdb_utils import parse_pdb, modify_residue_name, is_nucleic_acid


class NaMpnnRunner:
    """
    Wrapper for NA-MPNN sequence design.
    
    Handles:
    - PDB preparation with forced cytosine mutations
    - NA-MPNN inference execution
    - Result collection and validation
    """
    
    def __init__(self,
                 nampnn_dir: str = None,
                 output_dir: str = None,
                 python_exec: str = None):
        """
        Initialize NA-MPNN runner.
        
        Args:
            nampnn_dir: Path to NA-MPNN repository
            output_dir: Directory for output files
            python_exec: Python executable for NA-MPNN environment
        """
        self.nampnn_dir = nampnn_dir or PATHS["NA_MPNN_DIR"]
        self.output_dir = output_dir or PATHS["OUTPUT_NAMPNN"]
        self.python_exec = python_exec or PATHS["PYTHON_EXEC"]
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Verify NA-MPNN installation
        self._verify_installation()
    
    def _verify_installation(self):
        """Verify NA-MPNN is properly installed."""
        design_script = PATHS.get("NA_MPNN_DESIGN_SCRIPT")
        if design_script and os.path.exists(design_script):
            self.design_script = design_script
            print(f"[NaMpnnRunner] Using custom design script: {design_script}")
            return
        
        # Fallback to inference/run.py
        inference_script = os.path.join(self.nampnn_dir, "inference", "run.py")
        if os.path.exists(inference_script):
            self.design_script = inference_script
            self.use_original_script = True
            print(f"[NaMpnnRunner] Using original NA-MPNN: {inference_script}")
        else:
            raise FileNotFoundError(
                f"NA-MPNN not found at {self.nampnn_dir}. "
                "Please check config.py PATHS['NA_MPNN_DIR']"
            )
    
    def run(self,
            pdb_path: str,
            site_info: Dict[str, Any],
            force_cytosine: bool = True,
            num_sequences: int = None,
            temperature: float = None,
            batch_size: int = None) -> List[str]:
        """
        Run NA-MPNN sequence design.
        
        Args:
            pdb_path: Path to input PDB structure
            site_info: Output from SwitchSiteSelector.select_sites()
            force_cytosine: If True, mutate switch sites to cytosine before design
            num_sequences: Number of sequences to generate
            temperature: Sampling temperature
            batch_size: Batch size for generation
            
        Returns:
            List of paths to designed PDB files
        """
        num_sequences = num_sequences or NAMPNN_PARAMS["NUM_SEQUENCES"]
        temperature = temperature or NAMPNN_PARAMS["TEMPERATURE"]
        batch_size = batch_size or NAMPNN_PARAMS["BATCH_SIZE"]
        
        chain_id = site_info["chain_id"]
        switch_residues = site_info["switch_residues"]
        
        print(f"[NaMpnnRunner] Starting design with {len(switch_residues)} fixed sites")
        
        # Step 1: Prepare PDB with forced cytosine mutations
        if force_cytosine and switch_residues:
            prepared_pdb = self._prepare_pdb_with_cytosine(
                pdb_path, chain_id, switch_residues
            )
        else:
            prepared_pdb = pdb_path
        
        # Step 2: Build fixed positions string
        fixed_positions = self._build_fixed_positions(chain_id, switch_residues)
        
        # Step 3: Create output directory for this run
        run_name = Path(pdb_path).stem
        run_output = os.path.join(self.output_dir, run_name)
        os.makedirs(run_output, exist_ok=True)
        
        # Step 4: Run NA-MPNN
        design_pdbs = self._run_nampnn(
            pdb_path=prepared_pdb,
            output_folder=run_output,
            fixed_positions=fixed_positions,
            num_sequences=num_sequences,
            temperature=temperature,
            batch_size=batch_size
        )
        
        # Step 5: Validate outputs
        validated_pdbs = self._validate_designs(
            design_pdbs, chain_id, switch_residues
        )
        
        print(f"[NaMpnnRunner] Generated {len(validated_pdbs)} valid designs")
        
        return validated_pdbs
    
    def _prepare_pdb_with_cytosine(self,
                                    pdb_path: str,
                                    chain_id: str,
                                    switch_residues: List[int]) -> str:
        """
        Modify PDB to force switch sites to cytosine.
        
        This ensures NA-MPNN treats these as cytosine when designing.
        
        Args:
            pdb_path: Input PDB path
            chain_id: Nucleic acid chain ID
            switch_residues: List of residue IDs to mutate
            
        Returns:
            Path to modified PDB
        """
        # Determine if DNA or RNA from the structure
        structure = parse_pdb(pdb_path)
        is_dna = True  # Default to DNA
        
        for model in structure:
            for chain in model:
                if chain.get_id() == chain_id:
                    for residue in chain:
                        resname = residue.get_resname().strip()
                        if resname in {"A", "C", "G", "U"}:
                            is_dna = False
                            break
                    break
        
        cytosine_name = "DC" if is_dna else "C"
        
        # Create modified PDB
        output_pdb = os.path.join(
            self.output_dir, 
            f"{Path(pdb_path).stem}_cytosine_forced.pdb"
        )
        
        # Copy original
        shutil.copy(pdb_path, output_pdb)
        
        # Modify each switch residue
        for res_id in switch_residues:
            modify_residue_name(output_pdb, chain_id, res_id, cytosine_name, output_pdb)
        
        print(f"[NaMpnnRunner] Forced {len(switch_residues)} sites to {cytosine_name}")
        
        return output_pdb
    
    def _build_fixed_positions(self, 
                               chain_id: str, 
                               switch_residues: List[int]) -> str:
        """
        Build the fixed_positions argument for NA-MPNN.
        
        Format: "ChainResidue ChainResidue ..." e.g., "B12 B15 B18"
        """
        fixed = [f"{chain_id}{res_id}" for res_id in switch_residues]
        return " ".join(fixed)
    
    def _run_nampnn(self,
                    pdb_path: str,
                    output_folder: str,
                    fixed_positions: str,
                    num_sequences: int,
                    temperature: float,
                    batch_size: int) -> List[str]:
        """
        Execute NA-MPNN inference.
        
        Returns:
            List of generated PDB file paths
        """
        # Build command
        cmd = [
            self.python_exec,
            self.design_script,
        ]
        
        # Check which script we're using
        if "design_na_with_fixed_positions.py" in self.design_script:
            # Custom script
            cmd.extend([
                "--cif_path", pdb_path,
                "--out_folder", output_folder,
                "--fixed_positions", fixed_positions,
                "--num_sequences", str(num_sequences),
                "--temperature", str(temperature),
                "--batch_size", str(batch_size),
                "--design_na_only", "1",
                "--output_pdbs", "1"
            ])
        else:
            # Original NA-MPNN run.py
            cmd.extend([
                "--model_type", "na_mpnn",
                "--mode", "design",
                "--pdb_path", pdb_path,
                "--out_folder", output_folder,
                "--fixed_residues", fixed_positions,
                "--batch_size", str(batch_size),
                "--number_of_batches", str((num_sequences + batch_size - 1) // batch_size),
                "--temperature", str(temperature),
                "--design_na_only", "1"
            ])
        
        print(f"[NaMpnnRunner] Running: {' '.join(cmd)}")
        
        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.nampnn_dir,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"[NaMpnnRunner] Warning: Command returned {result.returncode}")
                print(f"[NaMpnnRunner] STDERR: {result.stderr[:1000]}")
            
            print(f"[NaMpnnRunner] STDOUT (last 500 chars): {result.stdout[-500:]}")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("NA-MPNN timed out after 1 hour")
        except Exception as e:
            raise RuntimeError(f"Failed to run NA-MPNN: {e}")
        
        # Collect output PDBs
        backbone_dir = os.path.join(output_folder, "backbones")
        if os.path.exists(backbone_dir):
            pdbs = [
                os.path.join(backbone_dir, f) 
                for f in os.listdir(backbone_dir) 
                if f.endswith(".pdb")
            ]
        else:
            # Try other common output locations
            pdbs = [
                os.path.join(output_folder, f)
                for f in os.listdir(output_folder)
                if f.endswith(".pdb")
            ]
        
        return pdbs
    
    def _validate_designs(self,
                          pdb_paths: List[str],
                          chain_id: str,
                          switch_residues: List[int]) -> List[str]:
        """
        Validate that switch sites are cytosine in designed structures.
        
        Returns:
            List of validated PDB paths
        """
        validated = []
        
        for pdb_path in pdb_paths:
            if not os.path.exists(pdb_path):
                continue
            
            try:
                structure = parse_pdb(pdb_path)
                is_valid = True
                
                for model in structure:
                    for chain in model:
                        if chain.get_id() != chain_id:
                            continue
                        
                        for residue in chain:
                            res_id = residue.get_id()[1]
                            if res_id in switch_residues:
                                resname = residue.get_resname().strip()
                                if resname not in {"DC", "C"}:
                                    print(f"[NaMpnnRunner] Warning: {pdb_path} has "
                                          f"{resname} at position {res_id}, expected C")
                                    is_valid = False
                
                if is_valid:
                    validated.append(pdb_path)
                    
            except Exception as e:
                print(f"[NaMpnnRunner] Error validating {pdb_path}: {e}")
        
        return validated
    
    def parse_fasta_output(self, fasta_path: str) -> List[Dict[str, Any]]:
        """
        Parse NA-MPNN FASTA output file.
        
        Args:
            fasta_path: Path to .fa file
            
        Returns:
            List of sequence dictionaries with metadata
        """
        sequences = []
        
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        current_header = None
        current_seq = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_header and current_seq:
                    sequences.append({
                        "header": current_header,
                        "sequence": ''.join(current_seq),
                        "metadata": self._parse_fasta_header(current_header)
                    })
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget last sequence
        if current_header and current_seq:
            sequences.append({
                "header": current_header,
                "sequence": ''.join(current_seq),
                "metadata": self._parse_fasta_header(current_header)
            })
        
        return sequences
    
    def _parse_fasta_header(self, header: str) -> Dict[str, Any]:
        """Parse metadata from NA-MPNN FASTA header."""
        metadata = {}
        
        # Parse key=value pairs
        parts = header.split(',')
        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                try:
                    # Try to convert to number
                    if '.' in value:
                        metadata[key.strip()] = float(value)
                    else:
                        metadata[key.strip()] = int(value)
                except ValueError:
                    metadata[key.strip()] = value
        
        return metadata
    
    def extract_na_sequence(self, full_sequence: str) -> str:
        """
        Extract nucleic acid sequence from combined protein/NA sequence.
        
        NA-MPNN outputs combined sequences separated by '/'.
        NA uses lowercase letters: a, c, g, t (DNA) or b, d, h, u (RNA)
        """
        # Split by chain separator
        chains = full_sequence.split('/')
        
        na_sequences = []
        for chain in chains:
            # Check if it's NA (lowercase)
            if chain and chain[0].islower():
                # Convert to standard nucleotide codes
                na_seq = chain.upper()
                na_seq = na_seq.replace('B', 'A').replace('D', 'C').replace('H', 'G')
                na_sequences.append(na_seq)
        
        return ''.join(na_sequences)

