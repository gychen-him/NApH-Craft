"""
[Module 0] AF3 Interaction Handler
===================================
Handles input/output processing with Google AlphaFold Server.

Features:
- Generate JSON input files for AF3 Server
- Parse CIF result files
- Extract pLDDT and PAE metrics
- Convert CIF to PDB format
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHS, PARAMS
from utils.pdb_utils import parse_cif, cif_to_pdb, get_structure_info


class AF3JobManager:
    """
    Manages AlphaFold 3 Server job preparation and result parsing.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize AF3 Job Manager.
        
        Args:
            output_dir: Directory for output files. Uses config default if not specified.
        """
        self.output_dir = output_dir or PATHS["OUTPUT_AF3"]
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_json(self, 
                      job_name: str,
                      protein_seq: str,
                      na_seq: str,
                      na_type: str = "dna",
                      model_seeds: Optional[List[int]] = None) -> str:
        """
        Generate AF3 Server compatible JSON input file.
        
        Args:
            job_name: Name for the prediction job
            protein_seq: Protein sequence (single letter code)
            na_seq: Nucleic acid sequence (single letter code: ACGT for DNA, ACGU for RNA)
            na_type: Type of nucleic acid ("dna" or "rna")
            model_seeds: List of random seeds for prediction. Default [1]
            
        Returns:
            Path to the generated JSON file
        """
        if model_seeds is None:
            model_seeds = [1]
        
        # Build the JSON structure for AF3 Server
        af3_input = {
            "name": job_name,
            "modelSeeds": model_seeds,
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": protein_seq.upper(),
                        "count": 1
                    }
                }
            ]
        }
        
        # Add nucleic acid chain
        if na_type.lower() == "dna":
            af3_input["sequences"].append({
                "dnaSequence": {
                    "sequence": na_seq.upper(),
                    "count": 1
                }
            })
        elif na_type.lower() == "rna":
            af3_input["sequences"].append({
                "rnaSequence": {
                    "sequence": na_seq.upper(),
                    "count": 1
                }
            })
        else:
            raise ValueError(f"Unknown na_type: {na_type}. Use 'dna' or 'rna'.")
        
        # Save to file
        output_path = os.path.join(self.output_dir, f"{job_name}_af3_input.json")
        with open(output_path, 'w') as f:
            json.dump(af3_input, f, indent=2)
        
        print(f"[AF3Handler] Generated input JSON: {output_path}")
        return output_path
    
    def parse_cif_results(self, 
                          cif_path: str,
                          output_name: str = "WT_converted") -> Dict[str, Any]:
        """
        Parse AF3 CIF result file and extract relevant information.
        
        Args:
            cif_path: Path to the AF3 output CIF file
            output_name: Base name for output files
            
        Returns:
            Dictionary containing:
            - pdb_path: Path to converted PDB file
            - structure_info: Basic structure information
            - plddt_scores: Per-residue pLDDT scores (if available)
            - pae_matrix: PAE matrix (if available)
            - warnings: List of any warnings
        """
        result = {
            "pdb_path": None,
            "structure_info": None,
            "plddt_scores": None,
            "pae_matrix": None,
            "warnings": []
        }
        
        # Parse the CIF file
        structure = parse_cif(cif_path)
        result["structure_info"] = get_structure_info(structure)
        
        # Convert to PDB
        pdb_path = os.path.join(self.output_dir, f"{output_name}.pdb")
        cif_to_pdb(cif_path, pdb_path)
        result["pdb_path"] = pdb_path
        print(f"[AF3Handler] Converted CIF to PDB: {pdb_path}")
        
        # Try to extract pLDDT from B-factor column
        plddt_scores = self._extract_plddt_from_structure(structure)
        if plddt_scores:
            result["plddt_scores"] = plddt_scores
            avg_plddt = np.mean(list(plddt_scores.values()))
            print(f"[AF3Handler] Average pLDDT: {avg_plddt:.2f}")
            
            if avg_plddt < PARAMS["PLDDT_THRESHOLD"]:
                warning = f"Low pLDDT score ({avg_plddt:.2f} < {PARAMS['PLDDT_THRESHOLD']}). Structure may be unreliable."
                result["warnings"].append(warning)
                print(f"[AF3Handler] WARNING: {warning}")
        
        # Try to parse PAE from associated JSON file
        pae_path = self._find_pae_file(cif_path)
        if pae_path:
            result["pae_matrix"] = self._parse_pae_json(pae_path)
            print(f"[AF3Handler] Loaded PAE matrix from: {pae_path}")
        
        return result
    
    def _extract_plddt_from_structure(self, structure) -> Dict[str, float]:
        """
        Extract pLDDT scores from B-factor column.
        
        AF3 stores pLDDT in the B-factor column of the structure.
        
        Returns:
            Dictionary mapping "chain_resid" to pLDDT score
        """
        plddt_scores = {}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Get first atom's B-factor as pLDDT
                    atoms = list(residue.get_atoms())
                    if atoms:
                        bfactor = atoms[0].get_bfactor()
                        key = f"{chain.get_id()}_{residue.get_id()[1]}"
                        plddt_scores[key] = bfactor
        
        return plddt_scores
    
    def _find_pae_file(self, cif_path: str) -> Optional[str]:
        """
        Find associated PAE JSON file based on CIF file path.
        
        AF3 typically outputs files with patterns like:
        - fold_xxx_model_0.cif
        - fold_xxx_full_data_0.json (contains PAE)
        """
        cif_dir = os.path.dirname(cif_path)
        cif_name = os.path.basename(cif_path)
        
        # Try common PAE file patterns
        patterns = [
            cif_name.replace('.cif', '_full_data.json'),
            cif_name.replace('_model_', '_full_data_').replace('.cif', '.json'),
            cif_name.replace('.cif', '_pae.json'),
        ]
        
        for pattern in patterns:
            pae_path = os.path.join(cif_dir, pattern)
            if os.path.exists(pae_path):
                return pae_path
        
        # Also check for any *full_data*.json in the same directory
        import glob
        full_data_files = glob.glob(os.path.join(cif_dir, '*full_data*.json'))
        if full_data_files:
            return full_data_files[0]
        
        return None
    
    def _parse_pae_json(self, json_path: str) -> Optional[np.ndarray]:
        """
        Parse PAE matrix from AF3 JSON output.
        
        Returns:
            2D numpy array of PAE values, or None if parsing fails
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # AF3 full_data format typically has 'pae' key
            if 'pae' in data:
                return np.array(data['pae'])
            
            # Alternative formats
            if 'predicted_aligned_error' in data:
                return np.array(data['predicted_aligned_error'])
            
            # Nested format
            if isinstance(data, list) and len(data) > 0:
                if 'pae' in data[0]:
                    return np.array(data[0]['pae'])
            
            return None
            
        except Exception as e:
            print(f"[AF3Handler] Warning: Could not parse PAE from {json_path}: {e}")
            return None
    
    def generate_validation_json(self, 
                                 designs: List[Dict[str, Any]],
                                 job_name_prefix: str = "validation") -> List[str]:
        """
        Generate AF3 JSON files for validating designed sequences.
        
        Args:
            designs: List of design dictionaries with 'protein_seq' and 'na_seq' keys
            job_name_prefix: Prefix for job names
            
        Returns:
            List of generated JSON file paths
        """
        json_paths = []
        
        for i, design in enumerate(designs):
            job_name = f"{job_name_prefix}_design_{i+1}"
            
            protein_seq = design.get('protein_seq', '')
            na_seq = design.get('na_seq', '')
            na_type = design.get('na_type', 'dna')
            
            if protein_seq and na_seq:
                json_path = self.generate_json(
                    job_name=job_name,
                    protein_seq=protein_seq,
                    na_seq=na_seq,
                    na_type=na_type
                )
                json_paths.append(json_path)
        
        print(f"[AF3Handler] Generated {len(json_paths)} validation JSON files")
        return json_paths
    
    def extract_sequences_from_pdb(self, pdb_path: str) -> Dict[str, str]:
        """
        Extract protein and nucleic acid sequences from a PDB file.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Dictionary with chain IDs as keys and sequences as values
        """
        from utils.pdb_utils import parse_pdb, get_chain_sequence, is_nucleic_acid, is_protein
        
        structure = parse_pdb(pdb_path)
        sequences = {}
        
        for model in structure:
            for chain in model:
                residues = list(chain.get_residues())
                if not residues:
                    continue
                
                # Determine if protein or NA
                first_res = residues[0]
                if is_nucleic_acid(first_res):
                    seq = get_chain_sequence(chain, na_mode=True)
                    sequences[chain.get_id()] = {"sequence": seq, "type": "na"}
                elif is_protein(first_res):
                    seq = get_chain_sequence(chain, na_mode=False)
                    sequences[chain.get_id()] = {"sequence": seq, "type": "protein"}
        
        return sequences

