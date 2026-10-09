"""
PDB/CIF Parsing and Manipulation Utilities.
============================================
Provides functions for reading, writing, and manipulating structure files.
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("Warning: Biopython not installed. Some functions will be unavailable.")

import numpy as np


def check_pdb_numbering(pdb_path: str) -> Dict[str, Any]:
    """检查PDB残基编号是否从1开始"""
    first_resnum = None
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and len(line) >= 26:
                try:
                    first_resnum = int(line[22:26].strip())
                    break
                except ValueError:
                    continue
    
    return {
        'needs_renumbering': first_resnum is not None and first_resnum != 1,
        'first_residue_num': first_resnum or 1
    }


def renumber_pdb(input_path: str, output_path: str = None) -> str:
    """重新编号PDB残基从1开始（按链独立编号）"""
    if output_path is None:
        output_path = input_path
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    chain_map = {}  # {(chain, orig_resnum): new_resnum}
    chain_counter = {}
    
    new_lines = []
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')) and len(line) >= 26:
            chain = line[21]
            try:
                orig = int(line[22:26].strip())
            except ValueError:
                new_lines.append(line)
                continue
            
            key = (chain, orig)
            if key not in chain_map:
                chain_counter[chain] = chain_counter.get(chain, 0) + 1
                chain_map[key] = chain_counter[chain]
            
            new_lines.append(line[:22] + f"{chain_map[key]:4d}" + line[26:])
        else:
            new_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    return output_path


def parse_pdb(pdb_path: str, structure_id: str = "structure") -> "Structure":
    """
    Parse a PDB file using Biopython.
    
    Args:
        pdb_path: Path to PDB file
        structure_id: Identifier for the structure
        
    Returns:
        Bio.PDB.Structure object
    """
    if not HAS_BIOPYTHON:
        raise ImportError("Biopython is required for PDB parsing")
    
    import warnings
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    
    # Suppress PDB construction warnings (malformed coordinates, etc.)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=PDBConstructionWarning)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(structure_id, pdb_path)
    return structure


def parse_cif(cif_path: str, structure_id: str = "structure") -> "Structure":
    """
    Parse a mmCIF file using Biopython.
    
    Args:
        cif_path: Path to CIF file
        structure_id: Identifier for the structure
        
    Returns:
        Bio.PDB.Structure object
    """
    if not HAS_BIOPYTHON:
        raise ImportError("Biopython is required for CIF parsing")
    
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, cif_path)
    return structure


def cif_to_pdb(cif_path: str, pdb_path: str) -> str:
    """
    Convert a CIF file to PDB format.
    
    Args:
        cif_path: Path to input CIF file
        pdb_path: Path for output PDB file
        
    Returns:
        Path to the generated PDB file
    """
    if not HAS_BIOPYTHON:
        raise ImportError("Biopython is required for CIF to PDB conversion")
    
    structure = parse_cif(cif_path)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    
    return pdb_path


def get_residue_coords(residue: "Residue") -> Dict[str, np.ndarray]:
    """
    Get coordinates of all atoms in a residue.
    
    Args:
        residue: Bio.PDB.Residue object
        
    Returns:
        Dictionary mapping atom names to coordinates
    """
    coords = {}
    for atom in residue.get_atoms():
        coords[atom.get_name()] = atom.get_coord()
    return coords


def get_atom_coord(residue: "Residue", atom_name: str) -> Optional[np.ndarray]:
    """
    Get the coordinate of a specific atom in a residue.
    
    Args:
        residue: Bio.PDB.Residue object
        atom_name: Name of the atom (e.g., 'N3', 'CA')
        
    Returns:
        numpy array of [x, y, z] coordinates, or None if atom not found
    """
    try:
        atom = residue[atom_name]
        return atom.get_coord()
    except KeyError:
        return None


def extract_chain(structure: "Structure", chain_id: str) -> Optional["Chain"]:
    """
    Extract a specific chain from a structure.
    
    Args:
        structure: Bio.PDB.Structure object
        chain_id: Chain identifier
        
    Returns:
        Chain object or None if not found
    """
    for model in structure:
        for chain in model:
            if chain.get_id() == chain_id:
                return chain
    return None


def get_chain_sequence(chain: "Chain", na_mode: bool = False) -> str:
    """
    Get the sequence of a chain.
    
    Args:
        chain: Bio.PDB.Chain object
        na_mode: If True, return nucleic acid single-letter codes
        
    Returns:
        Sequence string
    """
    # Amino acid 3-to-1 letter mapping
    aa_mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    
    # Nucleic acid mapping
    na_mapping = {
        'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T',
        'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U',
    }
    
    mapping = na_mapping if na_mode else aa_mapping
    
    sequence = []
    for residue in chain.get_residues():
        resname = residue.get_resname().strip()
        if resname in mapping:
            sequence.append(mapping[resname])
        elif resname in na_mapping:
            sequence.append(na_mapping[resname])
        else:
            sequence.append('X')
    
    return ''.join(sequence)


def get_residue_by_id(chain: "Chain", res_id: int) -> Optional["Residue"]:
    """
    Get a residue by its sequence ID.
    
    Args:
        chain: Bio.PDB.Chain object
        res_id: Residue sequence number
        
    Returns:
        Residue object or None
    """
    for residue in chain.get_residues():
        if residue.get_id()[1] == res_id:
            return residue
    return None


def is_nucleic_acid(residue: "Residue") -> bool:
    """Check if a residue is a nucleic acid."""
    na_residues = {"DA", "DC", "DG", "DT", "A", "C", "G", "U"}
    return residue.get_resname().strip() in na_residues


def is_protein(residue: "Residue") -> bool:
    """Check if a residue is a protein amino acid."""
    aa_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
        'THR', 'TRP', 'TYR', 'VAL'
    }
    return residue.get_resname().strip() in aa_residues


def get_structure_info(structure: "Structure") -> Dict[str, Any]:
    """
    Get summary information about a structure.
    
    Args:
        structure: Bio.PDB.Structure object
        
    Returns:
        Dictionary with structure information
    """
    info = {
        "n_models": 0,
        "chains": [],
        "n_residues": 0,
        "n_atoms": 0,
    }
    
    for model in structure:
        info["n_models"] += 1
        for chain in model:
            chain_info = {
                "id": chain.get_id(),
                "n_residues": len(list(chain.get_residues())),
                "type": "unknown",
            }
            
            # Determine chain type
            residues = list(chain.get_residues())
            if residues:
                first_res = residues[0]
                if is_nucleic_acid(first_res):
                    chain_info["type"] = "nucleic_acid"
                elif is_protein(first_res):
                    chain_info["type"] = "protein"
            
            info["chains"].append(chain_info)
            info["n_residues"] += chain_info["n_residues"]
        
        for atom in model.get_atoms():
            info["n_atoms"] += 1
    
    return info


def modify_residue_name(pdb_path: str, chain_id: str, res_id: int, 
                        new_name: str, output_path: str) -> str:
    """
    Modify a residue name in a PDB file.
    
    Args:
        pdb_path: Input PDB path
        chain_id: Chain identifier
        res_id: Residue number
        new_name: New residue name (3-letter code)
        output_path: Output PDB path
        
    Returns:
        Path to modified PDB
    """
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            # Parse PDB fixed-width format
            record_chain = line[21]
            try:
                record_res_id = int(line[22:26].strip())
            except ValueError:
                record_res_id = -1
            
            if record_chain == chain_id and record_res_id == res_id:
                # Modify residue name (columns 17-20)
                new_name_padded = f"{new_name:>3}"
                line = line[:17] + new_name_padded + line[20:]
        
        modified_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.writelines(modified_lines)
    
    return output_path

