"""
pH-Responsive Nucleic Acid Design Pipeline - Configuration
===========================================================
Central configuration for all paths and parameters.

Note: APBS-based energy calculation has been deprecated.
      Using Global Cytosine Network Score (GCNS) instead.
"""

import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()

NA_MPNN_DIR = os.environ.get(
    "NA_MPNN_DIR", 
    str(PROJECT_ROOT.parent)
)

PATHS = {
    # NA-MPNN related paths
    "NA_MPNN_DIR": NA_MPNN_DIR,
    "NA_MPNN_INFERENCE": os.path.join(NA_MPNN_DIR, "inference", "run.py"),
    "NA_MPNN_DESIGN_SCRIPT": os.path.join(NA_MPNN_DIR, "design_na_with_fixed_positions.py"),
    "NA_MPNN_MODEL": os.path.join(NA_MPNN_DIR, "models", "design_model", "s_19137.pt"),
    
    # Python executable
    "PYTHON_EXEC": os.environ.get("PYTHON_EXEC", "python"),
    
    # Output directories
    "OUTPUT_AF3": os.path.join(PROJECT_ROOT, "outputs", "01_af3_initial"),
    "OUTPUT_NAMPNN": os.path.join(PROJECT_ROOT, "outputs", "02_nampnn_designs"),
    "OUTPUT_REPORT": os.path.join(PROJECT_ROOT, "outputs", "04_final_report"),
    
    # Input directory
    "INPUT_DIR": os.path.join(PROJECT_ROOT, "inputs"),
}

# ============================================================================
# Physical/Chemical Parameters (For Reference)
# ============================================================================

PARAMS = {
    # Cytosine intrinsic pKa
    "PKA_C_BASE": 4.5,
    
    # Target pH conditions
    "PH_NEUTRAL": 7.4,
    "PH_ACIDIC": 5.5,
    
    # Temperature
    "TEMP_K": 298.15,
    
    # Distance thresholds (Angstrom) for C·C+ pair analysis
    "CC_PAIR_CLOSE_MAX": 6.0,     # Close pair threshold
    "CC_PAIR_MEDIUM_MAX": 10.0,   # Medium pair threshold
    
    # Site selection parameters
    "CONTACT_DISTANCE": 5.0,      # Max distance for protein-NA contact
    "LOCAL_ENV_RADIUS": 8.0,      # Radius for local environment analysis
    
    # Quality thresholds
    "PLDDT_THRESHOLD": 70.0,
}

# ============================================================================
# NA-MPNN Design Parameters
# ============================================================================

NAMPNN_PARAMS = {
    "NUM_SEQUENCES": 50,
    "BATCH_SIZE": 10,
    "TEMPERATURE": 0.3,
    "DESIGN_NA_ONLY": True,
    "NA_SHARED_TOKENS": True,
}

# ============================================================================
# Residue Type Mappings
# ============================================================================

DNA_RESIDUES = {"DA", "DC", "DG", "DT"}
RNA_RESIDUES = {"A", "C", "G", "U"}
NA_RESIDUES = DNA_RESIDUES | RNA_RESIDUES

POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}

# Key atoms for C·C+ pair analysis
NA_N3_ATOMS = {
    "DC": "N3",
    "DT": "N3",
    "C": "N3",
    "U": "N3",
}

# Proton acceptor atoms in nucleic acids (for site selection)
NA_PROTON_ACCEPTORS = {
    "DA": "N1",   # Adenine N1
    "DG": "N7",   # Guanine N7
    "DC": "N3",   # Cytosine N3 (key for pH response)
    "DT": "N3",   # Thymine N3
    "A": "N1",
    "G": "N7",
    "C": "N3",
    "U": "N3",
}

# Positive charge atoms in proteins (for complex mode)
PROTEIN_POSITIVE_ATOMS = {
    "LYS": ["NZ"],
    "ARG": ["NH1", "NH2", "NE"],
    "HIS": ["ND1", "NE2"],
}

# ============================================================================
# Utility Functions
# ============================================================================

def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for key in ["OUTPUT_AF3", "OUTPUT_NAMPNN", "OUTPUT_REPORT"]:
        os.makedirs(PATHS[key], exist_ok=True)

def get_path(key: str) -> str:
    """Get a path from configuration."""
    return PATHS.get(key, "")

def get_param(key: str):
    """Get a parameter from configuration."""
    return PARAMS.get(key)
