"""
Structure Validator for pH-Responsive Nucleic Acid Design.
============================================================
Validates G-Quadruplex formation and C·C+ base pair presence
using geometric criteria (without DSSR dependency).

References:
- G-Quadruplex: Hoogsteen H-bonds, G-tetrad planarity, stacking distance
- C·C+ pair: N3-N3 distance for protonated cytosine pairs
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from itertools import combinations

try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Residue import Residue
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ============================================================================
# Constants for G-Quadruplex Detection
# ============================================================================

# G-tetrad geometric parameters
G_TETRAD_PARAMS = {
    # Hoogsteen H-bond distance thresholds (Å)
    'hoogsteen_n7_n2_max': 3.2,  # N7(i) - N2(j) hydrogen bond
    'hoogsteen_o6_n1_max': 3.2,  # O6(i) - N1(j) hydrogen bond
    
    # Planarity threshold (Å) - max deviation from plane
    'planarity_threshold': 1.0,
    
    # Stacking distance between G-tetrads (Å)
    'stacking_min': 3.0,
    'stacking_max': 4.0,
    
    # G-tetrad center distance threshold
    'center_distance_max': 8.0,
}

# C·C+ pair parameters
CC_PAIR_PARAMS = {
    # N3-N3 distance for C·C+ Hoogsteen pair (Å)
    'n3_n3_min': 2.5,
    'n3_n3_max': 3.5,
    
    # Alternative: use C4 atoms for detection
    'c4_c4_min': 3.0,
    'c4_c4_max': 4.5,
}

# Atom names for nucleobases
G_ATOMS = ['N1', 'N2', 'N7', 'O6', 'C8', 'N9']  # Guanine key atoms
C_ATOMS = ['N1', 'N3', 'N4', 'O2', 'C4', 'C5', 'C6']  # Cytosine key atoms


# ============================================================================
# Utility Functions
# ============================================================================

def get_atom_coord(residue: "Residue", atom_name: str) -> Optional[np.ndarray]:
    """Get coordinate of a specific atom in a residue."""
    try:
        return residue[atom_name].get_coord()
    except KeyError:
        return None


def calculate_plane_deviation(points: np.ndarray) -> float:
    """
    Calculate the maximum deviation of points from their best-fit plane.
    
    Args:
        points: Nx3 array of 3D coordinates
        
    Returns:
        Maximum distance from the best-fit plane
    """
    if len(points) < 3:
        return 0.0
    
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # SVD to find the normal vector (smallest singular value)
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Normal to the best-fit plane
    
    # Calculate distances from plane
    distances = np.abs(np.dot(centered, normal))
    return np.max(distances)


def distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(coord1 - coord2)


# ============================================================================
# G-Quadruplex Detection
# ============================================================================

class GQuadruplexDetector:
    """
    Detects G-Quadruplex structures based on geometric features.
    
    Detection criteria:
    1. Planarity: 4 guanines with O6/N1 atoms in a plane
    2. Hoogsteen H-bonds: N7-N1/N2 distances < 3.2 Å
    3. Stacking: Two G-tetrads with ~3.4 Å vertical distance
    """
    
    def __init__(self, params: Dict = None):
        self.params = params or G_TETRAD_PARAMS
    
    def detect(self, structure: "Structure") -> Dict[str, Any]:
        """
        Detect G-Quadruplex in a structure.
        
        Returns:
            Dict with detection results:
            - has_gquadruplex: bool
            - g_tetrads: list of detected G-tetrads
            - stacked_tetrads: list of stacked tetrad pairs
            - confidence: float (0-1)
        """
        # Collect all guanine residues
        g_residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname in ['DG', 'G']:
                        g_residues.append({
                            'residue': residue,
                            'chain': chain.get_id(),
                            'resnum': residue.get_id()[1],
                        })
        
        if len(g_residues) < 4:
            return {
                'has_gquadruplex': False,
                'g_tetrads': [],
                'stacked_tetrads': [],
                'confidence': 0.0,
                'reason': f'Only {len(g_residues)} guanines found (need ≥4)'
            }
        
        # Find potential G-tetrads
        g_tetrads = self._find_g_tetrads(g_residues)
        
        # Find stacked tetrad pairs
        stacked_pairs = self._find_stacked_tetrads(g_tetrads)
        
        # Calculate confidence
        confidence = self._calculate_confidence(g_tetrads, stacked_pairs)
        
        return {
            'has_gquadruplex': len(g_tetrads) >= 1,
            'g_tetrads': g_tetrads,
            'stacked_tetrads': stacked_pairs,
            'n_tetrads': len(g_tetrads),
            'n_stacked_pairs': len(stacked_pairs),
            'confidence': confidence,
        }
    
    def _find_g_tetrads(self, g_residues: List[Dict]) -> List[Dict]:
        """
        Find all valid G-tetrads from guanine residues.
        
        OPTIMIZED: Uses spatial filtering to avoid O(N^4) full enumeration.
        """
        tetrads = []
        n_g = len(g_residues)
        
        if n_g < 4:
            return tetrads
        
        # Pre-compute representative coordinates (N7 atom) for fast filtering
        rep_coords = []
        for r in g_residues:
            coord = get_atom_coord(r['residue'], 'N7')
            if coord is None:
                coord = get_atom_coord(r['residue'], 'C8')  # fallback
            rep_coords.append(coord if coord is not None else np.array([999, 999, 999]))
        rep_coords = np.array(rep_coords)
        
        # Max distance threshold for G-tetrad candidates (Å)
        # A G-tetrad typically has diameter ~12-13Å, so 15Å is a safe upper bound
        MAX_TETRAD_DIAMETER = 15.0
        
        # Try all combinations of 4 guanines with spatial filtering
        for combo in combinations(range(n_g), 4):
            idx_list = list(combo)
            sub_coords = rep_coords[idx_list]
            
            # FAST FILTER 1: Check max pairwise distance
            # If any pair is > MAX_TETRAD_DIAMETER, skip this combination
            max_dist = 0.0
            skip = False
            for i in range(4):
                for j in range(i+1, 4):
                    d = np.linalg.norm(sub_coords[i] - sub_coords[j])
                    if d > MAX_TETRAD_DIAMETER:
                        skip = True
                        break
                    max_dist = max(max_dist, d)
                if skip:
                    break
            
            if skip:
                continue
            
            residues = [g_residues[i] for i in combo]
            
            # FAST FILTER 2: Check Hoogsteen bonds first (cheaper than planarity)
            hoogsteen_bonds = self._count_hoogsteen_bonds(residues)
            if hoogsteen_bonds < 4:
                continue
            
            # EXPENSIVE: Planarity check (only for candidates that passed filters)
            coords = []
            for res_info in residues:
                res = res_info['residue']
                o6 = get_atom_coord(res, 'O6')
                n1 = get_atom_coord(res, 'N1')
                if o6 is not None:
                    coords.append(o6)
                if n1 is not None:
                    coords.append(n1)
            
            if len(coords) < 4:
                continue
            
            coords = np.array(coords)
            planarity = calculate_plane_deviation(coords)
            
            if planarity > self.params['planarity_threshold']:
                continue
            
            # Valid G-tetrad found
            center = np.mean(coords, axis=0)
            tetrads.append({
                'residues': [(r['chain'], r['resnum']) for r in residues],
                'planarity': float(planarity),
                'hoogsteen_bonds': hoogsteen_bonds,
                'center': center.tolist(),
            })
        
        return tetrads
    
    def _count_hoogsteen_bonds(self, residues: List[Dict]) -> int:
        """Count Hoogsteen hydrogen bonds between guanines."""
        bond_count = 0
        
        for i, res_i in enumerate(residues):
            for j, res_j in enumerate(residues):
                if i >= j:
                    continue
                
                r_i = res_i['residue']
                r_j = res_j['residue']
                
                # Check N7(i) - N2(j) bond
                n7_i = get_atom_coord(r_i, 'N7')
                n2_j = get_atom_coord(r_j, 'N2')
                if n7_i is not None and n2_j is not None:
                    d = distance(n7_i, n2_j)
                    if d < self.params['hoogsteen_n7_n2_max']:
                        bond_count += 1
                
                # Check O6(i) - N1(j) bond
                o6_i = get_atom_coord(r_i, 'O6')
                n1_j = get_atom_coord(r_j, 'N1')
                if o6_i is not None and n1_j is not None:
                    d = distance(o6_i, n1_j)
                    if d < self.params['hoogsteen_o6_n1_max']:
                        bond_count += 1
        
        return bond_count
    
    def _find_stacked_tetrads(self, tetrads: List[Dict]) -> List[Dict]:
        """Find pairs of stacked G-tetrads."""
        stacked = []
        
        for i, t1 in enumerate(tetrads):
            for j, t2 in enumerate(tetrads):
                if i >= j:
                    continue
                
                # Check if tetrads share residues (should not)
                res1 = set(t1['residues'])
                res2 = set(t2['residues'])
                if res1 & res2:
                    continue
                
                # Calculate vertical distance between tetrad centers
                c1 = np.array(t1['center'])
                c2 = np.array(t2['center'])
                dist = distance(c1, c2)
                
                if (self.params['stacking_min'] <= dist <= 
                    self.params['stacking_max']):
                    stacked.append({
                        'tetrad1': t1['residues'],
                        'tetrad2': t2['residues'],
                        'stacking_distance': float(dist),
                    })
        
        return stacked
    
    def _calculate_confidence(self, tetrads: List, stacked: List) -> float:
        """Calculate confidence score for G-quadruplex detection."""
        if not tetrads:
            return 0.0
        
        # Base score from number of tetrads
        base_score = min(len(tetrads) / 3, 1.0) * 0.5
        
        # Bonus for stacking
        stack_score = min(len(stacked) / 2, 1.0) * 0.3
        
        # Bonus for planarity quality
        avg_planarity = np.mean([t['planarity'] for t in tetrads])
        planarity_score = max(0, 1 - avg_planarity / self.params['planarity_threshold']) * 0.2
        
        return min(base_score + stack_score + planarity_score, 1.0)


# ============================================================================
# C·C+ Pair Detection
# ============================================================================

class CCPairDetector:
    """
    Detects C·C+ (protonated cytosine) base pairs and calculates GCNS.
    
    At low pH, cytosine N3 becomes protonated, enabling
    C·C+ Hoogsteen base pairing with N3-N3 distance ~2.8-3.2 Å.
    
    This class combines:
    1. Strict CC pair detection (2.5-3.5Å) - for existence check
    2. GCNS-style scoring (close<6Å, medium<10Å) - for ranking
    """
    
    # GCNS thresholds (same as original GCNS implementation)
    GCNS_CLOSE_THRESHOLD = 6.0
    GCNS_MEDIUM_THRESHOLD = 10.0
    
    def __init__(self, params: Dict = None):
        self.params = params or CC_PAIR_PARAMS
    
    def detect(self, structure: "Structure") -> Dict[str, Any]:
        """
        Detect potential C·C+ pairs and calculate integrated GCNS score.
        
        Returns:
            Dict with detection results:
            - has_cc_pairs: bool (strict: 2.5-3.5Å)
            - cc_pairs: list of strict pairs
            - n_pairs: int
            - gcns_score: float (integrated score)
            - n_close_pairs: int (< 6Å)
            - n_medium_pairs: int (6-10Å)
        """
        # Collect all cytosine residues
        c_residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname in ['DC', 'C']:
                        n3_coord = get_atom_coord(residue, 'N3')
                        if n3_coord is not None:
                            c_residues.append({
                                'residue': residue,
                                'chain': chain.get_id(),
                                'resnum': residue.get_id()[1],
                                'n3_coord': n3_coord,
                            })
        
        if len(c_residues) < 2:
            return {
                'has_cc_pairs': False,
                'cc_pairs': [],
                'n_pairs': 0,
                'n_cytosines': len(c_residues),
                'gcns_score': 0.0,
                'n_close_pairs': 0,
                'n_medium_pairs': 0,
                'min_distance': 99.0,
                'reason': f'Only {len(c_residues)} cytosines found (need ≥2)'
            }
        
        # Calculate all pairwise distances
        strict_pairs = []  # 2.5-3.5Å (can form H-bond)
        close_pairs = []   # < 6Å
        medium_pairs = []  # 6-10Å
        all_distances = []
        
        for i, c1 in enumerate(c_residues):
            for j, c2 in enumerate(c_residues):
                if i >= j:
                    continue
                
                n3_dist = distance(c1['n3_coord'], c2['n3_coord'])
                all_distances.append(n3_dist)
                
                pair_info = {
                    'residue1': (c1['chain'], c1['resnum']),
                    'residue2': (c2['chain'], c2['resnum']),
                    'n3_n3_distance': float(n3_dist),
                }
                
                # Strict CC pair (can form Hoogsteen H-bond)
                if self.params['n3_n3_min'] <= n3_dist <= self.params['n3_n3_max']:
                    pair_info['quality'] = self._pair_quality(n3_dist)
                    strict_pairs.append(pair_info)
                
                # GCNS-style categorization
                if n3_dist < self.GCNS_CLOSE_THRESHOLD:
                    close_pairs.append(pair_info)
                elif n3_dist < self.GCNS_MEDIUM_THRESHOLD:
                    medium_pairs.append(pair_info)
        
        # Calculate integrated GCNS score
        min_dist = min(all_distances) if all_distances else 99.0
        avg_dist = np.mean(all_distances) if all_distances else 99.0
        
        gcns_score = (
            len(close_pairs) * 2.0 +
            len(medium_pairs) * 0.5 +
            (10.0 / avg_dist if avg_dist > 0 else 0)
        )
        
        # Bonus for strict pairs (actual H-bond capable)
        gcns_score += len(strict_pairs) * 3.0
        
        return {
            'has_cc_pairs': len(strict_pairs) > 0,
            'cc_pairs': strict_pairs,
            'n_pairs': len(strict_pairs),
            'n_cytosines': len(c_residues),
            # GCNS-style metrics
            'gcns_score': float(gcns_score),
            'n_close_pairs': len(close_pairs),
            'n_medium_pairs': len(medium_pairs),
            'min_distance': float(min_dist),
            'avg_distance': float(avg_dist),
        }
    
    def _pair_quality(self, n3_dist: float) -> str:
        """Assess quality of C·C+ pair based on N3-N3 distance."""
        # Ideal distance is around 2.8-3.0 Å
        if 2.7 <= n3_dist <= 3.1:
            return 'excellent'
        elif 2.5 <= n3_dist <= 3.3:
            return 'good'
        else:
            return 'marginal'


# ============================================================================
# Combined Structure Validator
# ============================================================================

class StructureValidator:
    """
    Combined validator for pH-responsive nucleic acid structures.
    
    Checks for:
    1. G-Quadruplex formation
    2. C·C+ base pair potential
    """
    
    def __init__(self):
        self.g4_detector = GQuadruplexDetector()
        self.cc_detector = CCPairDetector()
    
    def validate(self, structure_path: str) -> Dict[str, Any]:
        """
        Validate a structure file (PDB or CIF).
        
        Args:
            structure_path: Path to PDB or CIF file
            
        Returns:
            Validation results dictionary
        """
        if not HAS_BIOPYTHON:
            return {
                'success': False,
                'error': 'Biopython not installed'
            }
        
        # Parse structure
        path = Path(structure_path)
        try:
            if path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            
            import warnings
            from Bio.PDB.PDBExceptions import PDBConstructionWarning
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=PDBConstructionWarning)
                structure = parser.get_structure('structure', str(path))
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to parse structure: {e}'
            }
        
        # Run detectors
        g4_results = self.g4_detector.detect(structure)
        cc_results = self.cc_detector.detect(structure)
        
        # Determine overall validity for pH-responsive design
        is_valid = g4_results['has_gquadruplex'] and cc_results['has_cc_pairs']
        
        return {
            'success': True,
            'structure_file': str(path),
            'is_valid_ph_responsive': is_valid,
            'g_quadruplex': g4_results,
            'cc_pairs': cc_results,
            'summary': self._generate_summary(g4_results, cc_results),
        }
    
    def _generate_summary(self, g4: Dict, cc: Dict) -> str:
        """Generate human-readable summary."""
        lines = []
        
        # G-Quadruplex summary
        if g4['has_gquadruplex']:
            lines.append(f"✓ G-Quadruplex detected: {g4['n_tetrads']} tetrad(s), "
                        f"{g4['n_stacked_pairs']} stacked pair(s), "
                        f"confidence {g4['confidence']:.2f}")
        else:
            reason = g4.get('reason', 'No valid G-tetrads found')
            lines.append(f"✗ No G-Quadruplex: {reason}")
        
        # C·C+ pair summary (integrated with GCNS)
        if cc['has_cc_pairs']:
            lines.append(f"✓ C·C+ pairs detected: {cc['n_pairs']} strict pair(s) "
                        f"from {cc['n_cytosines']} cytosines")
            lines.append(f"  GCNS Score: {cc.get('gcns_score', 0):.2f} "
                        f"(close:{cc.get('n_close_pairs', 0)}, medium:{cc.get('n_medium_pairs', 0)})")
            lines.append(f"  Min N3-N3: {cc.get('min_distance', 99):.2f} Å")
            for pair in cc['cc_pairs'][:3]:  # Show top 3
                lines.append(f"  - {pair['residue1']} <-> {pair['residue2']}: "
                            f"{pair['n3_n3_distance']:.2f} Å ({pair['quality']})")
        else:
            reason = cc.get('reason', 'No valid C·C+ pairs found')
            lines.append(f"✗ No strict C·C+ pairs: {reason}")
            # Still show GCNS info
            if cc.get('n_close_pairs', 0) > 0:
                lines.append(f"  (But {cc['n_close_pairs']} close pairs < 6Å detected, "
                            f"GCNS={cc.get('gcns_score', 0):.2f})")
        
        return '\n'.join(lines)


# ============================================================================
# CLI Entry Point
# ============================================================================

def validate_structure(structure_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to validate a structure file.
    
    Args:
        structure_path: Path to PDB/CIF file
        verbose: Print summary to stdout
        
    Returns:
        Validation results
    """
    validator = StructureValidator()
    results = validator.validate(structure_path)
    
    if verbose and results['success']:
        print(f"\n{'='*60}")
        print(f"Structure Validation: {Path(structure_path).name}")
        print('='*60)
        print(results['summary'])
        print('='*60)
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python structure_validator.py <structure.pdb/cif>")
        sys.exit(1)
    
    results = validate_structure(sys.argv[1])
    
    if not results['success']:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if results['is_valid_ph_responsive'] else 1)

