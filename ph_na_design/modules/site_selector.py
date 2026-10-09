"""
[Module 1] Switch Site Selector (Unified)
==========================================
Identifies nucleic acid positions suitable for pH-sensitive cytosine introduction.

Supports TWO modes:
1. COMPLEX mode: Protein-NA complexes
   - Selects sites near positively charged residues (Lys, Arg, His)
   
2. NA_ONLY mode: Pure nucleic acid structures (e.g., G-quadruplex)
   - Selects non-essential positions (loops, non-G-run regions)
   - Identifies potential C·C+ base pair positions

Both modes use the same physics principle:
   pH↓ → C protonation → Charge change → ΔΔG (energy change)

Uses Biopython's NeighborSearch for efficient distance calculations.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PARAMS, 
    NA_PROTON_ACCEPTORS, 
    PROTEIN_POSITIVE_ATOMS, 
    POSITIVE_RESIDUES,
    NA_RESIDUES
)
from utils.pdb_utils import parse_pdb, get_atom_coord, is_nucleic_acid, is_protein

try:
    from Bio.PDB import NeighborSearch
    from Bio.PDB.Structure import Structure
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


class SwitchSiteSelector:
    """
    Selects nucleic acid positions for pH-sensitive switch design.
    
    Unified selection for both:
    1. COMPLEX mode: Protein-NA contacts
    2. NA_ONLY mode: G-quadruplex and other NA-only structures
    
    Core physics: All selections target positions where C protonation
    will cause significant ΔΔG upon pH change.
    """
    
    def __init__(self, 
                 contact_distance: float = None,
                 local_env_radius: float = None,
                 mode: str = "auto"):
        """
        Initialize the site selector.
        
        Args:
            contact_distance: Maximum distance for protein-NA contact (default from config)
            local_env_radius: Radius for local environment analysis (default from config)
            mode: Selection mode - "auto", "complex", or "na_only"
        """
        if not HAS_BIOPYTHON:
            raise ImportError("Biopython is required for SwitchSiteSelector")
        
        self.contact_distance = contact_distance or PARAMS["CONTACT_DISTANCE"]
        self.local_env_radius = local_env_radius or PARAMS["LOCAL_ENV_RADIUS"]
        self.mode = mode
    
    def select_sites(self, pdb_path: str, 
                     fixed_positions: List[int] = None,
                     force_c_positions: List[int] = None) -> Dict[str, Any]:
        """
        Analyze structure and select switch sites.
        
        Supports both protein-NA complexes and NA-only structures.
        
        Args:
            pdb_path: Path to the PDB file
            fixed_positions: Positions to keep fixed (e.g., G-runs in G-quadruplex)
            force_c_positions: Positions that must be cytosine (manual override)
            
        Returns:
            Dictionary containing:
            - chain_id: Nucleic acid chain identifier
            - switch_residues: List of selected residue numbers for C introduction
            - fixed_residues: List of positions to keep unchanged
            - reasons: Dict mapping residue ID to selection reason
            - scores: Dict mapping residue ID to score
            - mode: Detection mode used ("complex" or "na_only")
        """
        import warnings
        
        # Parse structure with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                structure = parse_pdb(pdb_path)
            except Exception as e:
                raise ValueError(f"Failed to parse PDB: {e}")
        
        # Identify chains
        na_chains, protein_chains = self._identify_chains(structure)
        
        if not na_chains:
            raise ValueError("No nucleic acid chain found in structure")
        
        # Determine mode
        if self.mode == "auto":
            if protein_chains:
                actual_mode = "complex"
                print("[SiteSelector] Mode: COMPLEX (protein-NA)")
            else:
                actual_mode = "na_only"
                print("[SiteSelector] Mode: NA_ONLY (pure nucleic acid)")
        else:
            actual_mode = self.mode
        
        # Get all atoms for neighbor search
        all_atoms = list(structure.get_atoms())
        ns = NeighborSearch(all_atoms)
        
        if actual_mode == "complex":
            # Complex mode: Find protein-NA contacts
            result = self._select_sites_complex(
                structure, ns, na_chains, protein_chains, fixed_positions
            )
        else:
            # NA-only mode: Analyze structure for C·C+ potential
            result = self._select_sites_na_only(
                structure, ns, na_chains, fixed_positions
            )
        
        result["mode"] = actual_mode
        
        # Apply manual force_c_positions if provided
        if force_c_positions:
            for pos in force_c_positions:
                if pos not in result["switch_residues"]:
                    result["switch_residues"].append(pos)
                    result["reasons"][pos] = "Manually specified"
                    result["scores"][pos] = 10.0  # High priority
            print(f"[SiteSelector] Added manual C positions: {force_c_positions}")
        
        print(f"[SiteSelector] Selected {len(result['switch_residues'])} switch sites: {result['switch_residues']}")
        
        return result
    
    def _select_sites_complex(self, structure, ns, na_chains, protein_chains, 
                               fixed_positions=None) -> Dict[str, Any]:
        """Select sites for protein-NA complex based on contacts."""
        # Find contacts between NA and positive protein residues
        contacts = self._find_positive_contacts(structure, ns, na_chains, protein_chains)
        
        # Score each NA residue
        scored_residues = self._score_residues(structure, contacts, na_chains)
        
        # Select top candidates
        selected = self._select_top_sites(scored_residues)
        
        # Format output
        result = {
            "chain_id": na_chains[0],
            "switch_residues": [r["res_id"] for r in selected],
            "fixed_residues": fixed_positions or [],
            "reasons": {r["res_id"]: r["reason"] for r in selected},
            "scores": {r["res_id"]: r["score"] for r in selected},
            "all_contacts": contacts,
            "all_scored": scored_residues
        }
        
        return result
    
    def _select_sites_na_only(self, structure, ns, na_chains, 
                               fixed_positions=None) -> Dict[str, Any]:
        """
        Select sites for NA-only structures (e.g., G-quadruplex).
        
        Strategy for pH-responsive G-quadruplex design:
        1. Identify G-runs (to be FIXED - structural core)
        2. Select BEST positions for C introduction (for C·C+ base pairs)
        3. Other loop positions: let NA-MPNN design freely
        
        Output categories:
        - fixed_residues: G-runs (must keep as G)
        - force_c_residues: Best positions for C (for pH response)
        - designable_residues: Other loops (let NA-MPNN design)
        """
        chain_id = na_chains[0]
        
        # Get sequence and identify structural features
        sequence_info = self._analyze_na_sequence(structure, chain_id)
        
        # Auto-detect fixed positions (G-runs) if not provided
        if fixed_positions is None:
            fixed_positions = sequence_info["g_run_positions"]
            print(f"[SiteSelector] Auto-detected G-runs to fix: {sorted(fixed_positions)}")
        
        # Analyze potential C·C base pair positions
        cc_pair_candidates = self._find_cc_pair_candidates(structure, chain_id, sequence_info)
        
        # Score positions for C introduction
        scored_residues = []
        designable_positions = []
        
        for res_info in sequence_info["residues"]:
            res_id = res_info["res_id"]
            resname = res_info["resname"]
            
            if res_id in fixed_positions:
                continue  # Skip G-runs
            
            score = 0.0
            reasons = []
            
            # Higher score for loop positions
            if res_info["is_loop"]:
                score += 2.0
                reasons.append("Loop region")
            
            # Higher score for existing non-G bases (easier to mutate)
            if resname not in ["DG", "G"]:
                score += 1.0
                reasons.append(f"Non-G base ({resname})")
            
            # CRITICAL: Check for C·C base pair potential
            if res_id in cc_pair_candidates:
                partner_info = cc_pair_candidates[res_id]
                score += 5.0  # High priority for C·C pair formation
                reasons.append(f"C·C+ pair with {partner_info['partner_id']} ({partner_info['distance']:.1f}Å)")
            
            # Check symmetry (for G-quadruplex, symmetric positions often pair)
            if res_info.get("has_symmetric_partner"):
                score += 2.0
                reasons.append("Symmetric position")
            
            scored_residues.append({
                "chain": chain_id,
                "res_id": res_id,
                "resname": resname,
                "score": score,
                "reason": "; ".join(reasons) if reasons else "Available position",
                "is_cc_candidate": res_id in cc_pair_candidates
            })
            
            designable_positions.append(res_id)
        
        # Sort by score (descending)
        scored_residues.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top C positions (positions that should be forced to C)
        # Prefer pairs for C·C base pair formation
        force_c_positions = self._select_c_positions(scored_residues, cc_pair_candidates)
        
        # Remaining designable positions (let NA-MPNN design freely)
        remaining_designable = [p for p in designable_positions if p not in force_c_positions]
        
        result = {
            "chain_id": chain_id,
            # For compatibility, switch_residues contains the C positions
            "switch_residues": force_c_positions,
            # G-runs: must be kept as G
            "fixed_residues": list(fixed_positions),
            # Positions forced to be C (for pH response)
            "force_c_residues": force_c_positions,
            # Other positions: let NA-MPNN design
            "designable_residues": remaining_designable,
            # All non-G-run positions
            "all_loop_positions": designable_positions,
            # Scoring details
            "reasons": {r["res_id"]: r["reason"] for r in scored_residues},
            "scores": {r["res_id"]: r["score"] for r in scored_residues},
            "cc_pair_candidates": cc_pair_candidates,
            "sequence_info": sequence_info,
            "all_scored": scored_residues
        }
        
        print(f"[SiteSelector] Design strategy:")
        print(f"  - Fixed G-runs: {sorted(fixed_positions)}")
        print(f"  - Force C at: {force_c_positions}")
        print(f"  - NA-MPNN designs: {remaining_designable}")
        
        return result
    
    def _find_cc_pair_candidates(self, structure, chain_id: str, 
                                  sequence_info: Dict) -> Dict[int, Dict]:
        """
        Find positions that could form C·C+ base pairs.
        
        In G-quadruplex, C·C+ pairs typically form between:
        - Residues in different loops that are spatially close
        - Non-adjacent residues with appropriate geometry
        
        Returns:
            Dict mapping res_id to partner information
        """
        cc_candidates = {}
        residues_list = sequence_info["residues"]
        g_run_positions = sequence_info["g_run_positions"]
        
        # Get loop residues (non-G-run)
        loop_residues = [r for r in residues_list if r["res_id"] not in g_run_positions]
        
        if len(loop_residues) < 2:
            return cc_candidates
        
        # Get coordinates for loop residues (using first model only)
        loop_coords = {}
        first_model = next(structure.get_models(), None)
        if first_model is None:
            return cc_candidates
        
        for chain in first_model:
            if chain.get_id() != chain_id:
                continue
            for residue in chain:
                res_id = residue.get_id()[1]
                if res_id in [r["res_id"] for r in loop_residues]:
                    # Get N3 coordinate (proton acceptor for C)
                    # Or use C1' (sugar) for general proximity
                    for atom_name in ["N3", "N1", "C1'"]:
                        coord = get_atom_coord(residue, atom_name)
                        if coord is not None:
                            loop_coords[res_id] = {"coord": coord, "atom": atom_name}
                            break
        
        # Find pairs with appropriate distance for C·C base pairing
        # In G-quadruplex context, C·C+ pairs can span larger distances
        # Typical range: 3-8 Å for base-base interaction
        CC_PAIR_DISTANCE_MIN = 3.0
        CC_PAIR_DISTANCE_MAX = 10.0  # Increased for G-quadruplex loop geometry
        
        # Also consider sequence separation (should be in different loops)
        MIN_SEQ_SEPARATION = 3
        
        for i, res1 in enumerate(loop_residues):
            res1_id = res1["res_id"]
            if res1_id not in loop_coords:
                continue
            
            best_partner = None
            best_distance = float('inf')
            
            for res2 in loop_residues[i+1:]:
                res2_id = res2["res_id"]
                if res2_id not in loop_coords:
                    continue
                
                # Require sequence separation (not adjacent)
                if abs(res1_id - res2_id) < MIN_SEQ_SEPARATION:
                    continue
                
                distance = np.linalg.norm(
                    loop_coords[res1_id]["coord"] - loop_coords[res2_id]["coord"]
                )
                
                if CC_PAIR_DISTANCE_MIN <= distance <= CC_PAIR_DISTANCE_MAX:
                    if distance < best_distance:
                        best_distance = distance
                        best_partner = res2_id
            
            if best_partner is not None:
                cc_candidates[res1_id] = {
                    "partner_id": best_partner,
                    "distance": best_distance
                }
                # Also add reverse mapping
                if best_partner not in cc_candidates:
                    cc_candidates[best_partner] = {
                        "partner_id": res1_id,
                        "distance": best_distance
                    }
        
        if cc_candidates:
            print(f"[SiteSelector] Found {len(cc_candidates)//2} potential C·C pair positions")
        
        return cc_candidates
    
    def _select_c_positions(self, scored_residues: List[Dict], 
                            cc_candidates: Dict[int, Dict],
                            max_c_sites: int = 4) -> List[int]:
        """
        Select best positions for forced C introduction.
        
        Prioritizes:
        1. Positions that can form C·C+ pairs (select as pairs)
        2. High-scoring individual positions
        
        Args:
            scored_residues: Scored and sorted residue list
            cc_candidates: Dict of C·C pair candidates
            max_c_sites: Maximum number of C positions to select
            
        Returns:
            List of residue IDs to force as C
        """
        selected = []
        selected_set = set()
        
        # First, try to select C·C pairs
        for res in scored_residues:
            if len(selected) >= max_c_sites:
                break
            
            res_id = res["res_id"]
            if res_id in selected_set:
                continue
            
            if res_id in cc_candidates:
                partner_id = cc_candidates[res_id]["partner_id"]
                
                # Add both as a pair if we have room
                if partner_id not in selected_set:
                    if len(selected) + 2 <= max_c_sites:
                        selected.append(res_id)
                        selected.append(partner_id)
                        selected_set.add(res_id)
                        selected_set.add(partner_id)
        
        # If no pairs found or room remains, add high-scoring individuals
        for res in scored_residues:
            if len(selected) >= max_c_sites:
                break
            
            res_id = res["res_id"]
            if res_id not in selected_set and res["score"] > 2.0:
                selected.append(res_id)
                selected_set.add(res_id)
        
        return sorted(selected)
    
    def _analyze_na_sequence(self, structure, chain_id: str) -> Dict[str, Any]:
        """
        Analyze nucleic acid sequence for structural features.
        
        Identifies:
        - G-runs (for G-quadruplex)
        - Loop regions
        - Potential C·C base pair positions
        
        Note: Only uses first model for NMR structures.
        """
        residues = []
        sequence = []
        
        # Only use first model (for NMR structures with multiple models)
        first_model = next(structure.get_models(), None)
        if first_model is None:
            return {"sequence": "", "residues": [], "g_run_positions": set(), "n_g_runs": 0}
        
        for chain in first_model:
            if chain.get_id() != chain_id:
                continue
            
            for residue in chain:
                if not is_nucleic_acid(residue):
                    continue
                
                res_id = residue.get_id()[1]
                resname = residue.get_resname().strip()
                base = resname.replace("D", "")  # DA -> A, DG -> G, etc.
                
                residues.append({
                    "res_id": res_id,
                    "resname": resname,
                    "base": base,
                    "is_loop": False,
                    "in_g_run": False
                })
                sequence.append(base)
        
        seq_str = "".join(sequence)
        
        # Identify G-runs (3+ consecutive Gs)
        g_run_positions = set()
        for match in re.finditer(r'G{3,}', seq_str):
            for i in range(match.start(), match.end()):
                if i < len(residues):
                    g_run_positions.add(residues[i]["res_id"])
                    residues[i]["in_g_run"] = True
        
        # Mark loop positions (non-G-run positions)
        for res in residues:
            if not res["in_g_run"]:
                res["is_loop"] = True
        
        return {
            "sequence": seq_str,
            "residues": residues,
            "g_run_positions": g_run_positions,
            "n_g_runs": len(re.findall(r'G{3,}', seq_str))
        }
    
    def _identify_chains(self, structure: Structure) -> Tuple[List[str], List[str]]:
        """
        Identify nucleic acid and protein chains.
        
        Returns:
            Tuple of (na_chain_ids, protein_chain_ids)
        """
        na_chains = []
        protein_chains = []
        
        for model in structure:
            for chain in model:
                residues = list(chain.get_residues())
                if not residues:
                    continue
                
                # Check first few residues to determine type
                na_count = sum(1 for r in residues[:5] if is_nucleic_acid(r))
                prot_count = sum(1 for r in residues[:5] if is_protein(r))
                
                if na_count > prot_count:
                    na_chains.append(chain.get_id())
                elif prot_count > 0:
                    protein_chains.append(chain.get_id())
        
        return na_chains, protein_chains
    
    def _find_positive_contacts(self, 
                                structure: Structure,
                                ns: NeighborSearch,
                                na_chains: List[str],
                                protein_chains: List[str]) -> List[Dict[str, Any]]:
        """
        Find contacts between NA proton acceptors and positive protein atoms.
        
        Returns:
            List of contact dictionaries
        """
        contacts = []
        
        for model in structure:
            for chain in model:
                if chain.get_id() not in na_chains:
                    continue
                
                for residue in chain:
                    if not is_nucleic_acid(residue):
                        continue
                    
                    resname = residue.get_resname().strip()
                    res_id = residue.get_id()[1]
                    
                    # Get proton acceptor atom for this residue type
                    acceptor_name = NA_PROTON_ACCEPTORS.get(resname)
                    if not acceptor_name:
                        continue
                    
                    acceptor_coord = get_atom_coord(residue, acceptor_name)
                    if acceptor_coord is None:
                        continue
                    
                    # Find nearby positive atoms
                    nearby_atoms = ns.search(acceptor_coord, self.contact_distance)
                    
                    for atom in nearby_atoms:
                        parent_res = atom.get_parent()
                        parent_chain = parent_res.get_parent()
                        
                        if parent_chain.get_id() not in protein_chains:
                            continue
                        
                        parent_resname = parent_res.get_resname().strip()
                        if parent_resname not in POSITIVE_RESIDUES:
                            continue
                        
                        atom_name = atom.get_name()
                        positive_atoms = PROTEIN_POSITIVE_ATOMS.get(parent_resname, [])
                        
                        if atom_name in positive_atoms:
                            distance = np.linalg.norm(acceptor_coord - atom.get_coord())
                            
                            contacts.append({
                                "na_chain": chain.get_id(),
                                "na_res_id": res_id,
                                "na_resname": resname,
                                "na_atom": acceptor_name,
                                "prot_chain": parent_chain.get_id(),
                                "prot_res_id": parent_res.get_id()[1],
                                "prot_resname": parent_resname,
                                "prot_atom": atom_name,
                                "distance": distance
                            })
        
        return contacts
    
    def _score_residues(self,
                        structure: Structure,
                        contacts: List[Dict[str, Any]],
                        na_chains: List[str]) -> List[Dict[str, Any]]:
        """
        Score each NA residue based on contacts and local environment.
        
        Scoring criteria:
        1. Number of contacts with positive residues
        2. Minimum distance to positive atoms
        3. Local phosphate density
        """
        # Group contacts by NA residue
        residue_contacts = {}
        for contact in contacts:
            key = (contact["na_chain"], contact["na_res_id"])
            if key not in residue_contacts:
                residue_contacts[key] = []
            residue_contacts[key].append(contact)
        
        scored = []
        
        # Also consider all NA residues (even without contacts)
        all_atoms = list(structure.get_atoms())
        ns = NeighborSearch(all_atoms)
        
        for model in structure:
            for chain in model:
                if chain.get_id() not in na_chains:
                    continue
                
                for residue in chain:
                    if not is_nucleic_acid(residue):
                        continue
                    
                    res_id = residue.get_id()[1]
                    resname = residue.get_resname().strip()
                    key = (chain.get_id(), res_id)
                    
                    # Get contacts for this residue
                    res_contacts = residue_contacts.get(key, [])
                    
                    # Calculate scores
                    contact_score = len(res_contacts) * 2.0
                    
                    min_distance = float('inf')
                    closest_contact = None
                    for c in res_contacts:
                        if c["distance"] < min_distance:
                            min_distance = c["distance"]
                            closest_contact = c
                    
                    if min_distance < float('inf'):
                        distance_score = (self.contact_distance - min_distance) / self.contact_distance * 3.0
                    else:
                        distance_score = 0
                    
                    # Count nearby phosphates (indicates negative local environment)
                    p_atom = get_atom_coord(residue, "P")
                    if p_atom is not None:
                        nearby_atoms = ns.search(p_atom, self.local_env_radius)
                        phosphate_count = sum(1 for a in nearby_atoms if a.get_name() == "P")
                        phosphate_score = phosphate_count * 0.5
                    else:
                        phosphate_score = 0
                    
                    total_score = contact_score + distance_score + phosphate_score
                    
                    # Build reason string
                    if closest_contact:
                        reason = f"Near {closest_contact['prot_resname']}{closest_contact['prot_res_id']} ({min_distance:.1f}Å)"
                    else:
                        reason = "No direct positive contact"
                    
                    scored.append({
                        "chain": chain.get_id(),
                        "res_id": res_id,
                        "resname": resname,
                        "score": total_score,
                        "n_contacts": len(res_contacts),
                        "min_distance": min_distance if min_distance < float('inf') else None,
                        "reason": reason,
                        "contacts": res_contacts
                    })
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        return scored
    
    def _select_top_sites(self, 
                          scored_residues: List[Dict[str, Any]],
                          min_score: float = 1.0,
                          max_sites: int = 10) -> List[Dict[str, Any]]:
        """
        Select top scoring sites for switch design.
        
        Args:
            scored_residues: List of scored residue dictionaries
            min_score: Minimum score threshold
            max_sites: Maximum number of sites to select
            
        Returns:
            List of selected residue dictionaries
        """
        selected = []
        
        for res in scored_residues:
            if res["score"] >= min_score:
                selected.append(res)
                if len(selected) >= max_sites:
                    break
        
        return selected
    
    def _empty_result(self, chain_id: str) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "chain_id": chain_id,
            "switch_residues": [],
            "reasons": {},
            "scores": {},
            "all_contacts": [],
            "all_scored": []
        }
    
    def visualize_contacts(self, 
                           pdb_path: str,
                           selection_result: Dict[str, Any],
                           output_script: str = None) -> str:
        """
        Generate a PyMOL script to visualize selected sites.
        
        Args:
            pdb_path: Path to the PDB file
            selection_result: Output from select_sites()
            output_script: Output path for PyMOL script
            
        Returns:
            PyMOL script content
        """
        script_lines = [
            f'load {pdb_path}',
            'hide all',
            'show cartoon',
            'color gray90, all',
            '',
            '# Color nucleic acid chain',
            f'color cyan, chain {selection_result["chain_id"]}',
            'show sticks, chain ' + selection_result["chain_id"],
            '',
            '# Highlight switch sites',
        ]
        
        for res_id in selection_result["switch_residues"]:
            reason = selection_result["reasons"].get(res_id, "")
            script_lines.append(
                f'color red, chain {selection_result["chain_id"]} and resi {res_id}'
            )
            script_lines.append(
                f'# Site {res_id}: {reason}'
            )
        
        script_lines.extend([
            '',
            '# Show positive residues',
            'select positive, resn LYS+ARG+HIS',
            'show sticks, positive',
            'color blue, positive',
            '',
            'zoom',
        ])
        
        script_content = '\n'.join(script_lines)
        
        if output_script:
            with open(output_script, 'w') as f:
                f.write(script_content)
            print(f"[SiteSelector] PyMOL script saved: {output_script}")
        
        return script_content

