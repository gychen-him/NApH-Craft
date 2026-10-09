"""
DSSR Analyzer Module for pH-Responsive Nucleic Acid Design
============================================================

Provides interface to x3dna-dssr for detailed structure analysis:
- G-quadruplex detection and annotation
- Base pair identification (including non-canonical pairs)
- Geometric measurements for C·C+ pair potential

Requirements:
- DSSR installed at tools/dssr/x3dna-dssr
- See README.md for installation instructions

Usage:
    from modules.dssr_analyzer import DSSRAnalyzer
    
    analyzer = DSSRAnalyzer()
    result = analyzer.analyze("structure.pdb")
    print(result['g_quadruplex'])
    print(result['base_pairs'])
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Try to import BioPython for coordinate extraction
try:
    from Bio.PDB import PDBParser, MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


class DSSRAnalyzer:
    """
    Wrapper for x3dna-dssr to analyze nucleic acid structures.
    
    Features:
    - G-quadruplex detection with detailed tetrad information
    - Base pair identification (WC, Hoogsteen, wobble, etc.)
    - CC distance measurements for pH-responsive potential
    - **New (2025-01)**: 3DNA-based Base Frame analysis (Buckle, Propeller, Opening, Stagger)
    - **New (2025-01)**: Geometric constraints for C·C+ pairing (Stagger < 1.0Å, C1'-C1' 8.5-12.5Å)
    - **New (2025-01)**: DSSR-integrated stacking detection to filter out non-pairing cytosines
    """
    
    # Default DSSR path relative to project root
    DEFAULT_DSSR_PATH = "tools/dssr/x3dna-dssr"
    
    def __init__(self, dssr_path: Optional[str] = None):
        """
        Initialize DSSRAnalyzer.
        
        Args:
            dssr_path: Path to x3dna-dssr executable. If None, uses default.
        """
        if dssr_path:
            self.dssr_path = dssr_path
        else:
            # Try to find DSSR relative to this file
            module_dir = Path(__file__).parent
            project_root = module_dir.parent.parent
            default_path = project_root / self.DEFAULT_DSSR_PATH
            
            if default_path.exists():
                self.dssr_path = str(default_path)
            else:
                # Try environment variable
                self.dssr_path = os.environ.get("DSSR_PATH", self.DEFAULT_DSSR_PATH)
        
        self._check_dssr()
    
    def _check_dssr(self) -> bool:
        """Check if DSSR is available."""
        if not os.path.isfile(self.dssr_path):
            print(f"Warning: DSSR not found at {self.dssr_path}")
            print("Please install DSSR following README.md instructions.")
            return False
        
        if not os.access(self.dssr_path, os.X_OK):
            print(f"Warning: DSSR at {self.dssr_path} is not executable")
            return False
        
        return True
    
    def analyze(self, 
                structure_path: str,
                output_json: bool = True,
                analyze_cc_distances: bool = True) -> Dict[str, Any]:
        """
        Analyze a nucleic acid structure using DSSR.
        
        Args:
            structure_path: Path to PDB or CIF file
            output_json: Whether to parse JSON output (default True)
            analyze_cc_distances: Whether to analyze C-C distances (default True)
            
        Returns:
            Dict with analysis results:
            - dssr_output: raw DSSR output
            - g_quadruplex: G-quadruplex information
            - base_pairs: list of base pairs
            - cc_analysis: C-C distance analysis (if analyze_cc_distances=True)
            - sequence: extracted sequence
        """
        if not os.path.isfile(structure_path):
            return {"error": f"Structure file not found: {structure_path}"}
        
        result = {
            "structure_file": structure_path,
            "success": False,
        }
        
        # Run DSSR
        dssr_result = self._run_dssr(structure_path, output_json)
        result["dssr_raw"] = dssr_result.get("raw_output", "")
        
        if not dssr_result.get("success"):
            result["error"] = dssr_result.get("error", "DSSR execution failed")
            return result
        
        # Parse JSON output if available
        if output_json and dssr_result.get("json_output"):
            parsed = self._parse_dssr_json(dssr_result["json_output"])
            result.update(parsed)
        
        # Analyze C-C distances if requested
        if analyze_cc_distances and HAS_BIOPYTHON:
            cc_analysis = self._analyze_cc_distances(structure_path)
            result["cc_analysis"] = cc_analysis
        
        result["success"] = True
        return result
    
    def _run_dssr(self, structure_path: str, output_json: bool = True) -> Dict[str, Any]:
        """
        Execute DSSR on a structure file.
        
        Args:
            structure_path: Path to structure file
            output_json: Whether to output JSON format
            
        Returns:
            Dict with execution results
        """
        # Build command - DSSR uses --key=val format
        cmd = [self.dssr_path, f"--input={structure_path}"]
        
        if output_json:
            cmd.append("--json")
        
        # Add options for detailed analysis
        cmd.extend([
            "--more",           # More detailed output
            "--non-pair",       # Include non-canonical pairs
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": result.returncode == 0,
                "raw_output": result.stdout,
                "json_output": result.stdout if output_json else None,
                "stderr": result.stderr,
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "DSSR timeout (60s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_dssr_json(self, json_str: str) -> Dict[str, Any]:
        """
        Parse DSSR JSON output.
        
        Args:
            json_str: JSON string from DSSR
            
        Returns:
            Parsed information
        """
        result = {
            "sequence": None,
            "n_nucleotides": 0,
            "base_pairs": [],
            "g_quadruplex": None,
            "helices": [],
            "stems": [],
        }
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # DSSR may output text before JSON, try to find JSON block
            try:
                json_start = json_str.find('{')
                if json_start >= 0:
                    data = json.loads(json_str[json_start:])
                else:
                    return result
            except:
                return result
        
        # Extract sequence and frames
        if "nts" in data:
            nts = data["nts"]
            result["n_nucleotides"] = len(nts)
            # Build sequence and store frames
            seq_parts = []
            result["nt_frames"] = {}
            for nt in nts:
                nt_id = nt.get("nt_id", "")
                nt_name = nt.get("nt_name", "")
                # Extract base
                if nt_name.startswith("D"):
                    seq_parts.append(nt_name[1])
                else:
                    seq_parts.append(nt_name[0] if nt_name else "?")
                
                # Store frame if available
                if "frame" in nt:
                    result["nt_frames"][nt_id] = nt["frame"]
                    
            result["sequence"] = "".join(seq_parts)
        
        # Extract base pairs with geometry
        if "pairs" in data:
            for pair in data["pairs"]:
                result["base_pairs"].append({
                    "nt1": pair.get("nt1", ""),
                    "nt2": pair.get("nt2", ""),
                    "bp_type": pair.get("bp", ""),
                    "name": pair.get("name", ""),
                    "saenger": pair.get("Saenger", ""),
                    "LW": pair.get("LW", ""),
                    "shear": pair.get("shear", 0.0),
                    "stretch": pair.get("stretch", 0.0),
                    "stagger": pair.get("stagger", 0.0),
                    "buckle": pair.get("buckle", 0.0),
                    "propeller": pair.get("propeller", 0.0),
                    "opening": pair.get("opening", 0.0),
                })
        
        # Extract G-quadruplex information
        # Use a set to track unique tetrads by their nucleotide IDs to avoid double counting
        unique_tetrad_nts = set()
        final_tetrads = []

        # Method 1: Strict G4tetrads
        if "G4tetrads" in data:
            for tetrad in data["G4tetrads"]:
                nts_list = tetrad.get("nts_long", "").split(",")
                sorted_nts = ",".join(sorted(nts_list))
                if sorted_nts not in unique_tetrad_nts:
                    unique_tetrad_nts.add(sorted_nts)
                    final_tetrads.append({
                        "nts": nts_list,
                        "type": "G-tetrad",
                        "planarity": tetrad.get("planarity"),
                    })
        
        # Method 2: Check multiplets for GGGG (backup for Method 1)
        if "multiplets" in data:
            for mult in data["multiplets"]:
                nts_str = mult.get("nts_long", mult.get("nts", ""))
                nts_list = nts_str.split(",")
                # Count G's in the multiplet
                g_count = sum(1 for nt in nts_list if 'G' in nt.upper())
                # Relaxed: Allow GCGG or GGGG (at least 3 G's)
                if g_count >= 3 and len(nts_list) >= 4:
                    sorted_nts = ",".join(sorted(nts_list))
                    if sorted_nts not in unique_tetrad_nts:
                        unique_tetrad_nts.add(sorted_nts)
                        final_tetrads.append({
                            "nts": nts_list,
                            "type": "G-multiplet" if g_count == 4 else "Mixed-multiplet",
                            "planarity": mult.get("planarity"),
                            "g_count": g_count
                        })
        
        # G4 detected if we have at least 2 unique tetrads
        n_tetrads = len(unique_tetrad_nts)
        has_g4 = n_tetrads >= 2
        
        result["g_quadruplex"] = {
            "detected": has_g4,
            "n_tetrads": n_tetrads,
            "tetrads": final_tetrads,
            "detection_method": "DSSR-Strict" if n_tetrads > 0 else "none",
        }
        
        # Store helices/stems for reference
        if "helices" in data:
            result["helices"] = data["helices"]
        if "stems" in data:
            result["stems"] = data["stems"]
        
        return result
    
    def _analyze_cc_distances(self, structure_path: str) -> Dict[str, Any]:
        """
        Analyze C-C distances for pH-responsive potential.
        
        Uses N3 atoms for distance calculation (key for C·C+ pairing).
        
        Args:
            structure_path: Path to structure file
            
        Returns:
            CC distance analysis results
        """
        result = {
            "n_cytosines": 0,
            "c_positions": [],
            "cc_pairs": [],
            "n_close_pairs": 0,    # < 4.0 Å (can form H-bond)
            "n_medium_pairs": 0,   # 4.0-6.0 Å (potential after conformational change)
            "min_distance": 99.0,
            "avg_distance": None,
            "gcns_score": 0.0,
        }
        
        if not HAS_BIOPYTHON:
            result["error"] = "BioPython not available"
            return result
        
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
                structure = parser.get_structure('s', str(path))
        except Exception as e:
            result["error"] = f"Failed to parse structure: {e}"
            return result
        
        # Collect cytosine N3 coordinates
        c_residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname in ['DC', 'C']:
                        try:
                            n3_coord = residue['N3'].get_coord()
                            resid = residue.get_id()[1]
                            c_residues.append({
                                'chain': chain.get_id(),
                                'resnum': resid,
                                'n3_coord': n3_coord,
                            })
                        except KeyError:
                            continue
            break  # Only first model
        
        result["n_cytosines"] = len(c_residues)
        result["c_positions"] = [(c['chain'], c['resnum']) for c in c_residues]
        
        if len(c_residues) < 2:
            return result
        
        # Calculate all pairwise distances
        distances = []
        close_pairs = []
        medium_pairs = []
        
        for i, c1 in enumerate(c_residues):
            for j, c2 in enumerate(c_residues):
                if j <= i:
                    continue
                
                dist = float(np.linalg.norm(c1['n3_coord'] - c2['n3_coord']))
                distances.append(dist)
                
                pair_info = {
                    'residue1': (c1['chain'], c1['resnum']),
                    'residue2': (c2['chain'], c2['resnum']),
                    'distance': dist,
                }
                
                if dist < 4.0:  # Close enough for H-bond
                    pair_info['category'] = 'close'
                    close_pairs.append(pair_info)
                elif dist < 6.0:  # Medium range
                    pair_info['category'] = 'medium'
                    medium_pairs.append(pair_info)
        
        result["cc_pairs"] = close_pairs + medium_pairs
        result["n_close_pairs"] = len(close_pairs)
        result["n_medium_pairs"] = len(medium_pairs)
        
        if distances:
            result["min_distance"] = float(min(distances))
            result["avg_distance"] = float(np.mean(distances))
            
            # Calculate GCNS score
            # Higher score = better pH-responsive potential
            result["gcns_score"] = (
                len(close_pairs) * 3.0 +      # Close pairs worth most
                len(medium_pairs) * 1.0 +     # Medium pairs have some value
                (10.0 / result["avg_distance"] if result["avg_distance"] > 0 else 0)
            )
        
        return result
    
    def _calculate_bp_geometry(self, frame1: Dict, frame2: Dict) -> Dict[str, float]:
        """
        Calculate 6 standard base-pair parameters (Shear, Stretch, Stagger, Buckle, Propeller, Opening).
        Based on 3DNA geometric definitions.
        """
        # Convert to numpy arrays
        x1, y1, z1 = np.array(frame1['x_axis']), np.array(frame1['y_axis']), np.array(frame1['z_axis'])
        x2, y2, z2 = np.array(frame2['x_axis']), np.array(frame2['y_axis']), np.array(frame2['z_axis'])
        o1, o2 = np.array(frame1['origin']), np.array(frame1['origin']) # Note: origin needs to be in frame
        
        # If origin is not provided, try to estimate from N3 if available, 
        # but better to have it in the frame dictionary.
        # Assuming frame dictionary now contains 'origin'
        if 'origin' not in frame1 or 'origin' not in frame2:
            # Fallback to zero if missing, but results will be inaccurate for translations
            o1, o2 = np.zeros(3), np.zeros(3)
        else:
            o1, o2 = np.array(frame1['origin']), np.array(frame2['origin'])

        # --- Rotational Parameters (BPO) ---
        # 1. Angle between normal vectors (Z axes)
        cos_z = np.clip(np.dot(z1, z2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_z))
        
        # 2. Propeller: rotation about the long axis (Y)
        y_avg = (y1 + y2) / 2.0
        y_avg /= np.linalg.norm(y_avg)
        x1_proj = x1 - np.dot(x1, y_avg) * y_avg
        x2_proj = x2 - np.dot(x2, y_avg) * y_avg
        x1_proj /= np.linalg.norm(x1_proj)
        x2_proj /= np.linalg.norm(x2_proj)
        propeller = np.degrees(np.arccos(np.clip(np.dot(x1_proj, x2_proj), -1.0, 1.0)))
        if np.dot(np.cross(x1_proj, x2_proj), y_avg) < 0: propeller = -propeller
            
        # 3. Buckle: rotation about the short axis (X)
        x_avg = (x1 + x2) / 2.0
        x_avg /= np.linalg.norm(x_avg)
        z1_proj = z1 - np.dot(z1, x_avg) * x_avg
        z2_proj = z2 - np.dot(z2, x_avg) * x_avg
        z1_proj /= np.linalg.norm(z1_proj)
        z2_proj /= np.linalg.norm(z2_proj)
        buckle = np.degrees(np.arccos(np.clip(np.dot(z1_proj, z2_proj), -1.0, 1.0)))
        if np.dot(np.cross(z1_proj, z2_proj), x_avg) < 0: buckle = -buckle
            
        # 4. Opening: rotation about the normal axis (Z)
        z_avg = (z1 + z2) / 2.0
        z_avg /= np.linalg.norm(z_avg)
        y1_proj = y1 - np.dot(y1, z_avg) * z_avg
        y2_proj = y2 - np.dot(y2, z_avg) * z_avg
        y1_proj /= np.linalg.norm(y1_proj)
        y2_proj /= np.linalg.norm(y2_proj)
        opening = np.degrees(np.arccos(np.clip(np.dot(y1_proj, y2_proj), -1.0, 1.0)))
        if np.dot(np.cross(y1_proj, y2_proj), z_avg) < 0: opening = -opening

        # --- Translational Parameters (Shear, Stretch, Stagger) ---
        # Vector between origins
        r = o2 - o1
        
        # Mean reference frame
        # (This is a simplified version of the 3DNA mid-frame calculation)
        shear = np.dot(r, x_avg)
        stretch = np.dot(r, y_avg)
        stagger = np.dot(r, z_avg)
            
        return {
            'shear': round(float(shear), 2),
            'stretch': round(float(stretch), 2),
            'stagger': round(float(stagger), 2),
            'buckle': round(float(buckle), 2),
            'propeller': round(float(propeller), 2),
            'opening': round(float(opening), 2),
            'angle': round(float(angle), 2)
        }

    def _analyze_cc_frames(self, structure_path: str, nt_frames: Dict = None) -> Dict[str, Any]:
        """
        Analyze C-C pairing potential using 3DNA Base Frame geometry (Buckle, Propeller, Opening).
        Now includes DSSR stacking detection to exclude stacked pairs.
        """
        result = {
            "n_cytosines": 0,
            "cc_frame_pairs": [],
            "n_planar_pairs": 0,
            "n_pairing_potential": 0,
            "best_pair": None,
            "frame_score": 0.0,
            "stacking_pairs": [],  # 记录哪些C-C对是堆叠关系
        }
        
        if not HAS_BIOPYTHON:
            result["error"] = "BioPython not available"
            return result
        
        # Run DSSR (已默认包含 --non-pair 选项) 获取 stacking 信息
        stacking_set = set()  # 存储有堆叠关系的碱基对 (nt1, nt2)
        dssr_res = self._run_dssr(structure_path, output_json=True)
        if dssr_res.get("success") and dssr_res.get("json_output"):
            parsed = self._parse_dssr_json(dssr_res["json_output"])
            nt_frames = parsed.get("nt_frames", {})
            
            # 解析 nonPairs 中的 stacking 信息
            raw_json = dssr_res["json_output"]
            if isinstance(raw_json, str):
                import json
                try:
                    dssr_data = json.loads(raw_json)
                except:
                    dssr_data = {}
            else:
                dssr_data = raw_json
            
            for np_item in dssr_data.get("nonPairs", []):
                if np_item.get("stacking"):  # 有 stacking 字段说明是堆叠
                    nt1 = np_item.get("nt1", "")
                    nt2 = np_item.get("nt2", "")
                    # 标准化: 按字母序排序，确保 (A,B) 和 (B,A) 等价
                    pair_key = tuple(sorted([nt1, nt2]))
                    stacking_set.add(pair_key)
        
        if not nt_frames:
            result["error"] = "No base frames found"
            return result

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
                structure = parser.get_structure('s', str(path))
        except Exception as e:
            result["error"] = f"Failed to parse structure: {e}"
            return result
        
        c_info = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname not in ['DC', 'C']: continue
                    resid = residue.get_id()[1]
                    chain_id = chain.get_id()
                    nt_id = f"{chain_id}.{resname}{resid}"
                    if nt_id not in nt_frames: nt_id = f"{chain_id}.C{resid}"
                    if nt_id in nt_frames:
                        try:
                            n3_coord = residue['N3'].get_coord()
                            # 获取C1'原子坐标 (糖环原子，用于标准碱基对距离测量)
                            c1p_coord = None
                            for atom_name in ["C1'", "C1*"]:
                                if atom_name in residue:
                                    c1p_coord = residue[atom_name].get_coord()
                                    break
                            c_info.append({
                                'id': nt_id, 'chain': chain_id, 'resnum': resid,
                                'frame': nt_frames[nt_id], 'n3_coord': n3_coord,
                                'c1p_coord': c1p_coord  # C1' 坐标
                            })
                        except KeyError: continue
            break
        
        result["n_cytosines"] = len(c_info)
        result["stacking_pairs"] = list(stacking_set)  # 记录所有堆叠对
        if len(c_info) < 2: return result
        
        cc_pairs = []
        planar_count = 0
        potential_count = 0
        stacking_count = 0
        best_pair = None
        best_score = -999
        
        for i, c1 in enumerate(c_info):
            for j, c2 in enumerate(c_info):
                if j <= i: continue
                geom = self._calculate_bp_geometry(c1['frame'], c2['frame'])
                n3_dist = float(np.linalg.norm(c1['n3_coord'] - c2['n3_coord']))
                
                # 计算 C1'-C1' 距离 (标准碱基对距离指标)
                c1p_dist = None
                if c1.get('c1p_coord') is not None and c2.get('c1p_coord') is not None:
                    c1p_dist = float(np.linalg.norm(c1['c1p_coord'] - c2['c1p_coord']))
                
                # 检查这对C是否被DSSR识别为堆叠关系
                # DSSR使用格式如 "A.DC1" 或 "A.DG5"
                nt1_id = c1['id']
                nt2_id = c2['id']
                pair_key = tuple(sorted([nt1_id, nt2_id]))
                is_stacking = pair_key in stacking_set
                
                # 也检查相邻位置（相邻C几乎总是堆叠）
                is_adjacent = abs(c1['resnum'] - c2['resnum']) == 1 and c1['chain'] == c2['chain']
                
                if is_stacking or is_adjacent:
                    stacking_count += 1
                
                # Updated Thresholds for pH-responsive C-C+ pairing (2025-01)
                # 核心逻辑：
                #   1. C1'-C1' 距离 (7.0-15.0Å) - 同时排除了堆叠/相邻的C（距离太近<7Å）
                #   2. Stagger < 1.0Å - 确保两个 C 在同一平面高度
                #   3. BPO 欧式距离作为排序指标
                is_planar = abs(geom['stagger']) < 1.0
                
                # C1'-C1' 距离约束：8.5-12.5Å（标准碱基配对距离范围）
                is_c1p_valid = (c1p_dist is not None and 8.5 <= c1p_dist <= 12.5)
                
                has_pairing_potential = is_planar and is_c1p_valid
                
                # 计算 BPO 欧式距离（用于排序，越小越好）
                bpo_score = (geom['buckle']**2 + geom['propeller']**2 + geom['opening']**2) ** 0.5
                
                if is_planar: planar_count += 1
                if has_pairing_potential: potential_count += 1
                
                # pair_score: 符合条件的用 -bpo_score（越小越好），不符合的直接 -999
                pair_score = -bpo_score if has_pairing_potential else -999
                
                pair_info = {
                    'residue1': (c1['chain'], c1['resnum']),
                    'residue2': (c2['chain'], c2['resnum']),
                    'n3_distance': round(n3_dist, 2),
                    'c1p_distance': round(c1p_dist, 2) if c1p_dist else None,  # C1'-C1' 距离
                    'buckle': geom['buckle'], 'propeller': geom['propeller'], 'opening': geom['opening'],
                    'shear': geom['shear'], 'stretch': geom['stretch'], 'stagger': geom['stagger'],
                    'bpo_score': round(bpo_score, 2),  # BPO 欧式距离
                    'angle': geom['angle'], 'is_planar': is_planar,
                    'is_c1p_valid': is_c1p_valid,  # C1'-C1' 距离在合理范围内
                    'is_stacking': is_stacking,  # DSSR检测到的堆叠
                    'is_adjacent': is_adjacent,  # 相邻位置
                    'has_pairing_potential': has_pairing_potential,
                    'pair_score': round(pair_score, 2),
                }
                cc_pairs.append(pair_info)
                
                # 只有非堆叠的配对才能成为最佳配对
                if has_pairing_potential and pair_score > best_score:
                    best_score = pair_score
                    best_pair = pair_info
        
        cc_pairs.sort(key=lambda x: x['pair_score'], reverse=True)
        result.update({
            "cc_frame_pairs": cc_pairs, 
            "n_planar_pairs": planar_count,
            "n_stacking_pairs": stacking_count,
            "n_pairing_potential": potential_count, 
            "best_pair": best_pair, 
            "frame_score": round(best_score, 2)
        })
        return result
    
    def analyze_with_frames(self, structure_path: str) -> Dict[str, Any]:
        """
        Full analysis including Frame-based C-C+ pairing potential.
        
        Args:
            structure_path: Path to structure file
            
        Returns:
            Complete analysis including DSSR, CC distances, and Frame analysis
        """
        # Run standard analysis
        result = self.analyze(structure_path)
        
        # Add Frame-based analysis
        frame_analysis = self._analyze_cc_frames(structure_path)
        result["cc_frame_analysis"] = frame_analysis
        
        return result
    
    def get_frame_summary(self, analysis_result: Dict) -> str:
        """
        Generate human-readable Frame-based C-C+ analysis summary with 3DNA parameters.
        """
        lines = []
        frame = analysis_result.get("cc_frame_analysis", {})
        n_c = frame.get("n_cytosines", 0)
        n_planar = frame.get("n_planar_pairs", 0)
        n_potential = frame.get("n_pairing_potential", 0)
        frame_score = frame.get("frame_score", 0.0)
        
        lines.append(f"Frame-Based C-C+ Analysis ({n_c} cytosines):")
        lines.append(f"  Planar pairs (|Stagger| < 1.0Å): {n_planar}")
        lines.append(f"  Pairing potential (planar + close): {n_potential}")
        lines.append(f"  Best Frame Score: {frame_score:.2f}")
        
        best = frame.get("best_pair")
        if best:
            r1, r2 = best['residue1'], best['residue2']
            lines.append(f"\n  Best C-C+ candidate: C{r1[1]} - C{r2[1]}")
            lines.append(f"    N3-N3 distance: {best['n3_distance']:.2f}Å")
            lines.append(f"    Stagger: {best.get('stagger', 0):.2f}Å")
            lines.append(f"    Buckle: {best['buckle']:.1f}°")
            lines.append(f"    Propeller: {best['propeller']:.1f}°")
            lines.append(f"    Opening: {best['opening']:.1f}°")
            lines.append(f"    Planar: {'✓ YES' if best['is_planar'] else '✗ NO'}")
        
        pairs = frame.get("cc_frame_pairs", [])
        if len(pairs) > 1:
            lines.append(f"\n  Top C-C pairs by Frame Score:")
            for p in pairs[:5]:
                r1, r2 = p['residue1'], p['residue2']
                status = "✓" if p['has_pairing_potential'] else "○"
                lines.append(
                    f"    {status} C{r1[1]}-C{r2[1]}: "
                    f"N3={p['n3_distance']:.1f}Å, S={p.get('stagger', 0):.1f}Å, "
                    f"B={p['buckle']:.0f}°, P={p['propeller']:.0f}°, O={p['opening']:.0f}°, "
                    f"score={p['pair_score']:.1f}"
                )
        return "\n".join(lines)
    
    def get_g4_summary(self, analysis_result: Dict) -> str:
        """
        Generate human-readable G-quadruplex summary.
        
        Args:
            analysis_result: Result from analyze()
            
        Returns:
            Summary string
        """
        lines = []
        
        g4 = analysis_result.get("g_quadruplex")
        if g4 and g4.get("detected"):
            lines.append(f"✓ G-Quadruplex detected: {g4['n_tetrads']} tetrad(s)")
            for i, tetrad in enumerate(g4.get("tetrads", []), 1):
                nts = ", ".join(tetrad.get("nts", []))
                lines.append(f"  Tetrad {i}: {nts}")
        else:
            lines.append("✗ No G-Quadruplex detected")
        
        return "\n".join(lines)
    
    def get_cc_summary(self, analysis_result: Dict) -> str:
        """
        Generate human-readable C-C distance summary.
        
        Args:
            analysis_result: Result from analyze()
            
        Returns:
            Summary string
        """
        lines = []
        
        cc = analysis_result.get("cc_analysis", {})
        n_c = cc.get("n_cytosines", 0)
        n_close = cc.get("n_close_pairs", 0)
        n_medium = cc.get("n_medium_pairs", 0)
        min_dist = cc.get("min_distance", 99.0)
        gcns = cc.get("gcns_score", 0.0)
        
        lines.append(f"Cytosine Analysis ({n_c} residues):")
        
        if n_close > 0:
            lines.append(f"  ✓ {n_close} close C-C pair(s) (<4.0Å) - can form C·C+ H-bond")
        if n_medium > 0:
            lines.append(f"  ○ {n_medium} medium C-C pair(s) (4-6Å) - potential with conformational change")
        
        if n_close == 0 and n_medium == 0:
            lines.append("  ✗ No C-C pairs within 6Å")
        
        lines.append(f"  Min distance: {min_dist:.2f}Å")
        lines.append(f"  GCNS Score: {gcns:.2f}")
        
        # Show top pairs
        pairs = cc.get("cc_pairs", [])
        if pairs:
            lines.append("\n  Top C-C pairs:")
            for p in sorted(pairs, key=lambda x: x['distance'])[:5]:
                r1, r2 = p['residue1'], p['residue2']
                lines.append(f"    C{r1[1]} - C{r2[1]}: {p['distance']:.2f}Å ({p['category']})")
        
        return "\n".join(lines)


def analyze_structure(structure_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to analyze a structure file.
    
    Args:
        structure_path: Path to PDB/CIF file
        verbose: Print summary to stdout
        
    Returns:
        Analysis results
    """
    analyzer = DSSRAnalyzer()
    result = analyzer.analyze(structure_path)
    
    if verbose and result.get("success"):
        print(f"\n{'='*60}")
        print(f"DSSR Analysis: {Path(structure_path).name}")
        print('='*60)
        print(f"Sequence: {result.get('sequence', 'N/A')}")
        print(f"Nucleotides: {result.get('n_nucleotides', 0)}")
        print()
        print(analyzer.get_g4_summary(result))
        print()
        print(analyzer.get_cc_summary(result))
        print('='*60)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dssr_analyzer.py <structure.pdb/cif>")
        print("\nThis module requires DSSR to be installed.")
        print("See README.md for installation instructions.")
        sys.exit(1)
    
    result = analyze_structure(sys.argv[1])
    
    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)
