"""
Mathematical Utilities for pH-NA Design Pipeline.
==================================================
Distance calculations and geometric analysis.
"""

import numpy as np
from typing import Tuple, List


def euclidean_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 3D coordinates.
    
    Args:
        coord1: First coordinate [x, y, z]
        coord2: Second coordinate [x, y, z]
        
    Returns:
        Distance in Angstroms
    """
    return float(np.linalg.norm(np.array(coord1) - np.array(coord2)))


def kj_to_kcal(kj: float) -> float:
    """Convert kJ/mol to kcal/mol."""
    return kj * 0.239006


def calculate_bounding_box(coords: List[np.ndarray], 
                           expansion: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate bounding box for a set of coordinates.
    
    Args:
        coords: List of [x, y, z] coordinates
        expansion: Factor to expand the bounding box
        
    Returns:
        Tuple of (min_coords, max_coords)
    """
    coords_array = np.array(coords)
    min_coords = np.min(coords_array, axis=0)
    max_coords = np.max(coords_array, axis=0)
    
    center = (min_coords + max_coords) / 2
    dims = max_coords - min_coords
    expanded_dims = dims * expansion
    
    new_min = center - expanded_dims / 2
    new_max = center + expanded_dims / 2
    
    return new_min, new_max


def calculate_pairwise_distances(coords: List[np.ndarray]) -> np.ndarray:
    """
    Calculate all pairwise distances between coordinates.
    
    Args:
        coords: List of [x, y, z] coordinates
        
    Returns:
        2D numpy array of distances
    """
    n = len(coords)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(coords[i], coords[j])
            distances[i, j] = d
            distances[j, i] = d
    
    return distances
