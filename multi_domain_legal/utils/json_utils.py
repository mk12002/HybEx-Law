# utils/json_utils.py
"""
JSON utility functions for safe serialization of NumPy and PyTorch types.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Union


def safe_json_dump(obj: Any, path: Union[str, Path], indent: int = 2, encoding: str = 'utf-8'):
    """
    Safely dump Python object to JSON file, handling NumPy and PyTorch types.
    
    Args:
        obj: Python object to serialize (dict, list, etc.)
        path: File path to write JSON to
        indent: Indentation level for pretty printing (default: 2)
        encoding: File encoding (default: 'utf-8')
    
    Handles:
        - np.integer types (np.int32, np.int64, etc.) -> int
        - np.floating types (np.float32, np.float64, etc.) -> float
        - np.ndarray -> list
        - torch.Tensor -> list (detached from computation graph, moved to CPU)
        - Other non-serializable types -> str (fallback)
    
    Example:
        >>> results = {
        ...     'accuracy': np.float64(0.95),
        ...     'predictions': torch.tensor([1, 0, 1]),
        ...     'counts': np.array([10, 20, 30])
        ... }
        >>> safe_json_dump(results, 'results.json')
    """
    def convert(o):
        """Convert non-JSON-serializable types to serializable ones."""
        # Handle NumPy integer types
        if isinstance(o, (np.integer,)):
            return int(o)
        # Handle NumPy floating types
        if isinstance(o, (np.floating,)):
            return float(o)
        # Handle NumPy arrays
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        # Handle PyTorch tensors
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        # Fallback: convert to string
        return str(o)
    
    # Ensure path is Path object
    path = Path(path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON file with safe conversion
    with open(path, 'w', encoding=encoding) as f:
        json.dump(obj, f, default=convert, indent=indent)
