# utils/repro.py
"""
Reproducibility utility for deterministic/seeded runs.
Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42, deterministic=False):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Random seed value. Default is 42.
        deterministic (bool): If True, enables deterministic mode for CUDA operations.
                            This may impact performance but ensures full reproducibility.
                            Default is False.
    
    Note:
        - Setting deterministic=True may reduce performance on CUDA operations
        - Some operations may still be non-deterministic even with these settings
        - For full reproducibility, also consider:
          * Setting num_workers=0 in DataLoader
          * Disabling data augmentation randomness
          * Using the same hardware/CUDA version
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
