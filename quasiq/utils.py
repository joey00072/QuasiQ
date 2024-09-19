import os
import numpy as np

# Constants
DEBUG = os.environ.get("DEBUG") == "1"
dtype = np.complex128  # double precision is needed

def debug_print(*args, **kwargs):
    """
    Print debug information if DEBUG is set to True.
    """
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def tensor_product(operators: list[np.ndarray]) -> np.ndarray:
    """
    compute the tensor product of a list of operators.
    
    the tensor product is defined as:
    R = A ⊗ B ⊗ C ⊗ ... = (A ⊗ B) ⊗ C ⊗ ...
    """
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result