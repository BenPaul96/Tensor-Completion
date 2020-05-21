# Initialize global settings used across files

import numpy as np

# Use cupy if available, else numpy
cupy_imported = False
try:
    import cupy as cp
    xp = cp
    cupy_imported = True
except:
    xp = np

use_cupy = cupy_imported

def init(use_gpu):
    # Set the global GPU related variables.
    global cupy_imported
    global use_cupy
    global xp

    if use_gpu:
        if cupy_imported:
            xp = cp
            use_cupy = True
        else:
            xp = np
            use_cupy = False
            print("Cannot use gpu because cupy cannot be imported")
    else:
        xp = np
        use_cupy = False
