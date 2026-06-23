import numpy as np

def generate_maze(D, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # starting with grid of all ones (walls)
    maze = np.ones((D, D), dtype=int)

    # start Primms at (0,0) - it shouldn't matter,maze should be possible for any starting point
    