import torch
from functools import lru_cache

@lru_cache()
def _generate_grid(H, W):
    """
    Internal function to generate a grid of coordinates and cache it.
    This function uses purely hashable types (integers) for caching.
    """
    return torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1)


def static_grid(H, W, device, dtype):
    """
    Generates a grid of coordinates for a given height (H) and width (W), with caching.

    Args:
        H (int): Height of the grid.
        W (int): Width of the grid.
        device (torch.device): The device on which to place the generated grid (e.g., 'cpu', 'cuda').
        dtype (torch.dtype): The desired data type of the grid (e.g., torch.float32, torch.int32).

    Returns:
        torch.Tensor: The generated grid of coordinates.
    """
    if not isinstance(H, int) or H <= 0:
        raise ValueError("H should be a positive integer.")
    if not isinstance(W, int) or W <= 0:
        raise ValueError("W should be a positive integer.")

    if not isinstance(device, torch.device):
        raise ValueError("device should be a torch.device instance.")

    if not isinstance(dtype, torch.dtype):
        raise ValueError("dtype should be a torch.dtype instance.")

    # Generate the grid using the cached internal function
    grid = _generate_grid(H, W)

    # Move the grid to the specified device and convert it to the desired dtype
    return grid.to(device=device, dtype=dtype)