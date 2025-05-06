import torch

import torch
import math

def slerp_depth_maps(v0, v1, frames):
    """
    Spherical linear interpolation between two depth maps over a number of frames,
    excluding the start and end tensors.

    Args:
        v0: Starting depth map with shape (V, C) (torch.Tensor)
        v1: Ending depth map with shape (V, C) (torch.Tensor)
        frames: Number of intermediate interpolation steps (excluding endpoints)

    Returns:
        Tensor of interpolated tensors with shape (frames, V, C)
    """
    # Normalize along channel dimension
    v0_norm = v0 / v0.norm(dim=1, keepdim=True)
    v1_norm = v1 / v1.norm(dim=1, keepdim=True)

    # Compute dot product along channels
    dot = (v0_norm * v1_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)

    # Handle small angles with linear interpolation fallback
    sin_omega = torch.sin(omega)
    use_linear = sin_omega.abs() < 1e-6

    # Generate interpolation steps
    ts = torch.linspace(0, 1, frames + 2, device=v0.device)[1:-1]
    interpolated = []

    for t in ts:
        t = t.item()
        factor0 = torch.sin((1 - t) * omega) / sin_omega
        factor1 = torch.sin(t * omega) / sin_omega

        # Where sin_omega is small, fall back to linear interpolation
        factor0 = torch.where(use_linear, 1 - t, factor0)
        factor1 = torch.where(use_linear, t, factor1)

        interp = factor0 * v0 + factor1 * v1
        interpolated.append(interp)

    return torch.stack(interpolated, dim=0)

def fill_masked_slerp(tensor, augment=None):
        """
        Fill in masked (zero) time frames in a (T, V, C) tensor using SLERP.
        
        Parameters:
            tensor (torch.Tensor): A tensor of shape (T, V, C) with some frames masked as zeros.
            
        Returns:
            torch.Tensor: The filled tensor of shape (T, V, C).
        """
        T, _, _ = tensor.shape
        mask = torch.any(tensor != 0, dim=(1, 2))  # shape: (T,), True if frame is unmasked

        result = tensor.clone()
        unmasked_indices = torch.where(mask)[0]

        if len(unmasked_indices) == 0:
            return result  # Nothing to interpolate, all are masked
        
        # Fill masked frames before the first unmasked frame
        first = unmasked_indices[0].item()
        if first > 0:
            first_frames = result[first].unsqueeze(0).repeat(first, 1, 1)
            result[:first] = augment(first_frames) if augment else first_frames

        # Fill masked frames after the last unmasked frame
        last = unmasked_indices[-1].item()
        if last < T - 1:
            last_frames = result[last].unsqueeze(0).repeat(T-last-1, 1, 1)
            result[last+1:] = augment(last_frames) if augment else last_frames

        # Interpolate between unmasked frames
        for i in range(len(unmasked_indices) - 1):
            start_idx = unmasked_indices[i].item()
            end_idx = unmasked_indices[i + 1].item()
            gap = end_idx - start_idx - 1
            if gap > 0:
                interpolated = slerp_depth_maps(result[start_idx], result[end_idx], gap)
                interpolated = augment(interpolated) if augment else interpolated
                result[start_idx + 1:end_idx] = interpolated

        return result




def remove_common_padding(l_depth_maps, r_depth_maps, body_depth_maps):
    N, T_max, _, _ = l_depth_maps.shape
    unpadded_l_depth_maps = []
    unpadded_r_depth_maps = []
    unpadded_body_depth_maps = []

    for i in range(N):
        # Get slices for each sample
        l_sample = l_depth_maps[i]           # shape: (T_max, V, C)
        r_sample = r_depth_maps[i]
        body_sample = body_depth_maps[i]     # shape: (T_max, V', C)

        # Check for padding across all time frames
        combined = torch.stack([
            torch.all(l_sample == 0, dim=(1, 2)),  # (T_max,)
            torch.all(r_sample == 0, dim=(1, 2)),
            torch.all(body_sample == 0, dim=(1, 2))
        ], dim=0)  # shape: (3, T_max)

        # A frame is padding only if all three inputs are zero at that frame
        padding_mask = torch.all(combined, dim=0)
        # Find last non-padding index
        valid_frames = torch.where(~padding_mask)[0]
        if valid_frames.numel() == 0:
            end = 0
        else:
            end = valid_frames[-1].item() + 1

        # Slice out the valid part
        unpadded_l_depth_maps.append(l_sample[:end])
        unpadded_r_depth_maps.append(r_sample[:end])
        unpadded_body_depth_maps.append(body_sample[:end])

    return unpadded_l_depth_maps, unpadded_r_depth_maps, unpadded_body_depth_maps


def truncate_and_slide(tensor, window_size, stride):
        """
        Applies a sliding window along the first dimension of a tensor (e.g., time axis).
        
        Args:
            tensor (torch.Tensor): Input tensor of shape (T, ...) where T >= window_size
            window_size (int): Length of each sliding window along the first dimension
            stride (int): Stride between windows
            
        Returns:
            torch.Tensor: Tensor of shape (num_windows, window_size, ...) where
                        num_windows = floor((T - window_size) / stride + 1)
        """
        T = tensor.size(0)
        if T < window_size:
            raise ValueError("Tensor length along the first dimension must be >= window_size")

        # Compute number of windows
        num_windows = (T - window_size) // stride + 1

        # Use unfold via advanced indexing for arbitrary-shaped tensors
        slices = []
        for i in range(0, stride * num_windows, stride):
            window = tensor[i : i + window_size]
            slices.append(window)

        return torch.stack(slices, dim=0)