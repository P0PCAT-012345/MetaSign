from preprocess.mediapipe import RIGHT_HANDS_IDX, LEFT_HANDS_IDX, BODY_IDX
import torch
import random
from random import randrange
from augmentations.augmentations import augment_rotate, augment_shear, augment_arm_joint_rotate



def normalize_depth_map(input):
    """
    Normalize input with shape (T, N, 2) where T is the number of frames,
    such that for each frame, data points (N, 2) have their bounding box from 0 to 1.
    
    Args:
        input: PyTorch tensor of shape (T, N, 2)
            T: number of frames
            N: number of points per frame
            2: x and y coordinates
    
    Returns:
        normalized: PyTorch tensor of shape (T, N, 2)
            Data normalized so that each frame's points fit within a 0-1 bounding box
    """
    # Create a new tensor to store the result
    normalized = input.clone()
    
    # Get min and max for both x and y coordinates per frame
    # Shape: (T, 2)
    mins, _ = torch.min(input, dim=1)
    maxs, _ = torch.max(input, dim=1)
    
    # Calculate range for both x and y coordinates per frame
    # Shape: (T, 2)
    ranges = maxs - mins
    
    # Handle zero ranges to avoid division by zero
    # Replace zeros with ones (will result in setting coordinates to 0.5 later)
    mask = (ranges == 0)
    ranges[mask] = 1.0
    
    # Reshape mins and ranges for broadcasting
    # Shape: (T, 1, 2)
    mins = mins.unsqueeze(1)
    ranges = ranges.unsqueeze(1)
    
    # Normalize all coordinates at once using broadcasting
    normalized = (input - mins) / ranges
    
    # Set coordinates where range was zero to 0.5
    mask = mask.unsqueeze(1)  # Shape: (T, 1, 2)
    normalized[mask.expand_as(normalized)] = 0.5
    
    return normalized


def augment(depth_map):
    depth_map[:, :, :2] = augment_rotate(depth_map[:, :, :2], (-13, 13))

    depth_map[:, :, :2] = augment_shear(depth_map[:, :, :2], "perspective", (0, 0.1))

    depth_map[:, :, :2] = augment_shear(depth_map[:, :, :2], "squeeze", (0, 0.15))

    depth_map[:, :, :2] = augment_arm_joint_rotate(depth_map[:, :, :2], 0.3, (-4, 4))

    return depth_map


def process_depth_map(depth_map, transform=None, normalize=True, augmentations=False):
    """
    Processes the depth map by applying the specified transformations and normalizations.

    :param depth_map: The depth map to be processed
    :param transform: The transformation to be applied
    :param normalize: Whether to normalize the depth map (default: True)
    :param augmentat: Whether to apply augmentations (default: False)
    :return: The processed depth map
    """
    mask = (depth_map != 0).any(dim=-1).any(dim=-1)
    depth_map = depth_map[mask]
    
    l_hand_depth_mask = (depth_map[:, LEFT_HANDS_IDX] != 0).any(dim=-1).any(dim=-1)
    r_hand_depth_mask = (depth_map[:, RIGHT_HANDS_IDX] != 0).any(dim=-1).any(dim=-1)
    body_depth_mask = (depth_map[:, BODY_IDX] != 0).any(dim=-1).any(dim=-1)

    depth_masks = [l_hand_depth_mask, r_hand_depth_mask, body_depth_mask]
    
    depth_map = augment(depth_map) if augmentations else depth_map
            
    l_hand_depth_map = depth_map[:, LEFT_HANDS_IDX, :]
    r_hand_depth_map = depth_map[:, RIGHT_HANDS_IDX, :]
    body_depth_map = depth_map[:, BODY_IDX, :]


    if transform:
        l_hand_depth_map = transform(l_hand_depth_map)  
        r_hand_depth_map = transform(r_hand_depth_map) 
        body_depth_map = transform(body_depth_map) 

    if normalize:
        l_hand_depth_map = normalize_depth_map(l_hand_depth_map)
        r_hand_depth_map = normalize_depth_map(r_hand_depth_map)
        body_depth_map = normalize_depth_map(body_depth_map)
    
    depth_maps = [l_hand_depth_map, r_hand_depth_map, body_depth_map]
    depth_maps = [map - 0.5 for map in depth_maps]
    depth_maps = [map * mask.view(-1, 1, 1) for map, mask in zip(depth_maps, depth_masks)]

    return *depth_maps, depth_masks