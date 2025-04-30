
import random
import torch
import math
from preprocess.mediapipe import RIGHT_ARM_IDX, LEFT_ARM_IDX



def __random_pass(prob: float) -> bool:
    return random.random() < prob

def __rotate_tensor(points: torch.Tensor, origin: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotate points (T, N, 2) around an origin (2,) by a given angle (in radians).
    """
    shifted = points - origin
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    rotation_matrix = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        device=points.device,
        dtype=points.dtype,
    )
    rotated = shifted @ rotation_matrix.T
    return rotated + origin


def augment_rotate(sign: torch.Tensor, angle_range: tuple) -> torch.Tensor:
    """
    Rotate entire skeleton around center (0.5, 0.5) by random angle from range.
    sign: (T, N, 2) tensor
    """
    angle = math.radians(random.uniform(*angle_range))
    center = torch.tensor([0.5, 0.5], device=sign.device)

    return __rotate_tensor(sign, center, angle)


def augment_shear(sign: torch.Tensor, type: str, squeeze_ratio: tuple) -> torch.Tensor:
    """
    Apply squeeze or perspective-like transform to the tensor landmarks.
    """
    T, N, _ = sign.shape

    src = torch.tensor([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=torch.float32, device=sign.device)

    if type == "squeeze":
        move_left = random.uniform(*squeeze_ratio)
        move_right = random.uniform(*squeeze_ratio)

        dest = torch.tensor([
            [0 + move_left, 1],
            [1 - move_right, 1],
            [0 + move_left, 0],
            [1 - move_right, 0]
        ], dtype=torch.float32, device=sign.device)

    elif type == "perspective":
        move_ratio = random.uniform(*squeeze_ratio)

        if __random_pass(0.5):
            dest = torch.tensor([
                [0 + move_ratio, 1 - move_ratio],
                [1, 1],
                [0 + move_ratio, 0 + move_ratio],
                [1, 0]
            ], dtype=torch.float32, device=sign.device)
        else:
            dest = torch.tensor([
                [0, 1],
                [1 - move_ratio, 1 - move_ratio],
                [0, 0],
                [1 - move_ratio, 0 + move_ratio]
            ], dtype=torch.float32, device=sign.device)
    else:
        raise ValueError(f"Unsupported shear type {type}")

    matrix = torch.linalg.lstsq(src, dest).solution.to(sign.dtype)
    matrix = matrix.T


    # Apply to all points
    sign_reshaped = sign.reshape(-1, 2)  # (T*N, 2)
    transformed = (sign_reshaped @ matrix.T).reshape(T, N, 2)

    return transformed


def augment_arm_joint_rotate(sign: torch.Tensor, probability: float, angle_range: tuple) -> torch.Tensor:
    """
    Slight rotation of joints in arms sequentially.
    sign: (T, N, 2) tensor
    """
    sign = sign.clone()

    for arm_indices in [LEFT_ARM_IDX, RIGHT_ARM_IDX]:
        for i in range(len(arm_indices) - 1):
            origin_idx = arm_indices[i]
            for j in range(i + 1, len(arm_indices)):
                target_idx = arm_indices[j]
                if __random_pass(probability):
                    angle = math.radians(random.uniform(*angle_range))
                    origin = sign[:, origin_idx, :]  # (T, 2)
                    target = sign[:, target_idx, :]  # (T, 2)
                    rotated = __rotate_tensor(target, origin, angle)
                    sign[:, target_idx, :] = rotated

    return sign
