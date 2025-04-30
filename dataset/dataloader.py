import torch
import torch.utils.data as torch_data
import os
import numpy as np
import random
from collections import Counter
from tqdm import tqdm
from augmentations import process_depth_map, augment



DATASET_PATH = 'dataset/preprocessed/WLASL/raw'



def load_dataset(dataset_dir: str = DATASET_PATH) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Loads the dataset from the specified directory.

    :param dataset_dir: Path to the directory containing the dataset
    :return: Tuple containing the data and labels
    """
    data = []
    gloss_to_label = dict()
    labels = []

    for file in tqdm(os.listdir(dataset_dir), desc="Loading dataset", total=len(os.listdir(dataset_dir))):
        if file.endswith('.npy'):
            file_path = os.path.join(dataset_dir, file)
            loaded_data = np.load(file_path, allow_pickle=True).item()

            data.append(loaded_data['landmark'])
            gloss = loaded_data['label']
            
            if gloss not in gloss_to_label.keys():
                gloss_to_label[gloss] = len(gloss_to_label)
            labels.append(gloss_to_label[gloss])


    return data, labels, gloss_to_label




class MetaSignGlossDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: list[np.ndarray]
    labels: list[np.ndarray]

    def __init__(self, dataset_dir: str = DATASET_PATH, num_classes: int =None, transform=None, augmentations=False,
                 augmentations_prob=0.5, normalize=True, pad_to_max=False):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        data, labels, gloss_to_label  = load_dataset(dataset_dir=dataset_dir)

        self.data = data
        self.labels = labels
        self.gloss_to_label = gloss_to_label
        self.gloss = list(gloss_to_label.keys())
        self.label_to_gloss = {v: k for k, v in gloss_to_label.items()}

        if num_classes is not None:
            class_counts = Counter(labels)
            top_classes = set([cls for cls, _ in class_counts.most_common(num_classes)])
            
            filtered_data = []
            filtered_labels = []
            
            for d, l in zip(data, labels):
                if l in top_classes:
                    filtered_data.append(d)
                    filtered_labels.append(l)

            self.data = filtered_data
            self.labels = filtered_labels
            self.num_classes = num_classes
        else:
            self.data = data
            self.labels = labels
            self.num_classes = len(self.gloss)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Total dataset length: {len(self.data)}")

        self.max_seq_len = max(self.data, key=lambda x: x.shape[0]).shape[0]

        self.targets = list(self.labels)
        self.transform = transform

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize
        self.pad_to_max = pad_to_max

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        l_hand_depth_map, r_hand_depth_map, body_depth_map, _ = process_depth_map(depth_map, self.transform, self.normalize, self.augmentations, self.augmentations_prob)
        label = torch.Tensor([self.labels[idx]])

        if self.pad_to_max:
            l_hand_depth_map, r_hand_depth_map, body_depth_map = map(self.pad_depth_map, (l_hand_depth_map, r_hand_depth_map, body_depth_map))

        return l_hand_depth_map, r_hand_depth_map, body_depth_map, label

    def __len__(self):
        return len(self.data)

    def pad_depth_map(self, depth_map):
        pad_size = self.max_seq_len - depth_map.shape[0]
        padding = torch.zeros(pad_size, *depth_map.shape[1:], dtype=depth_map.dtype)
        depth_map = torch.cat([depth_map, padding], dim=0)
        return depth_map






class MetaSignSentenceDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: list[np.ndarray]
    labels: list[np.ndarray]

    def __init__(self, dataset_dir: str = DATASET_PATH, num_episodes = 1000, total_frames=100, interpolation_step=(5,15), 
                 transform=None, augmentations=False, augmentations_prob=0.5, normalize=True):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        data, labels, gloss_to_label  = load_dataset(dataset_dir=dataset_dir)

        self.data = data
        self.labels = labels
        self.gloss_to_label = gloss_to_label
        self.gloss = list(gloss_to_label.keys())
        self.label_to_gloss = {v: k for k, v in gloss_to_label.items()}
        
        self.num_episodes = num_episodes
        self.total_frames = total_frames
        self.interpolation_step = interpolation_step
        self.transform = transform

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def get_gloss(self, idx):
        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        l_hand_depth_map, r_hand_depth_map, body_depth_map, mask = process_depth_map(depth_map, self.transform, self.normalize, self.augmentations, self.augmentations_prob)
        label = torch.Tensor([self.labels[idx]])

        return l_hand_depth_map, r_hand_depth_map, body_depth_map, label, mask
    
    def __getitem__(self, _):
        l_hand_sequence, r_hand_sequence, body_sequence = [], [], []
        sentence_labels = []
        
        total_frames = 0

        l_hand, r_hand, body, label, mask = self.get_gloss(random.randint(0, len(self.labels) - 1))
        cut_off = random.randint(0, len(l_hand))

        l_hand_sequence.append(l_hand[cut_off:])
        r_hand_sequence.append(r_hand[cut_off:])
        body_sequence.append(body[cut_off:])
        sentence_labels.append(label)
        total_frames += len(l_hand) - cut_off

        last_visibilities = [m[-1] for m in mask]

        while total_frames < self.total_frames:
            next_idx = random.randint(0, len(self.labels) - 1)
            l_next, r_next, body_next, label_next, mask_next = self.get_gloss(next_idx)
            first_visibilities = [m[0] for m in mask_next]

            interp_steps = random.randint(self.interpolation_step[0], self.interpolation_step[1])
    
            l_hand_sequence.append(self.slerp(l_hand[-1], l_next[0], interp_steps, transform=self.transform, augmentations=self.augmentations, augmentations_prob=self.augmentations_prob) if last_visibilities[0] and first_visibilities[0] else torch.zeros((interp_steps, l_hand.shape[1], l_hand.shape[2])))
            r_hand_sequence.append(self.slerp(r_hand[-1], r_next[0], interp_steps, transform=self.transform, augmentations=self.augmentations, augmentations_prob=self.augmentations_prob) if last_visibilities[1] and first_visibilities[1] else torch.zeros((interp_steps, r_hand.shape[1], r_hand.shape[2])))
            body_sequence.append(self.slerp(body[-1], body[0], interp_steps, transform=self.transform, augmentations=self.augmentations, augmentations_prob=self.augmentations_prob) if last_visibilities[2] and first_visibilities[2] else torch.zeros((interp_steps, body.shape[1], body.shape[2])))
            
            total_frames += interp_steps

            l_hand_sequence.append(l_next)
            r_hand_sequence.append(r_next)
            body_sequence.append(body_next)

            sentence_labels.append(label_next)

            total_frames += len(l_next)

            last_visibilities = [m[-1] for m in mask_next]


        l_hand_sequence = torch.cat(l_hand_sequence, dim=0)[:self.total_frames]
        r_hand_sequence = torch.cat(r_hand_sequence, dim=0)[:self.total_frames]
        body_sequence = torch.cat(body_sequence, dim=0)[:self.total_frames]

        sentence_labels = torch.tensor(sentence_labels)

        return l_hand_sequence, r_hand_sequence, body_sequence, sentence_labels


    def slerp(self, v0, v1, steps, transform=None, augmentations=True, augmentations_prob=0.5):
        """
        Spherical linear interpolation between two points.

        :param start: Start point
        :param end: End point
        :param steps: Number of steps
        :return: Interpolated points
        """
        t = torch.linspace(0, 1, steps + 2, device=v0.device)[1:-1]
        v0_norm = v0 / v0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v1_norm = v1 / v1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        dot = (v0_norm * v1_norm).sum(dim=-1).clamp(-1.0, 1.0)  # (N,)
        omega = torch.acos(dot)  # (N,)
        sin_omega = torch.sin(omega).clamp(min=1e-8)  # (N,)

            
        t = t.to(v0.device).view(-1, 1, 1)  # (steps, 1, 1)

        omega = omega.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        sin_omega = sin_omega.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

        v0 = v0.unsqueeze(0)  # (1, N, 2)
        v1 = v1.unsqueeze(0)  # (1, N, 2)

        term1 = torch.sin((1.0 - t) * omega) / sin_omega * v0
        term2 = torch.sin(t * omega) / sin_omega * v1

        result = term1 + term2
        result = augment(result, augmentations_prob) if augmentations else result
        result = transform(result) if transform else result
        
        return result
    
    def __len__(self):
        return self.num_episodes
