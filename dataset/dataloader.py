import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
from torch.utils.data import Subset, Dataset

import os
import numpy as np
import random
from collections import Counter, defaultdict
from tqdm import tqdm
from augmentations import process_depth_map, augment
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple



DATASET_PATH = 'dataset/preprocessed/WLASL/raw'



def load_dataset(dataset_dir: str = DATASET_PATH) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Loads the dataset from the specified directory.

    :param dataset_dir: Path to the directory containing the dataset
    :return: Tuple containing the data and labels
    """
    data = []
    class_to_indices = defaultdict(list)
    gloss_to_class = dict()
    labels = []
    curr_idx = 0

    for file in tqdm(os.listdir(dataset_dir), desc="Loading dataset", total=len(os.listdir(dataset_dir))):
        if file.endswith('.npy'):
            file_path = os.path.join(dataset_dir, file)
            loaded_data = np.load(file_path, allow_pickle=True).item()

            data.append(loaded_data['landmark'])
            gloss = loaded_data['label']

            if not gloss in gloss_to_class:
                gloss_to_class[gloss] = len(gloss_to_class)
            label = gloss_to_class[gloss]

            class_to_indices[label].append(curr_idx)
            curr_idx += 1
            
            labels.append(label)


    return data, labels, class_to_indices, gloss_to_class




class SignGlossDataset(torch_data.Dataset):
    """Advanced object representation of the WLASL dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: list[np.ndarray]
    labels: list[np.ndarray]

    def __init__(self, dataset_dir: str = DATASET_PATH, transform=None, augmentations=False, normalize=True, pad_to_max=False):
        """
        Initiates the WLASLDataset with the pre-loaded data from the numpy file.

        :param dataset_filename: Path to the numpy file
        :param transform: Any data transformation to be applied (default: None)
        """

        data, labels, class_to_indices, gloss_to_class  = load_dataset(dataset_dir=dataset_dir)

        self.data = data
        self.labels = labels
        self.class_to_indices = class_to_indices
        self.gloss_to_class = gloss_to_class
        self.classes = list(self.class_to_indices.keys())
        
        
        print(f"Number of classes: {len(self.classes)}")
        print(f"Total dataset length: {len(self.data)}")

        self.max_seq_len = max(self.data, key=lambda x: x.shape[0]).shape[0]

        self.targets = list(self.labels)
        self.transform = transform

        self.augmentations = augmentations
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


class EpisodicBatchSampler(torch_data.Sampler):
    """
    Samples batches in the form of episodes for few-shot learning tasks.
    """
    
    def __init__(self, class_to_indices: Dict, n_way: int, n_support: int, n_query: int, 
                 num_episodes: int, shuffle: bool = True):
        """
        Args:
            class_to_indices: Dictionary mapping each class to its sample indices
            n_way: Number of classes per episode
            n_support: Number of support examples per class
            n_query: Number of query examples per class
            num_episodes: Number of episodes per epoch
            shuffle: Whether to shuffle the classes when creating episodes
        """
        self.class_to_indices = class_to_indices
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.num_episodes = num_episodes
        self.shuffle = shuffle
        self.classes = list(class_to_indices.keys())
        
        # Verify we have enough classes and examples per class
        if len(self.classes) < self.n_way:
            raise ValueError(f"Dataset has only {len(self.classes)} classes, but n_way={self.n_way}")
        
        for class_idx in self.classes:
            class_indices = self.class_to_indices[class_idx]
            if len(class_indices) < (self.n_support + self.n_query):
                raise ValueError(f"Class {class_idx} has only {len(class_indices)} examples, "
                                f"but need {self.n_support + self.n_query}")
    
    def __len__(self) -> int:
        """Return the number of episodes per epoch."""
        return self.num_episodes
    
    def __iter__(self):
        """
        Yield indices for each episode.
        """
        for _ in range(self.num_episodes):
            episode_indices = []
            
            # Randomly sample classes for the episode
            episode_classes = random.sample(self.classes, self.n_way)
            
            # For each class, select support and query examples
            for class_label in episode_classes:
                indices = self.class_to_indices[class_label]
                # Sample without replacement
                sampled_indices = random.sample(indices, self.n_support + self.n_query)
                episode_indices.extend(sampled_indices)
            
            # Yield the indices for this episode
            yield episode_indices


# Define the collate function outside of any other function to make it picklable
def episodic_collate_fn(batch: List, n_way: int, n_support: int, n_query: int) -> Tuple:
    """
    Custom collate function for episodic few-shot learning.
    
    Args:
        batch: List of samples from the dataset
        n_way: Number of classes per episode
        n_support: Number of support examples per class
        n_query: Number of query examples per class
        
    Returns:
        Tuple containing:
            - Support set as tuple of (l_hand, r_hand, body) tensors
            - Support labels tensor
            - Query set as tuple of (l_hand, r_hand, body) tensors
            - Query labels tensor
    """
    # Total samples per class
    samples_per_class = n_support + n_query
    
    # Get the input tensors
    l_hand_tensors = [sample[0] for sample in batch]
    r_hand_tensors = [sample[1] for sample in batch]
    body_tensors = [sample[2] for sample in batch]
    labels = [sample[3] for sample in batch]
    
    # Create new labels using the order of appearance (0 to n_way-1)
    original_labels = torch.cat(labels).view(-1).tolist()
    unique_labels = []
    for label in original_labels:
        if label not in unique_labels:
            unique_labels.append(label)
    
    label_mapping = {original: i for i, original in enumerate(unique_labels[:n_way])}
    new_labels = [label_mapping[original.item()] for original in torch.cat(labels)]
    
    # Split into support and query sets
    support_indices = []
    query_indices = []
    
    for i in range(n_way):
        class_indices = [idx for idx, label in enumerate(new_labels) if label == i]
        support_indices.extend(class_indices[:n_support])
        query_indices.extend(class_indices[n_support:samples_per_class])
    
    # Create support set
    l_support = torch.stack([l_hand_tensors[i] for i in support_indices])
    r_support = torch.stack([r_hand_tensors[i] for i in support_indices])
    b_support = torch.stack([body_tensors[i] for i in support_indices])
    support_labels = torch.tensor([new_labels[i] for i in support_indices], dtype=torch.long)
    
    # Create query set
    l_query = torch.stack([l_hand_tensors[i] for i in query_indices])
    r_query = torch.stack([r_hand_tensors[i] for i in query_indices])
    b_query = torch.stack([body_tensors[i] for i in query_indices])
    query_labels = torch.tensor([new_labels[i] for i in query_indices], dtype=torch.long)
    
    return (l_support, r_support, b_support), support_labels, (l_query, r_query, b_query), query_labels


# Define a separate class for the collate function to make it fully picklable
class EpisodicCollator:
    def __init__(self, n_way, n_support, n_query):
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
    
    def __call__(self, batch):
        return episodic_collate_fn(batch, self.n_way, self.n_support, self.n_query)


def get_meta_gloss_dataloader(dataset, n_way=10, n_support=3, n_query=3, num_episodes=500, 
                           num_workers=4, pin_memory=True):
    """
    Create a PyTorch DataLoader with custom episodic sampling.
    
    Args:
        dataset: SignGlossDataset instance
        n_way: Number of classes per episode
        n_support: Number of support examples per class
        n_query: Number of query examples per class
        num_episodes: Number of episodes per epoch
        num_workers: Number of worker processes for parallel loading
        pin_memory: Whether to pin memory to GPU for faster transfer
        
    Returns:
        PyTorch DataLoader instance
    """
    # Create the batch sampler
    batch_sampler = EpisodicBatchSampler(
        class_to_indices=dataset.class_to_indices,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        num_episodes=num_episodes
    )
    
    # Create a picklable collate function using the EpisodicCollator class
    collate_fn = EpisodicCollator(n_way, n_support, n_query)
    
    # Create and return the DataLoader
    return torch_data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )







