import os
from tqdm import tqdm
import json
from typing import Any, Tuple
from collections import defaultdict


WSASL_DIR = 'dataset\WLASL'
VIDEO_DIR_NAME = 'videos'
JSON_NAME = "WLASL_V0.3.json"


def download_wslasl():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("sttaseen/wlasl2000-resized")

    print("Path to dataset files:", path)



class WLASLFeeder():
    """
    Feeder for the WLASL dataset.
    """

    def __init__(self, root_dir: str = WSASL_DIR, video_dir_name: str = VIDEO_DIR_NAME, json_name: str = JSON_NAME) -> None:
        """
        Initialize the WLASL feeder.

        Args:
            root_dir (str): Root directory of the WLASL dataset.
        """
        super().__init__()
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, VIDEO_DIR_NAME)
        self.json_path = os.path.join(root_dir, JSON_NAME)

        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        self.video_id_to_gloss = {}
        for entry in tqdm(json_data, desc="Reading WLASL data"):
            gloss = entry.get('gloss')
            for instance in entry.get('instances', []):
                vid = instance.get('video_id')
                if vid is not None:
                    self.video_id_to_gloss[vid] = gloss
        

        self.data = []
        self.label = []
        
        self.label_to_data = defaultdict(list)
        self.length = len(os.listdir(self.video_dir))

        for video_path in os.listdir(self.video_dir):
            if video_path.endswith('.mp4'):
                video_id = os.path.splitext(video_path)[0]
                gloss = self.video_id_to_gloss.get(video_id, None)
                if gloss is not None:
                    path = os.path.join(self.video_dir, video_path)
                    self.data.append(path)
                    self.label.append(gloss)
                    self.label_to_data[gloss].append(path)
                else:
                    print(f"Warning: Ignoring video ID {video_id} as it is not found in JSON data.")
        
        for label, paths in self.label_to_data.items():
            if len(paths) < 5:
                print(f"Warning: Label {label} has less than 5 samples. Please use a different dataset.")

        print(f"Total samples: {len(self)}")


    def __len__(self) -> int:
        """
        Returns:
            int: Total number of samples in the WLASL dataset.
        """
        # Implement logic to return the number of samples based on the JSON file or video directory
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        """
        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[Any, str]: A tuple containing the video data (e.g., frames, path, tensor) and the corresponding gloss (text label).
        """
        # Implement logic to retrieve video and gloss data based on index
        return self.data[idx], self.label[idx]



if __name__ == "__main__":

    feeder = WLASLFeeder()
    for video, gloss in feeder:
        print(f"Video: {video}, Gloss: {gloss}")
        break  # Remove this line to iterate through all samples