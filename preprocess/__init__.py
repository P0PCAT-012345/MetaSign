import os
import numpy as np
import time
from tqdm import tqdm
from dataset import WLASLFeeder
from .mediapipe import MP_LandmarkExtractor, BODY_IDX, HANDS_IDX
from .visualize import render

DATASET_TYPE = 'WLASL' #['WLASL', ]
VIDEO_DATASET_DIR = 'dataset\WLASL'
ONLY_PREPROCESS_LANDMARK = True

def get_feeder(dataset_type: str = 'WLASL', root_dir: str = 'dataset\WLASL'):
    feeder = None
    if dataset_type == 'WLASL':
        feeder = WLASLFeeder(root_dir=root_dir)
    return feeder

def get_raw_landmark_dataset(feeder, dataset_name = 'WLASL', preprocess_path = 'dataset/preprocessed'):
    """
    Preprocess the dataset.

    Args:
        feeder: The dataset feeder.
    """
    

    save_at = os.path.join(preprocess_path, dataset_name, 'raw')
    os.makedirs(save_at, exist_ok=True)

    extractor = MP_LandmarkExtractor()
    for idx, (data, label) in tqdm(enumerate(feeder), desc="Extracting landmark from dataset", total=len(feeder)):
        # t1 = time.perf_counter()
        landmark = extractor.get_landmark_from_path(data)
        save_path = os.path.join(save_at, f"{idx}.npy")
        np.save(save_path, {"landmark": landmark, "label": label})
        # print(time.perf_counter() - t1)
        # break

    return save_at

def check_dataset(dataset_name = 'WLASL', preprocess_path = 'dataset/preprocessed'):
    """
    Preprocess the dataset.

    Args:
    """

    raw_at = os.path.join(preprocess_path, dataset_name, 'raw')
    files_to_remove = []
    for file in tqdm(os.listdir(raw_at), desc="Checking dataset", total=len(os.listdir(raw_at))):
        if file.endswith('.npy'):
            file_path = os.path.join(raw_at, file)
            loaded_data = np.load(file_path, allow_pickle=True).item()
            landmark = loaded_data['landmark']
            if np.all(landmark[:, HANDS_IDX] == 0):
                tqdm.write(f"WARNING: No hands were detected for {file}. Marked for removal.")
                files_to_remove.append(file_path)
            if np.all(landmark[:, BODY_IDX] == 0):
                tqdm.write(f"WARNING: No body was detected for {file} at some frame. Marked for removal.")
                files_to_remove.append(file_path)

    if files_to_remove:
        confirm = input(f"\n{len(files_to_remove)} files marked for removal. Do you want to delete them? (y/n): ").strip().lower()
        if confirm == 'y':
            for path in files_to_remove:
                os.remove(path)
            print(f"{len(files_to_remove)} files have been removed.")
        else:
            print("No files were removed.")
    


def main():
    feeder = get_feeder(DATASET_TYPE)
    if not ONLY_PREPROCESS_LANDMARK:
        get_raw_landmark_dataset(feeder, DATASET_TYPE)

    check_dataset()
    print("Preprocessing completed!")
    print(f"Total samples: {len(os.listdir(os.path.join('dataset/preprocessed', DATASET_TYPE, 'raw')))}")


if __name__ == "__main__":
    print("Preprocessing dataset...")
    main()