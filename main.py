import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import argparse



from torchvision import transforms
from tqdm import tqdm

from dataset import test_train_split, get_meta_gloss_dataloader, SignGlossDataset
from model.model import SiFormerMeta, MetaLearning
from model.utils import train_epoch, evaluate
from model.gaussian_noise import GaussianNoise





if __name__ == "__main__":

        
    parser = argparse.ArgumentParser(description="Main training script")
    parser.add_argument("--num_epochs", type=int, default=350)
    parser.add_argument("--path", type=str, default="dataset/preprocessed/WLASL/raw/", help="Dataset path")
    parser.add_argument("--save_at", type=str, default="checkpoints/", help="Save best model at path")
    parser.add_argument("--use_checkpoint", type=str, default="", help="Save best model at path")
    parser.add_argument(
        "--output_episode_progress",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Set to false for Jupyter notebook to avoid excessive tqdm output"
    )
    parser.add_argument("--seed", type=int, default=42, help="Dataset path")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    num_epochs = args.num_epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator()
        
    transform = transforms.Compose([GaussianNoise(0, 0.001)])
    interp_transform = transforms.Compose([GaussianNoise(0, 0.005)])
    dataset = SignGlossDataset(dataset_dir=args.path, transform=transform, max_pad_len=300)

    train_dataset, test_dataset = test_train_split(dataset)
    train_loader = get_meta_gloss_dataloader(train_dataset, interpolation_transform=interp_transform)
    test_loader = get_meta_gloss_dataloader(test_dataset, interpolation_transform=interp_transform, num_episodes=20)



    backbone = SiFormerMeta(emb_dim=128, seq_len=300, max_class_per_input=3)
    model = MetaLearning(backbone).to(device)

    if args.use_checkpoint:
        state_dict = torch.load(args.use_checkpoint)
        model.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)  # 40, 60, 80

    curr_best = 0
    for epoch in (pbar:=tqdm(range(num_epochs), desc="Currently training...", position=0)):
        loss, train_accuracy = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_progress=args.output_episode_progress
        )
        val_accuracy = evaluate(model=model, dataloader=test_loader, device=device, output_progress=args.output_episode_progress)

        pbar.set_postfix({
            "Loss": f"{loss:.4f}",
            "Train Acc": f"{train_accuracy*100:.2f}%",
            "Val Acc": f"{val_accuracy*100:.2f}%"
        })

        if val_accuracy >= curr_best:
            curr_best = val_accuracy
            torch.save(model.state_dict(), os.path.join(args.save_at, f"{epoch}_{train_accuracy}"))