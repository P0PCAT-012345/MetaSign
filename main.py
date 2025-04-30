import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from torchvision import transforms
from tqdm import tqdm

from dataset import MetaSignGlossDataset
from model.model import SiFormer
from model.utils import train_epoch, evaluate
from model.gaussian_noise import GaussianNoise



def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )

    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)

    return train_dataset, val_dataset



def get_dataloader(train_set, generator):
    train_set, val_set = __balance_val_split(train_set, 0.2)

    val_set.transform = None
    val_set.augmentations = False
    val_loader = DataLoader(val_set, batch_size=24, shuffle=True, generator=generator,
                            num_workers=12)
    train_loader = DataLoader(train_set, batch_size=24, shuffle=True, generator=generator,
                            num_workers=12)

    return train_loader, val_loader




if __name__ == "__main__":
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator()
    transform = transforms.Compose([GaussianNoise(0, 0.001)])


    dataset = MetaSignGlossDataset(transform=transform, augmentations=True, pad_to_max = True, num_classes=100)
    num_classes = len(dataset)
    seq_len = dataset.max_seq_len

    train_loader, val_loader = get_dataloader(dataset, generator)



    model = SiFormer(num_classes=num_classes, seq_len=seq_len)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)  # 40, 60, 80

    curr_best = 0
    for epoch in (pbar:=tqdm(range(num_epochs), desc="Currently training...", position=0)):
        loss, train_accuracy = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        val_accuracy = evaluate(model=model, dataloader=val_loader, device=device)

        pbar.set_postfix({
            "Loss": f"{loss:.4f}",
            "Train Acc": f"{train_accuracy*100:.2f}%",
            "Val Acc": f"{val_accuracy*100:.2f}%"
        })

        if val_accuracy >= curr_best:
            curr_best = val_accuracy
            torch.save(model.state_dict(), f"checkpoints/{epoch}_{val_accuracy}")