from typing import Optional

import torch.optim as optim
from torch import nn, save
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from underwater_imagery.models.metrics import (logits_to_idx_class,
                                               mean_pixel_acc)


def train_model(
    model: nn.Module,
    loss_module: nn.Module,
    ds_loader: DataLoader,
    num_epochs: int,
    file_name: Optional[str] = None,
) -> nn.Module:
    """Train given model.

    Args:
        model (nn.Module): Model, with device parameter implemented.
        loss_module (nn.Module): pytorch or custom loss function.
        ds_loader (DataLoader): dataloader
        num_epochs (int): number of epochs
        file_name (Optional[str], optional): Save model to this file. Defaults to None.

    Returns:
        nn.Module: model
    """
    optimizer = optim.Adam(model.parameters())
    pbar = tqdm(total=num_epochs * len(ds_loader))
    step = 0
    for epoch in range(0, num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(ds_loader):
            images, label_masks, label_images = data
            images = images.to(model.device)
            label_masks = label_masks.to(model.device)
            label_images = label_images.to(model.device)

            # forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_module(logits, label_masks)

            # optimize
            loss.backward()
            optimizer.step()

            # calculate pixel accuracy
            pred = logits_to_idx_class(logits)
            true = logits_to_idx_class(label_masks)
            running_acc += mean_pixel_acc(pred, true)

            # running loss
            running_loss += loss.item()
            step += 1
            print_every = 10
            if (i % print_every) == (print_every - 1):
                desc = f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_every:.3f} acc: {running_acc / print_every:.3f}"
                _ = pbar.update(print_every)
                _ = pbar.set_description(desc)
                running_loss = 0.0
                running_acc = 0.0
    pbar.close()
    if file_name:
        save(model, file_name + ".pth")
    return model
