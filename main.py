#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Main Execution Script for Speech Emotion Recognition (SigWavNet)
Author: Alaa Nfissi
Date: March 31, 2024
Description: Trains and tests the SigWavNet dysarthria-severity model on the TORGO
dataset for a single trial with a fixed configuration (no hyperparameter search).
"""

import copy

from utils import *          # utility functions, classes, and globals (incl. `device`)
from model import *          # model definition
from custom_layers import *  # custom layer definitions


# Data loading runs in the main process (see utils.py for the rationale).
num_workers = 0
pin_memory = False

# Output folder for the trained checkpoint.
torgo_experiments_folder = "experiments"


# Fixed configuration for a single training trial.
def build_config(severityclasses):
    wt = pywt.Wavelet('db10')  # wavelet used to initialise the learnable kernels
    return {
        "n_input": 1,
        "hidden_dim": 32,
        "n_layers": 3,
        "n_output": len(severityclasses),
        "weight_decay": 1e-4,
        "level": 4,
        "trainHT": True,
        "initHT": 0.8,
        "kernTrainable": True,
        "kernelInit": np.array(wt.filter_bank[0]),
        "alpha": 10,
        "mode": "PerLayer",
        "lr": 1e-4,
        "batch_size": 4,
        "num_splits": 5,
    }


def build_model(config):
    """Instantiates a SigWavNet model from a configuration dictionary."""
    return SigWavNet(
        n_input=config['n_input'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_output=config['n_output'],
        inputSize=None,
        kernelInit=config['kernelInit'],
        kernTrainable=config['kernTrainable'],
        level=config['level'],
        kernelsConstraint=config['mode'],
        initHT=config['initHT'],
        trainHT=config['trainHT'],
        alpha=config['alpha'],
    )


def train_SigWavNet(config, data, max_num_epochs, device):
    """
    Trains the SigWavNet model for a single trial.

    A single stratified train/validation split (the first fold) is used to monitor
    training; the model state with the lowest validation loss is kept and returned.

    Parameters:
    - config: configuration dictionary for the model and training.
    - data: training DataFrame (path/label/source columns).
    - max_num_epochs: number of epochs to train.
    - device: torch device to train on.

    Returns:
    - The trained model loaded with its best (lowest validation loss) weights.
    """

    # Class weights for the focal loss (inverse class frequency).
    _, class_counts = np.unique(data['label'], return_counts=True)
    total_samples = len(data['label'])
    class_weights = [1.0 / (count / total_samples) for count in class_counts]
    criterion = FocalLoss(alpha=torch.FloatTensor(class_weights), gamma=2)

    model = build_model(config).to(device)

    optimiser = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)

    # Single stratified train/validation split.
    train_loader, val_loader = get_dataloaders(
        data, batch_size=config["batch_size"], num_splits=config["num_splits"])[0]

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, max_num_epochs + 1):

        # ---- Training ----
        model.train()
        right = 0
        h = model.init_hidden(config["batch_size"])
        for inputs, target in train_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            h = [i.data for i in h]  # detach hidden states

            output, h = model(inputs, h)
            right += nr_of_right(get_probable_idx(output), target)

            loss = criterion(output.squeeze(), target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        train_acc = 100. * right / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        right = 0
        val_loss = 0.0
        val_steps = 0
        h = model.init_hidden(config["batch_size"])
        with torch.no_grad():
            for inputs, target in val_loader:
                inputs = inputs.to(device)
                target = target.to(device)
                h = [i.data for i in h]

                output, h = model(inputs, h)
                right += nr_of_right(get_probable_idx(output), target)

                val_loss += criterion(output.squeeze(), target).item()
                val_steps += 1

        val_acc = 100. * right / len(val_loader.dataset)
        val_loss = val_loss / max(val_steps, 1)
        scheduler.step()

        print(f"Epoch {epoch:>3}/{max_num_epochs} | "
              f"train acc {train_acc:5.1f}% | "
              f"val loss {val_loss:.4f} | val acc {val_acc:5.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist the best model.
    os.makedirs(torgo_experiments_folder, exist_ok=True)
    checkpoint_path = os.path.join(torgo_experiments_folder, "SigWavNet_best.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Finished training. Best model saved to {checkpoint_path}")

    return model


def test(model, batch_size, data, device):
    """
    Tests the given model on a test dataset.

    Parameters:
    - model: the trained model to evaluate.
    - batch_size: batch size for testing.
    - data: the test DataFrame.
    - device: torch device to evaluate on.

    Returns:
    - Test-set accuracy, predicted labels, and true labels.
    """

    model.eval()
    right = 0

    # Normalise with statistics computed over the test waveforms.
    train_mean, train_std = compute_precise_mean_std(data['path'])
    test_transform = MyTransformPipeline(train_mean=train_mean, train_std=train_std)
    test_set = MyDataset(
        data['path'].reset_index(drop=True),
        data['label'].reset_index(drop=True),
        test_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    h = model.init_hidden(batch_size)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            y_true.extend(target.data.cpu().numpy())

            h = [i.data for i in h]
            output, h = model(inputs, h)

            pred = get_probable_idx(output)
            right += nr_of_right(pred, target)
            y_pred.extend(pred.squeeze().data.cpu().numpy())

    test_acc = 100. * right / len(test_loader.dataset)
    print(f"Test set accuracy: {right}/{len(test_loader.dataset)} ({test_acc:.1f}%)")

    return test_acc, y_pred, y_true


def main(max_num_epochs=5):
    """
    Sets up and runs a single training/testing trial of SigWavNet on TORGO.

    Parameters:
    - max_num_epochs: number of epochs to train.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (TORGO dysarthria severity).
    data, severityclasses = load_data(TORGO_ROOT)
    config = build_config(severityclasses)

    print(f"SigWavNet | severity classes: {severityclasses} | "
          f"n_input: {config['n_input']} | n_output: {config['n_output']}")

    # Speaker-aware 90/10 train/test split. With val_split=0 the split function
    # returns (train=90%, middle=10%, last=empty); the 10% middle is our test set.
    train_ds, test_ds, _ = get_dataset_partitions_pd(
        data, train_split=0.9, val_split=0.0, test_split=0.1,
        target_variable='label', data_source='source')

    model = train_SigWavNet(config, data=train_ds, max_num_epochs=max_num_epochs, device=device)

    test_acc, y_pred, y_true = test(model, config["batch_size"], test_ds, device)
    print(f"Best trial test set accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main(max_num_epochs=5)
