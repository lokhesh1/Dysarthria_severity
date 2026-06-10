

"""
Title: Utility Functions for Speech Emotion Recognition (SigWavNet)
Description: This file contains utility functions for preprocessing, loading data, and 
performing various auxiliary tasks for the speech emotion recognition project.
"""

import os
import re
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import soundfile as sf


def load_audio(path):
    """
    Loads an audio file as a float32 waveform tensor.

    torchaudio (>=2.9) delegates ``load`` to TorchCodec/FFmpeg; to stay
    dependency-light and Python 3.13-friendly we read with libsndfile via
    soundfile and return a tensor shaped like torchaudio's ``load`` output.

    Parameters:
    - path (str): Path to the audio file.

    Returns:
    - Tuple of (waveform tensor of shape (channels, frames), sample_rate).
    """
    data, sample_rate = sf.read(path, dtype='float32', always_2d=True)  # (frames, channels)
    waveform = torch.from_numpy(data.T).contiguous()                    # (channels, frames)
    return waveform, sample_rate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
import math
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
torch.manual_seed(123)
import random
import pywt
random.seed(123)

from sklearn.model_selection import StratifiedKFold, KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# TORGO dysarthria severity configuration
# -----------------------------------------------------------------------------
# Root directory of the TORGO dataset. Expected to contain the gender/condition
# folders: F_Con, F_Dys, M_Con, M_Dys. Only the dysarthric folders (F_Dys, M_Dys)
# are used for severity classification. Override with the TORGO_ROOT env var.
TORGO_ROOT = '/home/hp/Desktop/TORGO'

# Only the dysarthric speaker folders are used.
DYSARTHRIC_FOLDERS = ['F', 'M']

# Per-speaker severity labels (perceptual severity of dysarthria).
SEVERITY_MAP = {
    'F03': 'very_low', 'F04': 'very_low', 'M03': 'very_low',
    'F01': 'low',      'M05': 'low',
    'M01': 'medium',   'M02': 'medium',  'M04': 'medium',
}

# Canonical, alphabetically-sorted class list (matches sorted(unique(labels))).
# Used as a fallback so the module can be imported before the dataset exists.
SEVERITY_CLASSES = sorted(set(SEVERITY_MAP.values()))


def build_torgo_dataframe(root_dir):
    """
    Builds a DataFrame describing the usable TORGO recordings for dysarthria
    severity classification.

    Only the dysarthric speaker folders (F, M) are scanned, and within
    them only subject folders (F01, M01, ...) containing Session{n} subfolders
    are used. Within each session, only a ``wav_arrayMic`` folder is considered;
    head-microphone folders and unrelated files are ignored.

    Parameters:
    - root_dir (str): Path to the TORGO dataset root.

    Returns:
    - DataFrame with columns:
        * ``path``   : absolute path to a .wav file
        * ``label``  : severity class (very_low / low / medium)
        * ``source`` : speaker id (used for stratified, speaker-aware splitting)
        * ``session``: session id (e.g. ``F01S01``)
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"TORGO root directory not found: {root_dir}")

    # Matches e.g. "Session1", "Session2", ... "Session9"
    session_re = re.compile(r'^Session(\d+)$', re.IGNORECASE)

    records = []
    for cond_folder in DYSARTHRIC_FOLDERS:          # "F", "M"
        cond_path = os.path.join(root_dir, cond_folder)
        if not os.path.isdir(cond_path):
            continue

        for speaker_folder in sorted(os.listdir(cond_path)):   # F01, F02, ...
            speaker = speaker_folder.upper()
            severity = SEVERITY_MAP.get(speaker)
            if severity is None:
                continue

            speaker_path = os.path.join(cond_path, speaker_folder)
            if not os.path.isdir(speaker_path):
                continue

            for session_folder in sorted(os.listdir(speaker_path)):   # Session1, Session2, ...
                match = session_re.match(session_folder)
                if match is None:
                    continue

                session_num = int(match.group(1))
                session_id = f"{speaker}S{session_num:02d}"   # e.g. F01S01

                wav_dir = os.path.join(speaker_path, session_folder, 'wav_arrayMic')
                if not os.path.isdir(wav_dir):
                    continue

                wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
                for wav_path in sorted(wav_files):
                    records.append({
                        'path': os.path.abspath(wav_path),
                        'label': severity,
                        'source': speaker,
                        'session': session_id,
                    })

    data = pd.DataFrame(records, columns=['path', 'label', 'source', 'session'])
    if data.empty:
        raise ValueError(
            f"No usable array-microphone dysarthric recordings found under {root_dir}. "
            f"Expected structure: {DYSARTHRIC_FOLDERS} → speaker → Session{{n}} → wav_arrayMic → *.wav"
        )
    return data


def load_data(data_path=None):
    """
    Loads the TORGO dysarthria-severity dataset from a directory tree.

    Parameters:
    - data_path (str, optional): Path to the TORGO dataset root. Defaults to
      ``TORGO_ROOT``. For backward compatibility, if a path to an existing
      ``.csv`` file is given it is read directly (it must contain ``path`` and
      ``label`` columns, and ideally a ``source`` speaker column).

    Returns:
    - Tuple of (DataFrame, list of severity classes sorted alphabetically).
    """
    if data_path is None:
        data_path = TORGO_ROOT

    if isinstance(data_path, str) and data_path.lower().endswith('.csv'):
        data = pd.read_csv(os.path.abspath(data_path))
        data = data[data['label'].isin(SEVERITY_MAP.values())].reset_index(drop=True)
    else:
        data = build_torgo_dataframe(data_path)

    severityclasses = sorted(list(data['label'].unique()))

    return data, severityclasses


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None, data_source=None):
    
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - df (DataFrame): The dataset to split.
    - train_split (float): Proportion of the dataset to use for training.
    - val_split (float): Proportion of the dataset to use for validation.
    - test_split (float): Proportion of the dataset to use for testing.
    - target_variable (str, optional): Name of the column containing the target variable for stratification.
    - data_source (str, optional): Name of the column containing the data source for stratification.

    Returns:
    - DataFrames for the training, validation, and test sets.
    """
    
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    #assert val_split == test_split 
    # Shuffle
    df_sample = df.sample(frac=1, random_state=42)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    # Helper: split a frame into [train, val, test] chunks at two cut points.
    # Uses positional .iloc slicing (np.split no longer preserves DataFrames
    # under numpy 2.x / pandas 3.x).
    def _split_three(frame, first, second):
        return [frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]]

    arr_list = []
    if target_variable is not None and data_source is not None:
        grouped_df = df_sample.groupby([data_source, target_variable])
        for i, g in grouped_df:
            if len(g) == 3:
                arr_list.append(_split_three(g, 1, 2))
            else:
                arr_list.append(_split_three(g, int(train_split * len(g)), int((1 - val_split) * len(g))))
        train_ds = pd.concat([t[0] for t in arr_list])
        val_ds = pd.concat([v[1] for v in arr_list])
        test_ds = pd.concat([t[2] for t in arr_list])

    else:
        first = int(train_split * len(df_sample))
        second = int((1 - val_split) * len(df_sample))
        train_ds, val_ds, test_ds = _split_three(df_sample, first, second)
    
    return train_ds.reset_index(drop=True), val_ds.reset_index(drop=True), test_ds.reset_index(drop=True)



class MyDataset(torch.utils.data.Dataset):
    
    """
    Custom PyTorch Dataset for loading and processing the speech emotion recognition dataset.

    Attributes:
    - paths (list): List of file paths to the audio files.
    - labels (list): List of labels corresponding to the audio files.
    - transform (callable): A function/transform that takes in an audio file and returns a transformed version.
    """
    
    def __init__(self, paths, labels, transform):
        self.files = paths
        self.labels = labels
        self.transform = transform
    def __getitem__(self, item):
        #print(self.files)
        file = self.files[item]
        label = self.labels[item]
        file, sampling_rate = load_audio(file)
        file = file if file.shape[0] == 1 else file[0].unsqueeze(0)
        file = self.transform(file)
        
        return file, sampling_rate, label
    
    def __len__(self):
        return len(self.files)
    
def compute_precise_mean_std(file_paths):
    
    """
    Computes the mean and standard deviation of the waveforms in the dataset.

    Parameters:
    - file_paths (list): List of paths to the audio files in the dataset.

    Returns:
    - Tuple containing the global mean and standard deviation of the waveforms.
    """
    
    sum_waveform = 0.0
    sum_squares = 0.0
    total_samples = 0
    
    for file_path in file_paths:
        waveform, _ = load_audio(file_path)
        sum_waveform += waveform.sum()
        sum_squares += (waveform ** 2).sum()
        total_samples += waveform.numel()  # Count total number of samples across all files
    
    # Compute global mean and std
    mean = sum_waveform / total_samples
    std = (sum_squares / total_samples - mean ** 2) ** 0.5
    
    return mean.item(), std.item()

class MyTransformPipeline(nn.Module):
    
    """
    Custom transform pipeline for processing audio data.
    
    Parameters:
    - train_mean (float): The mean of the training data.
    - train_std (float): The standard deviation of the training data.
    - input_freq (int): The original frequency of the audio data.
    - resample_freq (int): The target frequency to resample the audio data.
    """
    
    def __init__(
        self,
        train_mean = 0,
        train_std = 1,
        input_freq=16000,
        resample_freq=16000,
    ):
        super().__init__()
        
        self.train_mean = train_mean
        self.train_std = train_std
        self.input_freq = input_freq
        self.resample_freq = resample_freq
        
        self.resample = torchaudio.transforms.Resample(orig_freq=self.input_freq, new_freq=self.resample_freq).to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        waveform = waveform.to(device)
        resampled = self.resample(waveform)
        normalized_waveform = (resampled - self.train_mean) / self.train_std

        return normalized_waveform
    

def count_parameters(model):
    
    """
    Counts the number of trainable parameters in a model.
    
    Parameters:
    - model (torch.nn.Module): The model to count parameters for.
    
    Returns:
    - The number of trainable parameters.
    """
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# The transform pipeline moves tensors onto the (CUDA) device inside __getitem__,
# so data loading must stay in the main process: forked workers cannot reinit CUDA
# and CUDA tensors cannot be page-locked. Keep workers at 0 and pinning off.
num_workers = 0
pin_memory = False


def get_dataloaders(data, batch_size=32, num_splits=5, stratify=True):
    
    """
    Creates dataloaders for cross-validation, with optional stratification.

    Parameters:
    - data (DataFrame): The dataset to be loaded into the dataloaders.
    - batch_size (int): Size of batches.
    - num_splits (int): Number of folds for cross-validation.
    - stratify (bool): Whether to stratify the folds based on labels.

    Returns:
    - List of tuples containing train and validation dataloaders for each fold.
    """

    if stratify:
        kf = StratifiedKFold(n_splits=num_splits)
        split_method = kf.split(data, data['label'])
    else:
        kf = KFold(n_splits=num_splits)
        split_method = kf.split(data)
    
    dataloaders = []

    for train_idx, val_idx in split_method:
        train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
        
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        
        
        train_mean, train_std = compute_precise_mean_std(train_data['path'])
        
        transform = MyTransformPipeline(input_freq=16000, resample_freq=8000, train_mean = train_mean, train_std = train_std)
        
        transform.to(device)
        
        train_dataset = MyDataset(train_data['path'], train_data['label'], transform=transform)
        val_dataset = MyDataset(val_data['path'], val_data['label'], transform=transform)
        
        # Create dataloaders for train and validation sets
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        dataloaders.append((train_loader, val_loader))
    
    return dataloaders

# Populate the severity class list at import time. Fall back to the canonical
# class list if the dataset has not been downloaded yet, so the module can be
# imported (and unit-tested) without the data being present.
try:
    _, severityclasses = load_data(TORGO_ROOT)
except (FileNotFoundError, ValueError):
    severityclasses = list(SEVERITY_CLASSES)


def index_to_severity(index):
    """
    Converts an index to a dysarthria severity class.

    Parameters:
    - index (int): Index of the severity class in the list of classes.

    Returns:
    - The name of the severity class corresponding to the given index.
    """

    return severityclasses[index]

def severity_to_index(severity):

    """
    Converts a dysarthria severity class to its corresponding index.

    Parameters:
    - severity (str): The severity class.

    Returns:
    - Index of the severity class in the list of classes.
    """

    return torch.tensor(severityclasses.index(severity))

def pad_sequence(batch):
    
    """
    Pads a batch of tensors to the same length with zeros.

    Parameters:
    - batch (list of Tensor): The batch of tensors to pad.

    Returns:
    - A tensor containing the padded batch.
    """
    
    batch = [item.t() for item in batch]
    
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    
    """
    Custom collate function to process batches of data.

    Parameters:
    - batch (list): A batch of data.

    Returns:
    - Processed batch of tensors and targets.
    """
    
    tensors, targets = [], []

    # Gather in lists, and encode severity classes as indices
    for waveform, _, severity, *_ in batch:
        tensors += [waveform]
        targets += [severity_to_index(severity)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    # stack - Concatenates a sequence of tensors along a new dimension
    targets = torch.stack(targets)

    return tensors, targets


class FocalLoss(nn.Module):
    
    """
    Implementation of the Focal Loss as a PyTorch module.

    Parameters:
    - alpha (Tensor): Weighting factor for the positive class.
    - gamma (float): Focusing parameter to adjust the rate at which easy examples contribute to the loss.
    """
    
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    
    
def nr_of_right(pred, target):
    
    """
    Counts the number of correct predictions.

    Parameters:
    - pred (Tensor): Predicted labels.
    - target (Tensor): True labels.

    Returns:
    - Number of correct predictions.
    """
    
    return pred.squeeze().eq(target).sum().item()

def get_probable_idx(tensor):
    
    """
    Finds the indices of the most probable class for each element in the batch.

    Parameters:
    - tensor (Tensor): Tensor containing class probabilities for each element.

    Returns:
    - Tensor of indices for the most probable class.
    """
    
    return tensor.argmax(dim=-1)


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, normalize=True):
    
    """
    Prints a confusion matrix using seaborn.

    Parameters:
    - confusion_matrix (ndarray): The confusion matrix to print.
    - class_names (list): List of class names corresponding to the indices of the confusion matrix.
    - figsize (tuple): Size of the figure.
    - fontsize (int): Font size for the labels.
    - normalize (bool): Whether to normalize the values in the confusion matrix.
    """
    
    fig = plt.figure(figsize=figsize)
    if normalize:
        confusion_matrix_1 = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        confusion_matrix_1 = confusion_matrix
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix_1, index=class_names, columns=class_names
    )
    labels = (np.asarray(["{:1.2f} % \n ({})".format(value, value_1) for value, value_1 in zip(confusion_matrix_1.flatten(),confusion_matrix.flatten())])).reshape(confusion_matrix.shape)
    try:
        heatmap = sn.heatmap(df_cm, cmap="Blues", annot=labels, fmt='' if normalize else 'd')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')