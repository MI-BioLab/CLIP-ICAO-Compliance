from pathlib import Path

import torch
import torchvision.io as tio


class OrderedSet:
    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def add(self, value):
        self._data[value] = None

    def update(self, values):
        for value in values:
            self.add(value)
    
    def __contains__(self, value):
        return value in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)


def load_image(filename: Path) -> torch.Tensor:
    img = tio.read_image(filename, mode="RGB").to(torch.float32)
    return img.unsqueeze(0) / 255.0


def compute_tpr_fpr_accuracies(similarities: torch.Tensor, thresholds: torch.Tensor, actual: torch.Tensor, chunk_size: int = 1000, output_confusion: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # distances is a (n_samples,) tensor
    # thresholds is a (n_thresholds,) tensor
    # actual is a (n_samples,) tensor
    # Returns a tuple of (tpr, fpr, accuracies), each of which is a (n_thresholds,) tensor
    if thresholds.ndim == 0:
        thresholds = thresholds.unsqueeze(0)
    n_thresholds = thresholds.shape[0]
    actual_expanded = actual.unsqueeze(1)  # (n_samples, 1)
    tpr = torch.zeros(n_thresholds, device=similarities.device)
    fpr = torch.zeros(n_thresholds, device=similarities.device)
    acc = torch.zeros(n_thresholds, device=similarities.device)
    confusion = torch.zeros(n_thresholds, 2, 2, device=similarities.device)
    for start in range(0, n_thresholds, chunk_size):
        end = min(start + chunk_size, n_thresholds)
        chunk_thresholds = thresholds[start:end]
        predictions = similarities.unsqueeze(1) >= chunk_thresholds.unsqueeze(0)  # (n_samples, chunk_size)
        tp = (predictions & actual_expanded).sum(dim=0, dtype=torch.int32)
        fp = (predictions & ~actual_expanded).sum(dim=0, dtype=torch.int32)
        tn = (~predictions & ~actual_expanded).sum(dim=0, dtype=torch.int32)
        fn = (~predictions & actual_expanded).sum(dim=0, dtype=torch.int32)
        tpr[start:end] = torch.nan_to_num(tp / (tp + fn), nan=0.0, posinf=0.0, neginf=0.0)
        fpr[start:end] = torch.nan_to_num(fp / (fp + tn), nan=0.0, posinf=0.0, neginf=0.0)
        acc[start:end] = (tp + tn) / similarities.shape[0]
        confusion[start:end, 0, 0] = tn
        confusion[start:end, 0, 1] = fn
        confusion[start:end, 1, 0] = fp
        confusion[start:end, 1, 1] = tp
    if output_confusion:
        return tpr, fpr, acc, confusion
    else:
        return tpr, fpr, acc


def compute_eer(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    thresholds = torch.linspace(0, 1, 1000).to(y_pred.device)
    tpr, fpr, _ = compute_tpr_fpr_accuracies(y_pred, thresholds, y_true)
    eer_idx = torch.argmin(torch.abs(fpr - (1 - tpr)))
    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
    return eer, thresholds[eer_idx]