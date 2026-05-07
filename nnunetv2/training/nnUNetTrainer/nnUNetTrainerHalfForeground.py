from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch

class nnUNetTrainerHalfForeground(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.5
