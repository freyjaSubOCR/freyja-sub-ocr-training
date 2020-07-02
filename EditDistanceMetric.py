from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError
import torch

import difflib


class EditDistanceMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)
        self._edit_distances = 0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self):
        self._edit_distances = 0
        self._num_examples = 0
        return super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        if isinstance(y_pred, torch.Tensor):
            y_pred = list(y_pred.cpu())
        if isinstance(y, torch.Tensor):
            y = list(y.cpu())
            
        for output, label in zip(y_pred, y):
            self._edit_distances += difflib.SequenceMatcher(None, label, output).ratio()
        self._num_examples += len(y)
        return super().update(output)

    @sync_all_reduce("_num_examples", "_edit_distances")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._edit_distances / self._num_examples
