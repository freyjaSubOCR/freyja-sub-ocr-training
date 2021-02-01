from logging import log
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError
import torch

import difflib
import logging


class ResultMetric(Metric):
    def __init__(self, chars=None, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)
        self._result = ''
        self.chars = chars

    @reinit__is_reduced
    def reset(self):
        self._result = ''
        return super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        if isinstance(y_pred, torch.Tensor):
            y_pred = list(y_pred.cpu())
        if isinstance(y, torch.Tensor):
            y = list(y.cpu())

        for output, label in zip(y_pred, y):
            label_text = ''.join([self.chars[i] for i in label])
            output_text = ''.join([self.chars[i] for i in output])
            self._result += f'{label_text}: {output_text}\n'
        return super().update(output)

    @sync_all_reduce("_result")
    def compute(self):
        return self._result
