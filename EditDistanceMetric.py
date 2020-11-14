from logging import log
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError
import torch

import difflib
import logging


class EditDistanceMetric(Metric):
    def __init__(self, chars, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)
        self._edit_distances = 0
        self._num_examples = 0
        self.chars = chars

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
            # label_text = ''.join([self.chars[i] for i in label])
            # output_text = ''.join([self.chars[i] for i in output])
            # if label_text != output_text:
            #     logging.info(f'{label_text}\n{output_text}')
            self._edit_distances += difflib.SequenceMatcher(None, label, output).ratio()
        self._num_examples += len(y)
        return super().update(output)

    @sync_all_reduce("_num_examples", "_edit_distances")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._edit_distances / self._num_examples
