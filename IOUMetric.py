from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError


class IOUMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)
        self._iou = 0
        self._num_examples = 0

    # code from https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
    def _calc_iou(self, boxA, boxB):
        # if boxes dont intersect
        if self._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = self._getIntersectionArea(boxA, boxB)
        union = self._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    def _boxesIntersect(self, boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    def _getIntersectionArea(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    def _getUnionAreas(self, boxA, boxB, interArea=None):
        area_A = self._getArea(boxA)
        area_B = self._getArea(boxB)
        if interArea is None:
            interArea = self._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    def _getArea(self, box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    @reinit__is_reduced
    def reset(self):
        self._iou = 0
        self._num_examples = 0
        return super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        for output, truth in zip(y_pred, y):
            ious = [self._calc_iou(truth['boxes'][0], box) for box in output['boxes']]
            if len(ious) != 0:
                self._iou += max(ious)
        self._num_examples += len(y)
        return super().update(output)

    @sync_all_reduce("_num_examples", "_iou")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._iou / self._num_examples
