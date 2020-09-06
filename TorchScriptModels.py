import torch
from typing import List
import torch.nn.functional as F


class OCRTorchScript(torch.nn.Module):
    def __init__(self, OCRModel):
        super().__init__()
        self.base_model = OCRModel

    def forward(self, x, boxes):
        x = x.permute((0, 3, 1, 2))
        img_list: List[torch.Tensor] = []
        for i, box in enumerate(boxes):
            img = x[i, :, box[1]:box[3], box[0]:box[2]]
            height = img.shape[1]
            img = F.interpolate(img.unsqueeze(0), scale_factor=40 / height).squeeze(0).true_divide_(255)
            img_list.append(img)
        x = self.cat_list(img_list)
        return self.base_model(x)

    def cat_list(self, images: List[torch.Tensor], fill_value: int = 0):
        max_size: List[int] = torch.tensor([img.shape for img in images]).max(dim=0)[0].tolist()
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, fill_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return batched_imgs


class RCNNTorchScript(torch.nn.Module):
    def __init__(self, RCNNModel):
        super().__init__()
        self.base_model = RCNNModel

    def forward(self, x: torch.Tensor):
        x = x.permute((0, 3, 1, 2)).true_divide_(255)
        img_list = [x[i, :, :, :] for i in range(x.size()[0])]
        _, result = self.base_model(img_list)
        return result

class MSETorchScript(torch.nn.Module):
    def forward(self, input, target):
        return F.mse_loss(input, target)