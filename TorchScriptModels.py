import torch


class OCRTorchScript(torch.nn.Module):
    def __init__(self, OCRModel):
        super().__init__()
        self.base_model = OCRModel

    def forward(self, x):
        return self.base_model(x)


class RCNNTorchScript(torch.nn.Module):
    def __init__(self, RCNNModel):
        super().__init__()
        self.base_model = RCNNModel

    def forward(self, x):
        img_list = [x[i, :, :, :] for i in range(x.size()[0])]
        _, result = self.base_model(img_list)
        return result
