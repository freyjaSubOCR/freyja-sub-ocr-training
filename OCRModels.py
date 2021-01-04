import torch
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.vgg import VGG
from typing import Optional, List
from efficientnet_pytorch import EfficientNet, get_model_params


class RNNDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size=256, bidirectional=True):
        super(RNNDecoder, self).__init__()
        self.rnn = torch.nn.GRU(in_channels, hidden_size, num_layers=2, bidirectional=bidirectional)
        self.out_channels = hidden_size * 2 if bidirectional else hidden_size
        self.hidden_size = hidden_size

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.view(b, c, w * h)
        x = x.permute(2, 0, 1)  # (w*h, b, c)
        x, _ = self.rnn(x)  # (seq_len, batch, input_size) -> (seq_len, batch, num_directions * hidden_size)
        x = x.permute(1, 0, 2)  # (batch, seq_len, num_directions * hidden_size)
        return x


class CNNDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(CNNDecoder, self).__init__()
        inter_channels = in_channels // 4
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(inter_channels, out_channels, 1)
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.cnn(x)
        x_shape = x.size()
        x = x.view(x_shape[0], x_shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


class OCR(torch.nn.Module):
    '''
    Basic network structure with CTC loss
    '''

    def __init__(self, n_classes, backbone, neck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.fc = torch.nn.Linear(self.neck.out_channels, n_classes)
        self.criterion = torch.nn.CTCLoss()

    def forward(self, x, targets: Optional[List[torch.Tensor]] = None):
        '''
        x = list([channels, height, width]), image
        targets = list([class1, class2, ...], [class1, class2, ...]), len=batch (only in train mode)
        '''
        cur_batch_size = x.shape[0]

        if self.training and targets is None:
            raise AttributeError('target cannot be none in training mode')
        x = self.backbone(x)
        x = self.neck(x)
        x = self.fc(x)

        if self.training and not torch.jit.is_scripting():
            assert(targets is not None)
            x = x.permute((1, 0, 2))
            x = x.log_softmax(2)
            input_lengths = torch.full((cur_batch_size,), x.size()[0], dtype=torch.long)
            target_length = torch.tensor([target.size()[0] for target in targets], dtype=torch.long)
            targets_cat = torch.cat(targets)
            loss = self.criterion(x, targets_cat, input_lengths, target_length)
            return loss

        else:
            classes: List[List[int]] = []
            x = x.log_softmax(2)
            values, indices = x.max(dim=2)
            values, indices = values.cpu(), indices.cpu()

            for prob_string, class_string in zip(values, indices):
                _class: List[int] = []
                for i in range(len(class_string)):
                    if class_string[i] != 0 and (i == 0 or class_string[i] != class_string[i - 1]):
                        _class.append(class_string[i].item())
                classes.append(_class)

            return classes


class Resnet50Backbone(ResNet):
    '''resnet50 backbone'''

    def __init__(self):
        block = Bottleneck
        layers = [3, 4, 6, 3]
        super().__init__(block, layers)
        self.inplanes = 1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Resnext50_32x4dBackbone(ResNet):
    '''resnext50_32x4d backbone'''

    def __init__(self):
        block = Bottleneck
        layers = [3, 4, 6, 3]
        super().__init__(block, layers, groups=32, width_per_group=4)
        self.inplanes = 1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Resnext101_32x8dBackbone(ResNet):
    '''resnext101_32x8d backbone'''

    def __init__(self):
        block = Bottleneck
        layers = [3, 4, 23, 3]
        super().__init__(block, layers, groups=32, width_per_group=8)
        self.inplanes = 1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class VGGBackbone(VGG):
    def _vgg_make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'M2':
                layers += [torch.nn.MaxPool2d(kernel_size=(1, 2), stride=2)]
            else:
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                in_channels = v
        return torch.nn.Sequential(*layers)

    def __init__(self):
        features = self._vgg_make_layers([64, 'M', 128, 'M', 256, 256, 'M2', 512, 512, 'M2', 512, 512], batch_norm=True)
        super().__init__(features, init_weights=False)

    def forward(self, x):
        x = self.features(x)
        return x


class EfficientNetB3Backbone(EfficientNet):
    '''EfficientNet-B3 backbone'''

    def __init__(self):
        blocks_args, global_params = get_model_params('efficientnet-b3', None)
        global_params = global_params._replace(image_size=(40, 900))
        blocks_args[-2] = blocks_args[-2]._replace(stride=[1])
        super().__init__(blocks_args, global_params)

    def forward(self, x):
        x = self.extract_features(x)
        return x


class EfficientNetB5Backbone(EfficientNet):
    '''EfficientNet-B5 backbone'''

    def __init__(self):
        blocks_args, global_params = get_model_params('efficientnet-b5', None)
        global_params = global_params._replace(image_size=(40, 900))
        blocks_args[-2] = blocks_args[-2]._replace(stride=[1])
        super().__init__(blocks_args, global_params)

    def forward(self, x):
        x = self.extract_features(x)
        return x


class EfficientNetB7Backbone(EfficientNet):
    '''EfficientNet-B7 backbone'''

    def __init__(self):
        blocks_args, global_params = get_model_params('efficientnet-b7', None)
        global_params = global_params._replace(image_size=(40, 900))
        blocks_args[-2] = blocks_args[-2]._replace(stride=[1])
        super().__init__(blocks_args, global_params)

    def forward(self, x):
        x = self.extract_features(x)
        return x


class CRNNVGG(OCR):
    def __init__(self, n_classes):
        backbone = VGGBackbone()
        neck = RNNDecoder(512, 256)
        super().__init__(n_classes, backbone, neck)


class CCNNResnext50(OCR):
    def __init__(self, n_classes):
        backbone = Resnext50_32x4dBackbone()
        neck = CNNDecoder(2048, 256)
        super().__init__(n_classes, backbone, neck)


class CRNNResnext50(OCR):
    def __init__(self, n_classes, rnn_hidden=256):
        backbone = Resnext50_32x4dBackbone()
        neck = RNNDecoder(2048, hidden_size=rnn_hidden)
        super().__init__(n_classes, backbone, neck)


class CRNNResnext101(OCR):
    def __init__(self, n_classes, rnn_hidden=256):
        backbone = Resnext101_32x8dBackbone()
        neck = RNNDecoder(2048, hidden_size=rnn_hidden)
        super().__init__(n_classes, backbone, neck)


class CRNNEfficientNetB3(OCR):
    def __init__(self, n_classes, rnn_hidden=256, bidirectional=True):
        backbone = EfficientNetB3Backbone()
        neck = RNNDecoder(1536, hidden_size=rnn_hidden, bidirectional=bidirectional)
        super().__init__(n_classes, backbone, neck)


class CRNNEfficientNetB5(OCR):
    def __init__(self, n_classes, rnn_hidden=256, bidirectional=True):
        backbone = EfficientNetB5Backbone()
        neck = RNNDecoder(2048, hidden_size=rnn_hidden, bidirectional=bidirectional)
        super().__init__(n_classes, backbone, neck)


class CRNNEfficientNetB7(OCR):
    def __init__(self, n_classes, rnn_hidden=256, bidirectional=True):
        backbone = EfficientNetB7Backbone()
        neck = RNNDecoder(2560, hidden_size=rnn_hidden, bidirectional=bidirectional)
        super().__init__(n_classes, backbone, neck)
