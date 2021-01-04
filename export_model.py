import torch
import torchvision

from Chars import *
from OCRModels import *
from TorchScriptModels import *

device = torch.device('cuda')


def export_rcnn_model():
    rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True,
                                                                      num_classes=2, min_size=400, max_size=600)

    rcnn_checkpoint = torch.load('models/FasterRCNN_last_checkpoint.pt')
    rcnn_model.load_state_dict(rcnn_checkpoint['model'])
    rcnn_model = RCNNTorchScript(rcnn_model)
    rcnn_model.eval()
    rcnn_model.to(device)
    rcnn_model_script = torch.jit.script(rcnn_model)
    torch.jit.save(rcnn_model_script, 'models/object_detection.torchscript')


def export_ocr_efficientnet_model():
    chars = SC5000Chars()

    chars.export('models/ocr_SC5000Chars.txt')

    ocr_model = CRNNEfficientNetB3(len(chars.chars), rnn_hidden=768, bidirectional=True)
    ocr_checkpoint = torch.load('models/ocr_v3_amp_SC5000Chars_yuan/CRNNEfficientNetB3_768_bi/CRNNEfficientNetB3_768_bi_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model.backbone.set_swish(memory_efficient=False)
    ocr_model.to(device)
    ocr_model.eval()
    ocr_model.backbone = torch.jit.trace(ocr_model.backbone, torch.rand(1, 3, 40, 900).to(device))
    ocr_model = OCRTorchScriptV3(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocrV3_SC5000Chars_yuan.torchscript')

    ocr_model = CRNNEfficientNetB3(len(chars.chars), rnn_hidden=768, bidirectional=True)
    ocr_checkpoint = torch.load('models/ocr_v3_amp_SC5000Chars_hei/CRNNEfficientNetB3_768_bi/CRNNEfficientNetB3_768_bi_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model.backbone.set_swish(memory_efficient=False)
    ocr_model.to(device)
    ocr_model.eval()
    ocr_model.backbone = torch.jit.trace(ocr_model.backbone, torch.rand(1, 3, 40, 900).to(device))
    ocr_model = OCRTorchScriptV3(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocrV3_SC5000Chars_hei.torchscript')

    chars = TC5000Chars()

    chars.export('models/ocr_TC5000Chars.txt')

    ocr_model = CRNNEfficientNetB3(len(chars.chars), rnn_hidden=768, bidirectional=True)
    ocr_checkpoint = torch.load('models/ocr_v3_amp_TC5000Chars_yuan/CRNNEfficientNetB3_768_bi/CRNNEfficientNetB3_768_bi_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model.backbone.set_swish(memory_efficient=False)
    ocr_model.to(device)
    ocr_model.eval()
    ocr_model.backbone = torch.jit.trace(ocr_model.backbone, torch.rand(1, 3, 40, 900).to(device))
    ocr_model = OCRTorchScriptV3(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocrV3_TC5000Chars_yuan.torchscript')

    ocr_model = CRNNEfficientNetB3(len(chars.chars), rnn_hidden=768, bidirectional=True)
    ocr_checkpoint = torch.load('models/ocr_v3_amp_TC5000Chars_hei/CRNNEfficientNetB3_768_bi/CRNNEfficientNetB3_768_bi_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model.backbone.set_swish(memory_efficient=False)
    ocr_model.to(device)
    ocr_model.eval()
    ocr_model.backbone = torch.jit.trace(ocr_model.backbone, torch.rand(1, 3, 40, 900).to(device))
    ocr_model = OCRTorchScriptV3(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocrV3_TC5000Chars_hei.torchscript')


def export_ocr_resnet_model():
    chars = SC5000Chars()

    ocr_model = CRNNResnext101(len(chars.chars), rnn_hidden=1280)
    ocr_checkpoint = torch.load('models/SC3500Chars_yuan_CRNNResnext101_1280_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model = OCRTorchScript(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocr_SC3500Chars_yuan.torchscript')

    ocr_model = CRNNResnext101(len(chars.chars), rnn_hidden=1280)
    ocr_checkpoint = torch.load('models/SC3500Chars_hei_CRNNResnext101_1280_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model = OCRTorchScript(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocr_SC3500Chars_hei.torchscript')

    chars.export('models/ocr_SC3500Chars.txt')

    chars = TC3600Chars()

    ocr_model = CRNNResnext101(len(chars.chars), rnn_hidden=1280)
    ocr_checkpoint = torch.load('models/TC3600Chars_yuan_CRNNResnext101_1280_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model = OCRTorchScript(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocr_TC3600Chars_yuan.torchscript')

    ocr_model = CRNNResnext101(len(chars.chars), rnn_hidden=1280)
    ocr_checkpoint = torch.load('models/TC3600Chars_hei_CRNNResnext101_1280_checkpoint.pt')
    ocr_model.load_state_dict(ocr_checkpoint['model'])
    ocr_model = OCRTorchScript(ocr_model)
    ocr_model.eval()
    ocr_model.to(device)
    ocr_model_script = torch.jit.script(ocr_model)
    torch.jit.save(ocr_model_script, 'models/ocr_TC3600Chars_hei.torchscript')

    chars.export('models/ocr_TC3600Chars.txt')


def export_mse():
    mse_model = MSETorchScript()
    mse_model.to(device)
    mse_model_script = torch.jit.script(mse_model)
    torch.jit.save(mse_model_script, 'models/mse.torchscript')


if __name__ == "__main__":
    export_ocr_efficientnet_model()
