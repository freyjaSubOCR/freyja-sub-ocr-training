import torch
import torchvision
from OCRModels import CRNNResnext101_JIT
from Chars import *

rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True,
                                                                  num_classes=2, min_size=400, max_size=600)

rcnn_checkpoint = torch.load('models/FasterRCNN_last_checkpoint.pt')
rcnn_model.load_state_dict(rcnn_checkpoint['model'])
rcnn_model.eval()
rcnn_model_script = torch.jit.script(rcnn_model)
torch.jit.save(rcnn_model_script, 'models/object_detection.pt')


chars = SC3500Chars()
ocr_model = CRNNResnext101_JIT(len(chars.chars), rnn_hidden=1280)

ocr_checkpoint = torch.load('models/SC3500Chars_yuan_CRNNResnext101_1280_checkpoint.pt')
ocr_model.load_state_dict(ocr_checkpoint['model'])
ocr_model.eval()
ocr_model_script = torch.jit.script(ocr_model)
torch.jit.save(ocr_model_script, 'models/ocr_SC3500Chars_yuan.pt')

ocr_checkpoint = torch.load('models/SC3500Chars_hei_CRNNResnext101_1280_checkpoint.pt')
ocr_model.load_state_dict(ocr_checkpoint['model'])
ocr_model.eval()
ocr_model_script = torch.jit.script(ocr_model)
torch.jit.save(ocr_model_script, 'models/ocr_SC3500Chars_hei.pt')

chars.export('models/ocr_SC3500Chars.txt')
