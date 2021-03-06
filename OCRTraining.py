import logging
from os import path

import torch
import torchvision
from ignite.contrib.engines.common import save_best_model_by_val_score
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ASSReader import ASSReader
from Chars import *
from EditDistanceMetric import EditDistanceMetric
from OCRModels import *
from SubtitleDataset import SubtitleDatasetOCRV3


def train(model, model_name, train_dataloader, eval_dataloader, labels_name, trainer_name='ocr', backbone_url=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    def _prepare_batch(batch, device=None, non_blocking=False):
        """Prepare batch for training: pass to a device with options.
        """
        images, labels = batch
        images = images.to(device)
        labels = [label.to(device) for label in labels]
        return (images, labels)

    writer = SummaryWriter(log_dir=f'logs/{trainer_name}/{model_name}')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=250, cooldown=100, min_lr=1e-6)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        # loss = model(x, y)
        # loss.backward()
        # optimizer.step()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    trainer = Engine(_update)
    evaluator = create_supervised_evaluator(model, prepare_batch=_prepare_batch,
                                            metrics={'edit_distance': EditDistanceMetric()}, device=device)

    if path.exists(f'{trainer_name}_{model_name}_checkpoint.pt'):
        checkpoint = torch.load(f'{trainer_name}_{model_name}_checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logging.info(f'load checkpoint {trainer_name}_{model_name}_checkpoint.pt')
    elif path.exists(f'{model_name}_backbone.pt'):
        pretrained_dict = torch.load(f'{model_name}_backbone.pt')['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'neck.' not in k and 'fc.' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info(f'load transfer parameters from {model_name}_backbone.pt')
    elif backbone_url is not None:
        pretrained_dict = torch.hub.load_state_dict_from_url(backbone_url, progress=False)
        model_dict = model.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.backbone.load_state_dict(model_dict)
        logging.info(f'load backbone from {backbone_url}')

    early_stop_arr = [0.0]

    def early_stop_score_function(engine):
        val_acc = engine.state.metrics['edit_distance']
        if val_acc < 0.8:  # do not early stop when acc is less than 0.9
            early_stop_arr[0] += 0.000001
            return early_stop_arr[0]
        return val_acc

    early_stop_handler = EarlyStopping(patience=20, score_function=early_stop_score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    checkpoint_handler = ModelCheckpoint(f'models/{trainer_name}/{model_name}', model_name, n_saved=10, create_dir=True)
    # trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), checkpoint_handler,
    #                           {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler})
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), checkpoint_handler,
                              {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'scaler': scaler})

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch[{}]: {} - Loss: {:.4f}, Lr: {}"
                     .format(trainer.state.epoch, trainer.state.iteration, trainer.state.output, lr))
        writer.add_scalar("training/loss", trainer.state.output, trainer.state.iteration)
        writer.add_scalar("training/learning_rate", lr, trainer.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def step_lr(trainer):
        lr_scheduler.step(trainer.state.output)

    @trainer.on(Events.ITERATION_COMPLETED(every=1000))
    def log_training_results(trainer):
        evaluator.run(eval_dataloader)
        metrics = evaluator.state.metrics
        logging.info("Eval Results - Epoch[{}]: {} - Avg edit distance: {:.4f}"
                     .format(trainer.state.epoch, trainer.state.iteration, metrics['edit_distance']))
        writer.add_scalar("evaluation/avg_edit_distance", metrics['edit_distance'], trainer.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def read_lr_from_file(trainer):
        if path.exists('lr.txt'):
            with open('lr.txt', 'r', encoding='utf-8') as f:
                lr = float(f.read())
            for group in optimizer.param_groups:
                group['lr'] = lr

    trainer.run(train_dataloader, max_epochs=1)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def OCR_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = list(targets)
    return batched_imgs, batched_targets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chars = SC5000Chars()
    texts = [text for text in ASSReader().getCompatible(chars) if len(text) <= 22]
    train_dataset = SubtitleDatasetOCRV3(chars=chars, styles_json=path.join('data', 'styles', 'styles_yuan.json'),
                                         texts=texts)
    eval_dataset = SubtitleDatasetOCRV3(styles_json=path.join('data', 'styles_eval', 'styles_yuan.json'),
                                        samples=path.join('data', 'samples_eval'),
                                        chars=chars, start_frame=500, end_frame=500 + 256, texts=texts)

    train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=OCR_collate_fn, num_workers=8, timeout=60)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, collate_fn=OCR_collate_fn)

    model = CRNNEfficientNetB3(len(chars.chars), rnn_hidden=768, bidirectional=True)

    train(model, 'CRNNEfficientNetB3_768_bi', train_dataloader, eval_dataloader, chars.chars, 'ocr_v3_amp_SC5000Chars_yuan',
          backbone_url='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth')
