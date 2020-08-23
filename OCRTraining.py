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
from OCRModels import CCNNResnext50, CRNNResnext50, CRNNResnext101
from SubtitleDataset import SubtitleDatasetOCR


def train(model, model_name, train_dataloader, test_dataloader, eval_dataloader, labels_name, trainer_name='ocr', backbone_url=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def _prepare_batch(batch, device=None, non_blocking=False):
        """Prepare batch for training: pass to a device with options.
        """
        images, labels = batch
        images = images.to(device)
        labels = [label.to(device) for label in labels]
        return (images, labels)

    writer = SummaryWriter(log_dir=f'logs/{trainer_name}/{model_name}')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=200)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(_update)
    evaluator = create_supervised_evaluator(model, prepare_batch=_prepare_batch,
                                            metrics={'edit_distance': EditDistanceMetric()}, device=device)
    evaluator2 = create_supervised_evaluator(model, prepare_batch=_prepare_batch,
                                             metrics={'edit_distance': EditDistanceMetric()}, device=device)

    if path.exists(f'{trainer_name}_{model_name}_checkpoint.pt'):
        checkpoint = torch.load(f'{trainer_name}_{model_name}_checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
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
        if val_acc < 0.5:  # do not early stop when acc is less than 0.5
            early_stop_arr[0] += 0.000001
            return early_stop_arr[0]
        return val_acc

    early_stop_handler = EarlyStopping(patience=100, score_function=early_stop_score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    checkpoint_handler = ModelCheckpoint(f'models/{trainer_name}/{model_name}', model_name, n_saved=10, create_dir=True)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), checkpoint_handler,
                              {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler})

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

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_results(trainer):
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        logging.info("Training Results - Epoch[{}]: {} - Avg edit distance: {:.4f}"
                     .format(trainer.state.epoch, trainer.state.iteration, metrics['edit_distance']))
        writer.add_scalar("training/avg_edit_distance", metrics['edit_distance'], trainer.state.iteration)

        evaluator2.run(eval_dataloader)
        metrics = evaluator2.state.metrics
        logging.info("Eval Results - Epoch[{}]: {} - Avg edit distance: {:.4f}"
                     .format(trainer.state.epoch, trainer.state.iteration, metrics['edit_distance']))
        writer.add_scalar("evaluation/avg_edit_distance", metrics['edit_distance'], trainer.state.iteration)

        model.eval()
        test_data = iter(eval_dataloader)
        x, y = next(test_data)
        x = x.to(device)
        y_pred = model(x)

        for label, output in zip(y, y_pred):
            result = ''
            result = result + ''.join([labels_name[i] for i in label]) + '\n'
            result = result + ''.join([labels_name[i] for i in output])
            writer.add_text("evaluation/example_result", result, trainer.state.iteration)
            break

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
    chars = CJKChars()
    texts = [text for text in ASSReader().getCompatible(chars) if len(text) <= 15]
    train_dataset = SubtitleDatasetOCR(chars=chars, styles_json=path.join('data', 'styles', 'styles_hei.json'), texts=texts)
    test_dataset = SubtitleDatasetOCR(chars=chars, start_frame=500, end_frame=500 + 64, grayscale=1,
                                      styles_json=path.join('data', 'styles', 'styles_hei.json'), texts=texts)
    eval_dataset = SubtitleDatasetOCR(styles_json=path.join('data', 'styles_eval', 'styles_hei.json'),
                                      samples=path.join('data', 'samples_eval'),
                                      chars=chars, start_frame=500, end_frame=500 + 64, grayscale=1, texts=texts)

    train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=OCR_collate_fn, num_workers=8, timeout=60)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=OCR_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, collate_fn=OCR_collate_fn)

    model = CRNNResnext101(len(chars.chars), rnn_hidden=1024)

    train(model, 'CRNNResnext101_1024', train_dataloader, test_dataloader, eval_dataloader, chars.chars, 'ocr_CJKChars_hei',
          backbone_url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth')
