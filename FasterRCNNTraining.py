from os import path

import torch
import torchvision
from ignite.contrib.engines.common import save_best_model_by_val_score
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from IPython.display import display
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Chars import *
from IOUMetric import IOUMetric
from SubtitleDataset import SubtitleDatasetRCNN


def train(model, model_name, train_dataloader, test_dataloader, trainer_name='bb_detection'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def _prepare_batch(batch, device=None, non_blocking=False):
        """Prepare batch for training: pass to a device with options.
        """
        images, boxes = batch
        images = [image.to(device) for image in images]
        targets = [{'boxes': box.to(device), 'labels': torch.ones((1), dtype=torch.int64).to(device)} for box in boxes]
        return images, targets

    writer = SummaryWriter(log_dir=path.join('logs', trainer_name, model_name))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=250)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)

        loss_dict = model(x, y)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        losses.backward()
        optimizer.step()
        return loss_value

    trainer = Engine(_update)
    evaluator = create_supervised_evaluator(model, prepare_batch=_prepare_batch,
                                            metrics={'iou': IOUMetric()}, device=device)

    if path.exists(f'{trainer_name}_{model_name}_checkpoint.pt'):
        checkpoint = torch.load(f'{trainer_name}_{model_name}_checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.load_state_dict(checkpoint['trainer'])

    def early_stop_score_function(engine):
        val_acc = engine.state.metrics['iou']
        return val_acc

    early_stop_handler = EarlyStopping(patience=20, score_function=early_stop_score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    checkpoint_handler = ModelCheckpoint(f'models/{trainer_name}/{model_name}', model_name, n_saved=20, create_dir=True)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), checkpoint_handler,
                              {'model': model, 'optimizer': optimizer, 'trainer': trainer})

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch[{}]: {} - Loss: {:.4f}, Lr: {}"
              .format(trainer.state.epoch, trainer.state.iteration, trainer.state.output, lr))
        writer.add_scalar("training/loss", trainer.state.output, trainer.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_results(trainer):
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch[{}]: {} - Avg IOU: {:.4f}"
              .format(trainer.state.epoch, trainer.state.iteration, metrics['iou']))
        writer.add_scalar("training/avg_iou", metrics['iou'], trainer.state.iteration)

        model.eval()
        test_data = iter(test_dataloader)
        x, y = _prepare_batch(next(test_data), device)
        y_pred = model(x)

        for image, output in zip(x, y_pred):
            writer.add_image_with_boxes("training/example_result", image, output['boxes'], trainer.state.iteration)
            break
        model.train()

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def step_lr(trainer):
        lr_scheduler.step(trainer.state.output)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def read_lr_from_file(trainer):
        if path.exists('lr.txt'):
            with open('lr.txt', 'r', encoding='utf-8') as f:
                lr = float(f.read())
            for group in optimizer.param_groups:
                group['lr'] = lr

    trainer.run(train_dataloader, max_epochs=100)


def RCNN_collate_fn(batch):
    imgs, boxes = tuple(map(list, zip(*batch)))
    # boxes = torch.stack(boxes)
    return imgs, boxes


if __name__ == '__main__':
    train_dataset = SubtitleDatasetRCNN(chars=SC3500Chars())
    test_dataset = SubtitleDatasetRCNN(chars=SC3500Chars(), start_frame=500, end_frame=500 + 64)

    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=8, collate_fn=RCNN_collate_fn, timeout=60)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=RCNN_collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True,
                                                                 num_classes=2, min_size=400, max_size=600)

    train(model, 'FasterRCNN', train_dataloader, test_dataloader, 'bb_detection')
