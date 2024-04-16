import argparse
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.optim.lr_scheduler
import yaml
from torch.cuda import amp
from tqdm import tqdm

import val_cls
from models.classifiersimple import *
from utils.general import set_random_seed, LOGGER, increment_path
from utils.load_utils import get_loaders, get_models


def log_values(args, epoch, values):
    file = os.path.join(args.save_dir, 'results.csv')
    n = len(values) + 1  # number of cols
    s = '' if os.path.isfile(file) else (('%20s,' * n % tuple(['epoch', 'train_loss', 'val_loss', 'auroc_macro', 'aupr_macro', 'auroc_weighted', 'aupr_weighted'])).rstrip(',') + '\n')  # add header
    with open(file, 'a') as f:
        f.write(s + ('%20.5g,' * n % tuple([epoch] + values)).rstrip(',') + '\n')


def prepare_env(args):
    set_random_seed(args.seed)

    with open(os.path.join(os.path.dirname(__file__), 'data', args.ind_dataset + '.yaml'), 'r') as f:
        dataset_attributes = yaml.safe_load(f)
        args.n_classes = dataset_attributes['nc']
        args.data_root = dataset_attributes['path']

    args.save_dir = str(increment_path(Path(args.project) / args.name))
    args.mode = 'train'

    Path(os.path.join(args.save_dir, 'weights')).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.save_dir, 'opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    shutil.copyfile(args.hyp, os.path.join(args.save_dir, 'hyp.yaml'))

    with open(args.hyp, 'r') as f:
        hyp = yaml.safe_load(f)

    LOGGER.info(args)
    LOGGER.info(hyp)
    return args, hyp


def train(args):
    # setup all folders and args
    args, hyp = prepare_env(args)

    # load splits
    train_loader, val_loader, _ = get_loaders(args, dataset_root=args.data_root, splits_to_load=['train', 'val'])

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, classifier = get_models(args, train_loader.dataset.n_classes, device)

    # init training
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': hyp['lr0'] / 10},
                                  {'params': classifier.parameters()}], lr=hyp['lr0'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hyp['scheduler_patience'], verbose=True)

    cuda = torch.cuda.is_available()
    scaler = amp.GradScaler(enabled=cuda)

    bce_loss = nn.BCEWithLogitsLoss()
    best_loss = np.inf
    for epoch in range(args.n_epoch):
        epoch_loss = []
        progress_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total=len(train_loader))
        prog_bar_desc = 'Loss: {:.6}'
        for i, (images, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            with amp.autocast(enabled=cuda):
                outputs = model(images)
                outputs = classifier(outputs)

            loss = bce_loss(outputs, labels.float())
            epoch_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

            progress_bar.set_postfix_str(prog_bar_desc.format(np.mean(epoch_loss)))

        aupr_dict, auroc_dict, val_loss = val_cls.validate(model, classifier, val_loader, bce_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            LOGGER.info('Better validation loss - updating weights')
            torch.save(deepcopy(model).half().state_dict(),
                       os.path.join(args.save_dir, 'weights', "backbone_best.pth"))
            torch.save(deepcopy(classifier).half().state_dict(),
                       os.path.join(args.save_dir, 'weights', 'classifier_best.pth'))

        scheduler.step(val_loss)

        log_values(args, epoch+1, [np.mean(epoch_loss), val_loss, auroc_dict['macro'], aupr_dict['macro'], auroc_dict['weighted'], aupr_dict['weighted']])
        LOGGER.info("Epoch [%d/%d] Train Loss: %.4f | Val Loss: %.4f | AuROC_macro: %.4f | AuPR_macro: %.4f | AuROC_weighted: %.4f | AuPR_weighted: %.4f"
              % (epoch+1, args.n_epoch-1, np.mean(epoch_loss), val_loss, auroc_dict['macro'], aupr_dict['macro'], auroc_dict['weighted'], aupr_dict['weighted']))
    print(f'saved to {args.save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--ind_dataset', type=str, default='pascal_voc', choices=['pascal_voc', 'coco2017', 'objects365_in'])
    parser.add_argument('--hyp', type=str, default="data/hyps/hyp.OOD.yaml")
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_workers', type=int, default=6)

    parser.add_argument('--arch', type=str, default='yolo_backbone')
    parser.add_argument('--cfg', type=str, default="models/yolov5s - backbone.yaml")
    parser.add_argument('--weights', type=str, default="weights/yolov5s.pth")
    parser.add_argument('--n_epoch', type=int, default=30, help='# of the epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')

    parser.add_argument('--project', type=str, default='runs/train_cls', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    parser.add_argument('--seed', type=int, default=0, help='save to project/name')
    args = parser.parse_args()

    train(args)
