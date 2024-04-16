import argparse
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda import amp
from torch.optim import Adam, lr_scheduler
from tqdm.auto import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val_yolood  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_img_size, check_suffix, check_yaml, colorstr,
                           increment_path, init_seeds, intersect_dicts, methods, print_args, strip_optimizer)
from utils.loggers import Loggers
from utils.loss import ComputeLossOOD
from utils.metrics import fitness
from utils.torch_utils import de_parallel, select_device, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_directories(save_dir):
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    return w, last, best


def get_hyperparameters(hyp, save_dir):
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    return hyp


def get_loggers(save_dir, weights, hyp, callbacks):
    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
    return loggers


def get_config(device, data, seed):
    # Config
    cuda = device.type != 'cpu'
    init_seeds(seed)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    return cuda, data_dict, train_path, val_path, nc, names, is_coco


def get_model(weights, cfg, nc, hyp, resp_cell_offset, device, exclude, freeze, imgsz):
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        # torch.save(ckpt['model'].float().state_dict(), 'yolov5s.pth')
        model = Model(cfg or ckpt['model'].yaml,
                      nc=nc,
                      resp_cell_offset=resp_cell_offset,
                      imgsz=imgsz).to(device)  # create
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=[str(num) for num in exclude])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        del ckpt, csd
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False
    return model


def get_sizes(model, batch_size, loggers):
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})
    return gs, imgsz, batch_size


def get_optimizer(hyp, model, exclude):
    seperator_idx = min(exclude) if len(exclude) > 0 else -1
    if seperator_idx > 0:
        optimizer = Adam(model.model[:seperator_idx].parameters(), lr=hyp['lr0'] / 10)
        optimizer.add_param_group({'params': model.model[seperator_idx:].parameters(), 'lr': hyp['lr0']})
    else:
        optimizer = Adam(model.model.parameters(), lr=hyp['lr0'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=hyp['scheduler_patience'], verbose=True)

    return optimizer, scheduler


def get_loaders(train_path, imgsz, batch_size, gs, hyp, workers, nc, data, val_path, noval, model):
    # Train loader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              pad=0.5,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        model.half().float()  # pre-reduce anchor precision

    return train_loader, dataset, nb, val_loader


def set_model_attributes(hyp, model, nc, names):
    # Model attributes
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, data, cfg, noval, nosave, workers, freeze, exclude, resp_cell_offset =\
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.data, opt.cfg, opt.noval, opt.nosave, \
        opt.workers, opt.freeze, opt.exclude, opt.resp_cell_offset

    w, last, best = get_directories(save_dir)

    hyp = get_hyperparameters(hyp, save_dir)

    loggers = get_loggers(save_dir, weights, hyp, callbacks)

    cuda, data_dict, train_path, val_path, nc, names, is_coco = get_config(device, data, opt.seed)

    model = get_model(weights, cfg, nc, hyp, resp_cell_offset, device, exclude, freeze, opt.imgsz)

    gs, imgsz, batch_size = get_sizes(model, batch_size, loggers)

    optimizer, scheduler = get_optimizer(hyp, model, exclude)

    train_loader, dataset, nb, val_loader = get_loaders(train_path, imgsz, batch_size, gs, hyp, workers, nc, data, val_path, noval, model)

    set_model_attributes(hyp, model, nc, names)

    # Start training
    start_epoch, best_fitness = 0, np.inf
    t0 = time.time()
    results = (0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move

    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLossOOD(model, resp_cell_offset=resp_cell_offset)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(2, device=device)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in (-1, 0):
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=150)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

            # Log
            if RANK in (-1, 0):
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        if RANK in (-1, 0):
            # mAP
            final_epoch = (epoch + 1 == epochs)
            if not noval or final_epoch:  # Calculate mAP
                results = val_yolood.run(data_dict,
                                   model=model,
                                   dataloader=val_loader,
                                   callbacks=callbacks,
                                   compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi < best_fitness:
                best_fitness = fi
            log_vals = list(loss_item.item() for loss_item in mloss) + list(results) # + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or final_epoch:  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'optimizer': optimizer.state_dict(),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                torch.cuda.empty_cache()

        scheduler.step(fi)

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in (-1, 0):
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results = val_yolood.run(
                        data_dict,
                        model=attempt_load(f, device).half(),
                        dataloader=val_loader,
                        callbacks=callbacks,
                        compute_loss=compute_loss)

        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s - OOD.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/pascal_voc.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.OOD.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', default=False, action='store_true', help='only validate final epoch')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--workers', type=int, default=6, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train_yolood', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers:backbone=10, first3=0 1 2')
    parser.add_argument('--exclude', nargs='+', type=int, default=list(range(10, 24)), help='Exclude pretrained layers: backbone=10, first3=0 1 2') # list(range(10, 24))
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--resp_cell_offset', nargs='+', type=float, default=[0.5, 0.1, 0.0], help='Center cell is responsible for detection / all bbox')
    parser.add_argument('--seed', type=int, default=0, help='Seed')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in (-1, 0):
        print_args(vars(opt))

    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    train(opt.hyp, opt, device, callbacks)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
