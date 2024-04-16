import os
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn import metrics as metrics

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks
from utils.general import LOGGER
from utils.metrics import auroc_aupr_scores
from utils.torch_utils import time_sync


def calc_mAP(gts, preds, nc):
    final_MAPs = []
    for i in range(0, nc):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            precision, recall, thresholds = metrics.precision_recall_curve(gts[i], preds[i])
            final_MAPs.append(metrics.auc(recall, precision))
    return np.mean(final_MAPs)


def calc_AUROC(gts, preds, nc):
    final_AUCs = []
    for i in range(0, nc):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fpr, tpr, thresholds = metrics.roc_curve(gts[i], preds[i])
            final_AUCs.append(metrics.auc(fpr, tpr))
    return np.mean(final_AUCs)


@torch.no_grad()
def run(
        data,
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        callbacks=Callbacks(),
        compute_loss=None,
):

    device = next(model.parameters()).device # get model device, PyTorch model
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    nc = int(data['nc'])  # number of classes

    s = ('%10s' * 5) % ('val_loss', 'auroc_m', 'aupr_m', 'auroc_w', 'aupr_w')
    dt = [0.0, 0.0, 0.0]
    loss = torch.zeros(2, device=device)
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=150)  # progress bar
    auc_dict = {key: {i: [] for i in range(0, nc)} for key in ['gts', 'preds']}
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im)
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # obj, cls

        pred = (out[..., 0:1] * out[..., 1:]).max(dim=1)[0]

        tcls, indices = compute_loss.build_targets(train_out, targets)
        tcls = [cls for cls in tcls if cls is not None]
        indices = [idx for idx in indices if idx is not None]
        gt = torch.full_like(pred, 0)

        t_indices = torch.stack([torch.concat([idx[0] for idx in indices]),
                                 torch.concat([cls for cls in tcls])], 1)
        gt[t_indices[:, 0], t_indices[:, 1]] = 1.

        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        for label in range(0, nc):
            auc_dict['gts'][label].extend(gt[:, label])
            auc_dict['preds'][label].extend(pred[:, label])

        callbacks.run('on_val_batch_end')

    # Compute metrics
    val_loss = sum(loss) / len(dataloader)
    auroc_dict, aupr_dict = auroc_aupr_scores(np.stack(list(auc_dict['gts'].values()), axis=1),
                                              np.stack(list(auc_dict['preds'].values()), axis=1),
                                              average_types=['macro', 'weighted'])
    # Print results
    pf = '%10.3g' * 5  # print format
    LOGGER.info(pf % (val_loss, auroc_dict['macro'], aupr_dict['macro'], auroc_dict['weighted'], aupr_dict['weighted']))

    model.float()  # for training

    return auroc_dict['macro'], aupr_dict['macro'], auroc_dict['weighted'], aupr_dict['weighted'], *(loss.cpu() / len(dataloader)).tolist()
