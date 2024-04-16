import os
from utils.general import cv2
from utils.plots import plot_results

LOGGERS = ('csv', 'tb', 'wandb')  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv('RANK', -1))

class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            'train/obj_loss',
            'train/cls_loss',  # train loss
            'metrics/AUROC_macro',
            'metrics/mAP_macro',
            'metrics/AUROC_weighted',
            'metrics/mAP_weighted',
            'val/obj_loss',
            'val/cls_loss',  # val loss
        ]  # params
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

