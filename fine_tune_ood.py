import argparse

from pathlib import Path
import os
import yaml
import torch
import numpy as np

from utils.general import increment_path, set_random_seed, LOGGER
from utils.load_utils import get_loaders, get_models
from utils import anom_utils
from utils import ood_lib
import fcntl


def get_scores(loader, model, classifier, args, pack, T, noise):
    if args.ood == "odin":
        scores = ood_lib.get_odin_scores(loader, model, classifier, args.method, T, noise)

    elif args.ood == "maha":
        scores = ood_lib.get_mahalanobis_score(model, classifier, loader, pack, noise, args.method, args)
    return scores


def get_score_gaussian_noise(args, model, classifier, pack, T, noise):
    dummy_targets = -torch.ones((args.ood_num_examples, args.n_classes))
    ood_data = torch.normal(mean=0.5, std=1, size=(args.ood_num_examples, 3, args.img_size, args.img_size)).clamp_(0, 1)
    ood_dataset = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    out_scores = get_scores(ood_loader, model, classifier, args, pack, T, noise)
    return out_scores


def get_score_uniform_noise(args, model, classifier, pack, T, noise):
    dummy_targets = -torch.ones((args.ood_num_examples, args.n_classes))
    ood_data = torch.FloatTensor(size=(args.ood_num_examples, 3, args.img_size, args.img_size)).uniform_(0, 1)
    ood_dataset = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    out_scores = get_scores(ood_loader, model, classifier, args, pack, T, noise)
    return out_scores


def arithmetic_mean(val_loader, args, model, classifier, pack, T, noise):
    class AvgOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

        def __len__(self):
            return len(self.dataset)

    ood_dataset = AvgOfPair(val_loader.dataset)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    out_scores = get_scores(ood_loader, model, classifier, args, pack, T, noise)
    return out_scores


def geometric_mean(val_loader, args, model, classifier, pack, T, noise):
    class GeomMeanOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0]), 0

        def __len__(self):
            return len(self.dataset)

    ood_dataset = GeomMeanOfPair(val_loader.dataset)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    out_scores = get_scores(ood_loader, model, classifier, args, pack, T, noise)
    return out_scores


def jigsaw(val_loader, args, model, classifier, pack, T, noise):
    img_size = args.img_size

    class ShufflePatches(object):
        def __init__(self, patch_size):
            self.ps = patch_size

        def __call__(self, x):
            x = x.unsqueeze(0)
            # divide the batch of images into non-overlapping patches
            u = torch.nn.functional.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
            # permute the patches of each image in the batch
            pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
            # fold the permuted patches back together
            f = torch.nn.functional.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
            f = f.squeeze(0)
            return f

    ood_loader = val_loader
    ood_loader.dataset.extra_transform = ShufflePatches(patch_size=img_size//16)
    out_scores = get_scores(ood_loader, model, classifier, args, pack, T, noise)
    return out_scores


def tune(temp, noise, model, classifier, val_loader, pack):
    auroc_list = []
    aupr_list = []
    fpr_list = []

    LOGGER.info('Calculating in scores..')
    in_scores = get_scores(val_loader, model, classifier, args, pack, temp, noise)

    LOGGER.info('Arithmetic mean:')
    out_scores = arithmetic_mean(val_loader, args, model, classifier, pack, temp, noise)
    auroc, aupr, fpr, _ = anom_utils.get_and_print_results(in_scores, out_scores, args, plot=False, save=False)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    LOGGER.info('Geometric mean:')
    out_scores = geometric_mean(val_loader, args, model, classifier, pack, temp, noise)
    auroc, aupr, fpr, _ = anom_utils.get_and_print_results(in_scores, out_scores, args, plot=False, save=False)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    LOGGER.info('Patches shuffle:')
    out_scores = jigsaw(val_loader, args, model, classifier, pack, temp, noise)
    auroc, aupr, fpr, _ = anom_utils.get_and_print_results(in_scores, out_scores, args, plot=False, save=False)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    LOGGER.info('Average results:')
    anom_utils.print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), args)
    return np.mean(fpr_list), np.mean(auroc_list), np.mean(aupr_list)


def fine_tune():
    LOGGER.info(args)
    set_random_seed(0)

    with open(os.path.join(os.path.dirname(__file__), 'data', args.ind_dataset + '.yaml'), 'r') as f:
        dataset_attributes = yaml.safe_load(f)
        args.n_classes = dataset_attributes['nc']
        args.in_data_root = dataset_attributes['path']

    args.mode = 'eval'
    args.save_dir = str(increment_path(Path(args.save_dir)))
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.save_dir, 'eval_opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    _, in_val_loader, _ = get_loaders(args, dataset_root=args.in_data_root, splits_to_load=['val'])  # ind data

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, classifier = get_models(args, in_val_loader.dataset.n_classes, device)

    pack = ood_lib.get_maha_data(model, args) if args.ood == 'maha' else None

    print("T = " + str(args.T) + "\tnoise = " + str(args.noise))
    fpr, auroc, aupr = tune(args.T, args.noise, model, classifier, in_val_loader, pack)

    if args.res_file != '':
        Path(os.path.dirname(args.res_file)).mkdir(parents=True, exist_ok=True)  # make dir
        if 'odin' in args.ood:
            values = [args.ind_dataset, args.ood, args.method, args.load_model, str(args.noise), str(args.T), str(fpr*100), str(auroc*100), str(aupr*100)]
        elif 'maha' in args.ood:
            values = [args.ind_dataset, args.ood, args.method, args.maha_type, args.load_model, str(args.noise), str(args.T), str(fpr*100), str(auroc*100), str(aupr*100)]
        with open(args.res_file, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(','.join(values) + '\n')
            fcntl.flock(f, fcntl.LOCK_UN)

    LOGGER.info(f'saved to {args.save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')

    # ood options
    parser.add_argument('--ood', type=str, default='odin', help='which methods to use', choices=['odin', 'maha'])
    parser.add_argument('--method', type=str, default='max', help='which method to use', choices=['max', 'sum'])

    # maha options
    parser.add_argument('--maha_type', type=str, default='ensemble', help='Mahalanobis type', choices=['vanilla', 'ensemble'])
    parser.add_argument('--ood_num_examples', type=int, default=1000)

    # odin options
    parser.add_argument('--noise', type=float, default=0)  # maha/odin
    parser.add_argument('--T', type=int, default=1) # odin

    # dataset options
    parser.add_argument('--img_size', type=int, default=640, help='Size of evaluated images')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers')
    parser.add_argument('--ind_dataset', type=str, default='pascal_voc', choices=['pascal_voc', 'coco2017', 'objects365_in'])
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # model options
    parser.add_argument('--load_model', type=str, default="runs/train_cls/exp1", help='Path to load models')
    parser.add_argument('--hyp', type=str, default="data/hyps/hyp.OOD.yaml", help='Path to hyperparameter file')
    parser.add_argument('--cfg', type=str, default="models/yolov5s - backbone.yaml", help='Path to model config')
    parser.add_argument('--save_dir', type=str, default="runs/fine_tune/exp", help='Path to save folder')
    parser.add_argument('--res_file', type=str, default="", help='Path to csv')

    args = parser.parse_args()
    fine_tune()
