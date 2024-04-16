import argparse
import os
from pathlib import Path
import yaml
import shutil
import fcntl
import torch

from utils import anom_utils
from utils.load_utils import get_models, get_loaders
from utils.general import increment_path, set_random_seed, LOGGER
from utils import ood_lib

def get_scores(in_test_loader, out_test_loader, model, clsfier):
    if args.ood in ("yolood", "energy", "msp", "logit") and args.arch == "yolo_ood":
        LOGGER.info("component type: {} | agg head method: {} |".format(args.ood_type, args.head_method))
        in_scores = ood_lib.get_yolood_scores(in_test_loader, model, args)
        out_scores = ood_lib.get_yolood_scores(out_test_loader, model, args)

    elif args.ood == "odin":
        LOGGER.info('temp: {} | noise: {} |'.format(args.T, args.noise))
        in_scores = ood_lib.get_odin_scores(in_test_loader, model, clsfier, args.method, args.T, args.noise)
        out_scores = ood_lib.get_odin_scores(out_test_loader, model, clsfier, args.method, args.T, args.noise)

    elif args.ood == "maha":
        sample_mean, precision = ood_lib.get_maha_data(model, args)

        pack = (sample_mean, precision)
        in_scores = ood_lib.get_mahalanobis_score(model, clsfier, in_test_loader, pack, args.noise, args.method, args)
        out_scores = ood_lib.get_mahalanobis_score(model, clsfier, out_test_loader, pack, args.noise, args.method, args)
    elif args.ood == 'yolo':
        in_scores = ood_lib.get_yolo_scores(in_test_loader, model, args)
        out_scores = ood_lib.get_yolo_scores(out_test_loader, model, args)
    else:
        in_scores = ood_lib.get_logits(in_test_loader, model, clsfier, args, name="in_test")
        out_scores = ood_lib.get_logits(out_test_loader, model, clsfier, args, name="out_test")

    return in_scores, out_scores


def evaluation(args):
    """
    main function - define data loaders,model and print the FPR,AUROC, AUPR of inscore and outscore
    """
    LOGGER.info(args)
    LOGGER.info('arch: {} | ind dataset: {} | ood dataset: {} | ood method: {} | aggregation method: {} |'
          .format(args.arch, args.ind_dataset, args.ood_dataset, args.ood, args.method))
    set_random_seed(0)

    with open(os.path.join(os.path.dirname(__file__), 'data', args.ind_dataset + '.yaml'), 'r') as f:
        dataset_attributes = yaml.safe_load(f)
        args.n_classes = dataset_attributes['nc']
        args.in_data_root = dataset_attributes['path']

    with open(os.path.join(os.path.dirname(__file__), 'data', args.ood_dataset + '.yaml'), 'r') as f:
        dataset_attributes = yaml.safe_load(f)
        args.out_data_root = dataset_attributes['path']

    args.mode = 'eval'
    args.save_dir = str(increment_path(Path(args.save_dir)))
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    with open(os.path.join(args.save_dir, 'eval_opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    train_opt_file = os.path.sep.join(args.load_model.split(os.path.sep) + ['opt.yaml'])
    if os.path.isfile(train_opt_file):
        shutil.copyfile(train_opt_file, os.path.join(args.save_dir, 'train_opt.yaml'))

    _, _, in_test_loader = get_loaders(args, dataset_root=args.in_data_root, splits_to_load=['test'])  # ind data
    _, _, out_test_loader = get_loaders(args, dataset_root=args.out_data_root, splits_to_load=['test'])  # ood data

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, classifier = get_models(args, in_test_loader.dataset.n_classes, device)

    in_scores, out_scores = get_scores(in_test_loader, out_test_loader, model, classifier)

    auroc, aupr, fpr, _ = anom_utils.get_and_print_results(in_scores, out_scores, args, plot=True, save=True)

    if args.res_file != '':
        Path(os.path.dirname(args.res_file)).mkdir(parents=True, exist_ok=True)  # make dir
        if 'train_cls' in args.load_model:
            values = [args.ind_dataset, args.ood_dataset, args.ood, args.method, args.load_model,
                      str(fpr*100), str(auroc*100), str(aupr*100)]
        elif 'train_yolood' in args.load_model:
            with open(os.path.join(args.save_dir, 'train_opt.yaml'), 'r') as f:
                train_opt = yaml.safe_load(f)
            values = [args.ind_dataset, args.ood_dataset, args.ood, args.method, args.ood_type, args.head_method, ' '.join(str(u) for u in args.use_heads), ' '.join(str(u) for u in train_opt['resp_cell_offset']),
                      args.load_model, str(fpr*100), str(auroc*100), str(aupr*100)]
        elif 'train_yolo' in args.load_model:
            values = [args.ind_dataset, args.ood_dataset, args.ood, args.method, args.ood_type, args.head_method,
                      args.load_model, str(fpr*100), str(auroc*100), str(aupr*100)]
        with open(args.res_file, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(','.join(values) + '\n')
            fcntl.flock(f, fcntl.LOCK_UN)

    LOGGER.info(f'saved to {args.save_dir}')

def main(args):
    evaluation(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')

    # ood options
    parser.add_argument('--ood', type=str, default='yolood', help='which methods to use',
                        choices=['yolood', 'odin', 'maha', 'logit', 'energy', 'msp', 'yolo'])
    parser.add_argument('--method', type=str, default='max', help='which method to use', choices=['max', 'sum'])

    # yolo-ood options
    parser.add_argument('--ood_type', type=str, default='obj*cls', help='which method to use', choices=['obj*cls', 'obj', 'cls'])
    parser.add_argument('--head_method', type=str, default='sum', help='which method to use', choices=['max', 'sum', 'multiply'])
    parser.add_argument('--use_heads', nargs='+', default=[0, 1, 2], type=int, help='')
    parser.add_argument('--sum_weights', nargs='+', default=[1., 1., 1.], help='sum weights')

    # maha options
    parser.add_argument('--maha_type', type=str, default='ensemble', help='Mahalanobis type', choices=['vanilla', 'ensemble'])

    # odin options
    parser.add_argument('--noise', type=float, default=0)  # maha option too
    parser.add_argument('--T', type=int, default=10)

    # dataset options
    parser.add_argument('--img_size', type=int, default=640, help='Size of evaluated images')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers')
    parser.add_argument('--ind_dataset', type=str, default='pascal_voc', choices=['objects365_in', 'pascal_voc', 'coco2017'])
    parser.add_argument('--ood_dataset', type=str, default='Objects365_OODv3', choices=['objects365_out', 'nus_wide_out'])
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # model options
    parser.add_argument('--arch', type=str, default='yolo_ood', help='Architecture to use', choices=['yolo_cls', 'yolo_ood', 'yolo'])
    parser.add_argument('--load_model', type=str, default="runs/train/exp29/", help='Path to load models')
    parser.add_argument('--hyp', type=str, default="data/hyps/hyp.OOD.yaml", help='Path to hyperparameter file')
    parser.add_argument('--cfg', type=str, default="models/yolov5s - backbone.yaml", help='Path to model config')
    parser.add_argument('--save_dir', type=str, default="runs/eval/exp", help='Path to save folder')

    parser.add_argument('--res_file', type=str, default="", help='Path to csv')

    args = parser.parse_args()
    main(args)