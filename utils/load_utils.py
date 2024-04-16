import os
from tqdm import tqdm
import pickle
import numpy as np
import cv2
import yaml
import random
import albumentations as A
import torch
from torch.utils import data

from models.yolo import Model
from models.classifiersimple import clssimp
from models.experimental import attempt_load
from utils.general import intersect_dicts, LOGGER
from utils.torch_utils import model_info
from utils.augmentations import letterbox, random_perspective

def get_models(args, num_classes, device):
    model, clsfier = None, None
    LOGGER.info('Model info:')
    if args.arch == "yolo_ood":
        model = attempt_load(os.path.join(args.load_model, 'weights', 'best.pt'),
                             map_location=device,
                             fuse=False)
        clsfier = None

    elif args.arch == "yolo_cls":
        model = Model(args.cfg)  # create
        clsfier = clssimp(num_classes=num_classes, ch=model.model[-1].cv1.conv.in_channels)
        if args.mode == 'train':
            csd = []
            if args.weights != '':
                csd = torch.load(args.weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
                csd = intersect_dicts(csd, model.state_dict())  # intersect
                model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items')
            model.train()
            clsfier.train()

        elif args.mode == 'eval':
            model.load_state_dict(torch.load(os.path.join(args.load_model, 'weights', 'backbone_best.pth')))
            clsfier.load_state_dict(torch.load(os.path.join(args.load_model, 'weights', 'classifier_best.pth')))
            model.eval()
            clsfier.eval()
        model_info(clsfier, name='Classifier')
        model.to(device)
        clsfier.to(device)

    elif args.arch == 'yolo':
        model = attempt_load(os.path.join(args.load_model, 'weights', 'best.pt'),
                             map_location=device,
                             fuse=False)
        clsfier = None

    return model, clsfier


class CustomDataset(data.Dataset):
    def __init__(self, root_path, hyp_path, split, augment, img_size, n_classes, extra_transform=None):
        self.n_classes = n_classes
        self.img_size = img_size

        images_path = os.path.join(root_path, 'images', split)
        self.img_list = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith(('jpg', 'png', 'JPEG', 'JPG'))]

        self.GT = None
        if split != 'test':
            if not os.path.isfile(os.path.join(root_path, 'labels', split + '_labels.pkl')):
                self.GT = [np.unique(np.loadtxt(os.path.splitext(img.replace('images', 'labels'))[0] + '.txt', usecols=0, dtype=np.long))
                           for img in tqdm(self.img_list, desc='Loading {} labels'.format(split))]
                with open(os.path.join(root_path, 'labels', split + '_labels.pkl'), 'wb') as f:  # Pickling
                    pickle.dump(self.GT, f)
            else:
                with open(os.path.join(root_path, 'labels', split + '_labels.pkl'), 'rb') as f:  # Pickling
                    self.GT = pickle.load(f)

        with open(hyp_path, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            self.hyp = hyp

        self.augment = augment
        self.transform = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0)])

        self.extra_transform = extra_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index].strip())
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA)

        hyp = self.hyp
        img, _, _ = letterbox(img, self.img_size, auto=False, scaleup=self.augment)
        if self.augment:
            img, _ = random_perspective(img,
                                        degrees=hyp['degrees'],
                                        translate=hyp['translate'],
                                        scale=hyp['scale'],
                                        shear=hyp['shear'],
                                        perspective=hyp['perspective'])
            img = self.transform(image=img)['image']
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        label = np.zeros(self.n_classes)
        if self.GT:
            label[self.GT[index]] = 1

        image = torch.from_numpy(img / 255).float()
        label = torch.from_numpy(label)

        if self.extra_transform:
            image = self.extra_transform(image)

        return image, label


def get_loaders(args, dataset_root, splits_to_load):
    train_loader, val_loader, test_loader = None, None, None
    LOGGER.info('Dataset info:')
    if 'train' in splits_to_load:
        train_data = CustomDataset(root_path=dataset_root,
                                   hyp_path=args.hyp,
                                   split='train',
                                   augment=True,
                                   img_size=args.img_size,
                                   n_classes=args.n_classes)

        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=False)
        LOGGER.info('Loaded train data with {} images'.format(len(train_data)))
    if 'val' in splits_to_load:
        val_data = CustomDataset(root_path=dataset_root,
                                 hyp_path=args.hyp,
                                 split='val',
                                 augment=False,
                                 img_size=args.img_size,
                                 n_classes=args.n_classes)
        val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False, pin_memory=False)
        LOGGER.info('Loaded validation data with {} images'.format(len(val_data)))
    if 'test' in splits_to_load:
        test_data = CustomDataset(root_path=dataset_root,
                                  hyp_path=args.hyp,
                                  split='test',
                                  augment=False,
                                  img_size=args.img_size,
                                  n_classes=args.n_classes)
        test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=False, pin_memory=False)
        LOGGER.info('Loaded test data with {} images'.format(len(test_data)))

    return train_loader, val_loader, test_loader
