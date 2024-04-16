import os
import shutil
import sys
from pathlib import Path
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def main(args):
    if args.dataset_name == 'pascal_voc':
        download_pascalvoc(args)
    elif args.dataset_name == 'coco2017':
        download_coco2017(args)
    elif args.dataset_name == 'objects365': # both objects365_in and objects365_out
        download_objects365(args)
    elif args.dataset_name == 'nus_wide_out':
        download_nus_wide_out(args)

def download_pascalvoc(args):
    import xml.etree.ElementTree as ET
    from tqdm import tqdm
    from utils.general import download

    data_dict = {
        'names': {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
                  8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                  15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
    }

    def convert_label(path, lb_path, year, image_id):
        def convert_box(size, box):
            dw, dh = 1. / size[0], 1. / size[1]
            x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
            return x * dw, y * dh, w * dw, h * dh

        in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
        out_file = open(lb_path, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        names = list(data_dict['names'].values())  # names list
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls in names and int(obj.find('difficult').text) != 1:
                xmlbox = obj.find('bndbox')
                bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                cls_id = names.index(cls)  # class id
                out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

    # Download
    dir = Path(ROOT / 'datasets' / 'pascal_voc')  # dataset root dir
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
    urls = [f'{url}VOCtrainval_11-May-2012.zip']
    download(urls, dir=dir, curl=True, threads=3)

    # Convert
    path = dir / 'VOCdevkit'
    for year, image_set in ('2012', 'train'), ('2012', 'val'):
        imgs_path = dir / 'images' / f'{image_set}'
        lbs_path = dir / 'labels' / f'{image_set}'
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)

        with open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
            image_ids = f.read().strip().split()
        for id in tqdm(image_ids, desc=f'{image_set}'):
            f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
            lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
            f.rename(imgs_path / f.name)  # move image
            convert_label(path, lb_path, year, id)  # convert labels to YOLO format

    # handle test images
    f = dir / 'download.tar'
    if not f.is_file():
        print(f'{f} does not exist, download test data tar file from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/')
        return

    print(f'Unzipping test data {f}')
    os.system(f'tar xf {f} --directory {dir}')
    test_path = dir / 'images' / 'test'
    test_path.mkdir(exist_ok=True, parents=True)
    with open(dir / 'test.list', 'r') as test_file:
        for file in tqdm(test_file.readlines(), desc='test'):
            if not (path / f'VOC2012/JPEGImages' / file.strip()).is_file():
                print(f'{file} not found in test data')
                continue
            shutil.copyfile(path / f'VOC2012/JPEGImages' / file.strip(), test_path / file.strip())

    if args.delete_temp:
        print('Removing temporary folder...')
        del_path = dir / 'VOCdevkit'
        os.system(f'rm -r {del_path}')

def download_coco2017(args):
    from utils.general import download
    # Download labels
    dir_tmp = ROOT / 'datasets' / 'coco'  # dataset root dir
    dir_final = ROOT / 'datasets' / 'coco2017'  # dataset root dir

    for p in 'images', 'labels':
        (dir_final / p).mkdir(parents=True, exist_ok=True)
        for q in 'train', 'val':
            (dir_final / p / q).mkdir(parents=True, exist_ok=True)
    (dir_final / 'images' / 'test').mkdir(parents=True, exist_ok=True)

    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
    urls = [url + 'coco2017labels.zip']  # labels
    download(urls, dir=dir_tmp.parent)

    for folder in ['train2017', 'val2017']:
        f = dir_tmp / 'labels' / folder
        f.rename(dir_final / 'labels' / folder.replace('2017', ''))

    # Download data
    urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
            'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
            'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
    download(urls, dir=dir_tmp, threads=3)

    for split in ['train', 'val', 'test']:
        f = dir_tmp / f'{split}2017'
        f.rename(dir_final / 'images' / split)

    # remove images that do not have labels
    for split in ['train', 'val']:
        labels_file_names = os.listdir(dir_final / 'labels' / split)
        for file in os.listdir(dir_final / 'images' / split):
            if file.replace('jpg', 'txt') not in labels_file_names:
                os.remove(dir_final / 'images' / split / file)
    if args.delete_temp:
        print('Removing temporary folder...')
        os.system(f'rm -r {dir_tmp}')

def download_objects365(args):
    from tqdm import tqdm
    from utils.general import Path, download

    # Make Directories
    dir_main = Path(ROOT / 'datasets' / 'objects365')  # dataset root dir
    dir_in = Path(ROOT / 'datasets' / 'objects365_in')
    dir_out = Path(ROOT / 'datasets' / 'objects365_out')
    for d in [dir_main, dir_in]:
        for p in 'images', 'labels':
            (d / p).mkdir(parents=True, exist_ok=True)
            for q in 'train', 'val':
                (d / p / q).mkdir(parents=True, exist_ok=True)
    (dir_in / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (dir_out / 'images' / 'test').mkdir(parents=True, exist_ok=True)

    in_train_files = open(dir_in / 'train.list', 'r').read().splitlines()
    in_val_files = open(dir_in / 'val.list', 'r').read().splitlines()
    in_test_files = open(dir_in / 'test.list', 'r').read().splitlines()
    out_test_files = open(dir_out / 'test.list', 'r').read().splitlines()

    # Train, Val Splits
    for split, patches in [('train', 50 + 1), ('val', 43 + 1)]:
    # for split, patches in [('train', 1)]:
        print(f"Processing {split} in {patches} patches ...")
        images, labels = dir_main / 'images' / split, dir_in / 'labels'

        # Download
        url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"
        if split == 'train':
            download([f'{url}zhiyuan_objv2_{split}.tar.gz'], dir=dir_main, delete=False)  # annotations json
            download([f'{url}patch{i}.tar.gz' for i in range(patches)], dir=images, curl=True, delete=False, threads=8)
        elif split == 'val':
            download([f'{url}zhiyuan_objv2_{split}.json'], dir=dir_main, delete=False)  # annotations json
            download([f'{url}images/v1/patch{i}.tar.gz' for i in range(15 + 1)], dir=images, curl=True, delete=False,
                     threads=8)
            download([f'{url}images/v2/patch{i}.tar.gz' for i in range(16, patches)], dir=images, curl=True,
                     delete=False, threads=8)

        # Move
        for f in tqdm(images.rglob('*.jpg'), desc=f'Moving {split} images'):
            if f.name in in_train_files:
                f.rename(dir_in / 'images'/ 'train' / f.name)
            elif f.name in in_val_files:
                f.rename(dir_in / 'images'/ 'val' / f.name)
            elif f.name in in_test_files:
                f.rename(dir_in / 'images'/ 'test' / f.name)
            elif f.name in out_test_files:
                f.rename(dir_out / 'images'/ 'test' / f.name)

    # Unzip labels
    labels_path = dir_in / 'objects365_in_labels.zip'
    print('Unzipping labels...')
    os.system(f'unzip {labels_path}')

    if args.delete_temp:
        print('Removing temporary folder...')
        os.system(f'rm -r {dir_main}')

def download_nus_wide_out(args):
    from utils.general import Path, download
    from tqdm import tqdm

    dir = Path(ROOT / 'datasets' / 'nus_wide')  # dataset root dir
    (dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)

    with open(dir / 'urls.txt', 'r') as f:
        urls = f.readlines()

    for pair in tqdm(urls, desc='Downloading images', total=len(urls)):
        save_name, url = pair.split()
        save_path = dir / 'images' / 'test' / save_name
        os.system(f'wget {url} -qO {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--dataset_name', type=str, choices=['pascal_voc', 'coco2017', 'objects365', 'nus_wide_out'])
    parser.add_argument('--delete_temp', action='store_true', default=True)
    args = parser.parse_args()
    main(args)