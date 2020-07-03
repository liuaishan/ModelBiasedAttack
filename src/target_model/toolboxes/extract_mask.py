import glob
import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def do_extract(path):
    annotation = annotations[os.path.basename(path)]
    bbox = annotation['bbox']
    x, y, w, h = [int(x) for x in bbox]
    img = cv2.imread(path)
    origin_height, origin_width = img.shape[:2]

    box_pad = 5
    crop_x1 = x - box_pad
    crop_y1 = y - box_pad
    crop_x2 = x + w + box_pad
    crop_y2 = y + h + box_pad

    x = x - crop_x1
    y = y - crop_y1

    origin_img = img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    img = cv2.bilateralFilter(img, 3, 75, 75)

    # -------------------------
    # edge detect
    # -------------------------
    edges = detector.detectEdges(np.float32(img) / 255)

    # -------------------------
    # edge process
    # -------------------------
    object_box_mask = np.zeros_like(edges, dtype=np.uint8)
    object_box_mask[y:y + h, x:x + w] = 1
    edges[(1 - object_box_mask) == 1] = 0
    edges[(edges < (edges.mean() * 0.5)) & (edges < 0.1)] = 0

    # -------------------------
    # erode and dilate
    # -------------------------
    filled = ndimage.binary_fill_holes(edges).astype(np.uint8)
    filled = cv2.erode(filled, np.ones((32, 32), np.uint8))
    filled = cv2.dilate(filled, np.ones((32, 32), np.uint8))
    filled = cv2.erode(filled, np.ones((8, 8), np.uint8))

    filled = cv2.medianBlur(filled, 17)
    save_image = np.zeros((origin_height, origin_width), np.uint8)
    save_image[crop_y1:crop_y2, crop_x1:crop_x2] = np.array(filled * 255, dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(path).split('.')[0] + '.png'), save_image)

    masked_img = origin_img * filled[:, :, None]
    compare_img = np.concatenate([origin_img, masked_img], axis=1)
    cv2.imwrite(os.path.join(compare_dir, os.path.basename(path)), compare_img)


def extract(paths):
    for path in tqdm(paths):
        do_extract(path)


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract masks")
    parser.add_argument('--ann_file', type=str, default='instances_train2019.json')
    parser.add_argument('--images_dir', type=str, default='train2019')
    parser.add_argument('--model_file', type=str, default='model.yml.gz')
    args = parser.parse_args()

    with open(args.ann_file) as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x
    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    output_dir = 'extracted_masks/masks'
    compare_dir = 'extracted_masks/masked_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)

    categories = [i + 1 for i in range(200)]
    paths = glob.glob(os.path.join(args.images_dir, '*.jpg'))
    detector = cv2.ximgproc.createStructuredEdgeDetection(args.model_file)
    extract(paths)
