import glob
import json
import os
import random
import scipy
import time
from argparse import ArgumentParser
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

NUM_CATEGORIES = 200
GENERATED_NUM = 100000


def buy_strategic(counter):
    categories = [i + 1 for i in range(NUM_CATEGORIES)]
    selected_categories = np.random.choice(categories, size=random.randint(3, 10), replace=False)
    num_categories = len(selected_categories)

    if 3 <= num_categories < 5:  # Easy mode: 3∼5
        num_instances = random.randint(num_categories, 10)
        counter['easy_mode'] += 1
    elif 5 <= num_categories < 8:  # Medium mode: 5∼8
        num_instances = random.randint(10, 15)
        counter['medium_mode'] += 1
    else:  # Hard mode: 8∼10
        num_instances = random.randint(15, 20)
        counter['hard_mode'] += 1

    num_per_category = {}
    generated = 0
    for i, category in enumerate(selected_categories):
        i += 1
        if i == num_categories:
            count = num_instances - generated
        else:
            count = random.randint(1, num_instances - (num_categories - i) - generated)
        generated += count
        num_per_category[int(category)] = count

    return num_per_category


def check_iou(annotations, box, threshold=0.5):
    """
    Args:
        annotations:
        box: (x, y, w, h)
        threshold:
    Returns: bool
    """

    cx1, cy1, cw, ch = box
    cx2, cy2 = cx1 + cw, cy1 + ch
    carea = cw * ch
    for ann in annotations:
        x1, y1, w, h = ann['bbox']
        x2, y2 = x1 + w, y1 + h
        area = w * h
        inter_x1 = max(x1, cx1)
        inter_y1 = max(y1, cy1)
        inter_x2 = min(x2, cx2)
        inter_y2 = min(y2, cy2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        iou = inter_area / (carea + area - inter_area + 1e-8)  # avoid division by zero
        if iou > threshold:
            return False
    return True


def sample_select_object_index(category, paths, ratio_annotations, threshold=0.45):
    high_threshold_paths = [path for path in paths if ratio_annotations[os.path.basename(path)] > threshold]
    index = random.randint(0, len(high_threshold_paths) - 1)
    path = high_threshold_paths[index]
    return path


def generated_position(width, height, w, h, pad=0):
    x = random.randint(pad, width - w - pad)
    y = random.randint(pad, height - h - pad)
    return x, y


def get_object_bbox(annotation):
    bbox = annotation['bbox']
    x, y, w, h = [int(x) for x in bbox]

    box_pad = max(160, int(max(w, h) * 0.3))
    crop_x1 = max(0, x - box_pad)
    crop_y1 = max(0, y - box_pad)
    x = x - crop_x1
    y = y - crop_y1
    return x, y, w, h


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # (x,y)
    leaf_size = 2048
    # build kd tree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leaf_size)
    # query kd tree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.085
            sigma = min(sigma, 999)  # avoid inf
        else:
            raise NotImplementedError('should not be here!!')
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def synthesize(strategics, save_json_file='', output_dir='', save_mask=False):
    with open('ratio_annotations.json') as fid:
        ratio_annotations = json.load(fid)

    with open('instances_train2019.json') as fid:
        data = json.load(fid)
    images = {}
    for x in data['images']:
        images[x['id']] = x

    annotations = {}
    for x in data['annotations']:
        annotations[images[x['image_id']]['file_name']] = x

    object_paths = glob.glob(os.path.join('/data7/lufficc/process_rpc/cropped_train2019/', '*.jpg'))

    object_category_paths = defaultdict(list)
    for path in object_paths:
        name = os.path.basename(path)
        category = annotations[name]['category_id']
        object_category_paths[category].append(path)
    object_category_paths = dict(object_category_paths)

    bg_img_cv = cv2.imread('bg.jpg')
    bg_height, bg_width = bg_img_cv.shape[:2]
    mask_img_cv = np.zeros((bg_height, bg_width), dtype=np.uint8)

    json_ann = []
    for image_id, num_per_category in tqdm(strategics):
        bg_img = Image.fromarray(bg_img_cv)
        mask_img = Image.fromarray(mask_img_cv)
        synthesize_annotations = []
        for category, count in num_per_category.items():
            category = int(category)
            for _ in range(count):
                paths = object_category_paths[category]

                object_path = sample_select_object_index(category, paths, ratio_annotations, threshold=0.45)

                name = os.path.basename(object_path)
                mask_path = os.path.join('/data7/lufficc/rpc/object_masks/', '{}.png'.format(name.split('.')[0]))

                obj = Image.open(object_path)
                mask = Image.open(mask_path).convert('L')

                # dense object bbox
                # ---------------------------
                # Crop according to json annotation
                # ---------------------------
                x, y, w, h = get_object_bbox(annotations[name])
                obj = obj.crop((x, y, x + w, y + h))
                mask = mask.crop((x, y, x + w, y + h))

                # ---------------------------
                # Random scale
                # ---------------------------
                scale = random.uniform(0.4, 0.7)
                w, h = int(w * scale), int(h * scale)
                obj = obj.resize((w, h), resample=Image.BILINEAR)
                mask = mask.resize((w, h), resample=Image.BILINEAR)

                # ---------------------------
                # Random rotate
                # ---------------------------
                angle = random.random() * 360
                obj = obj.rotate(angle, resample=Image.BILINEAR, expand=1)
                mask = mask.rotate(angle, resample=Image.BILINEAR, expand=1)

                # ---------------------------
                # Crop according to mask
                # ---------------------------
                where = np.where(np.array(mask))
                y1, x1 = np.amin(where, axis=1)
                y2, x2 = np.amax(where, axis=1)
                obj = obj.crop((x1, y1, x2, y2))
                mask = mask.crop((x1, y1, x2, y2))
                w, h = obj.width, obj.height

                pad = 2
                pos_x, pos_y = generated_position(bg_width, bg_height, w, h, pad)
                start = time.time()
                threshold = 0.5
                while not check_iou(synthesize_annotations, box=(pos_x, pos_y, w, h), threshold=threshold):
                    if (time.time() - start) > 3:  # cannot find a valid position in 3 seconds
                        start = time.time()
                        threshold += 0.1
                        continue
                    pos_x, pos_y = generated_position(bg_width, bg_height, w, h, pad)

                bg_img.paste(obj, box=(pos_x, pos_y), mask=mask)
                if save_mask:
                    mask_img.paste(mask, box=(pos_x, pos_y), mask=mask)

                # ---------------------------
                # Find center of mass
                # ---------------------------
                mask_array = np.array(mask)
                center_of_mass = ndimage.measurements.center_of_mass(mask_array)  # y, x
                center_of_mass = [int(round(x)) for x in center_of_mass]
                center_of_mass = center_of_mass[1] + pos_x, center_of_mass[0] + pos_y  # map to whole image

                synthesize_annotations.append({
                    'bbox': (pos_x, pos_y, w, h),
                    'category_id': category,
                    'center_of_mass': center_of_mass,
                })

        assert bg_height == 1815 and bg_width == 1815
        scale = 200.0 / 1815
        gt = np.zeros((round(bg_height * scale), round(bg_width * scale)))
        for item in synthesize_annotations:
            center_of_mass = item['center_of_mass']
            gt[round(center_of_mass[1] * scale), round(center_of_mass[0] * scale)] = 1

        assert gt.shape[0] == 200 and gt.shape[1] == 200

        density = gaussian_filter_density(gt)
        image_name = '{}.jpg'.format(image_id)

        bg_img.save(os.path.join(output_dir, image_name))
        np.save(os.path.join(output_dir, 'density_maps', image_id), density)

        # plt.subplot(121)
        # plt.imshow(density, cmap='gray')
        #
        # plt.subplot(122)
        # plt.imshow(bg_img)
        #
        # print(len(synthesize_annotations))
        # print(density.sum())
        # plt.show()
        # quit()

        if save_mask:
            mask_img.save(os.path.join(output_dir, 'masks', image_name))
        json_ann.append({
            'image_id': image_name,
            'objects': synthesize_annotations
        })
    if save_json_file:
        with open(save_json_file, 'w') as fid:
            json.dump(json_ann, fid)


if __name__ == '__main__':
    parser = ArgumentParser(description="Synthesize fake images")
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # counter = {
    #     'easy_mode': 0,
    #     'medium_mode': 0,
    #     'hard_mode': 0
    # }
    # strategics = []
    # for image_id in tqdm(range(GENERATED_NUM)):
    #     num_per_category = buy_strategic(counter)
    #     strategics.append(('synthesized_image_{}'.format(image_id), num_per_category))
    #
    # if os.path.exists('strategics.json'):
    #     os.remove('strategics.json')
    # with open('strategics.json', 'w') as f:
    #     json.dump(strategics, f)
    # print(counter)  # {'easy_mode': 25078, 'medium_mode': 37287, 'hard_mode': 37635}
    # quit()

    with open('strategics.json') as f:
        strategics = json.load(f)
    strategics = sorted(strategics, key=lambda s: s[0])
    version = 'density_map_0_45_threshold'

    output_dir = os.path.join('synthesize_{}'.format(version))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'density_maps')):
        os.mkdir(os.path.join(output_dir, 'density_maps'))

    threads = []
    num_threads = args.count
    sub_strategics = strategics[args.local_rank::num_threads]
    save_file = 'synthesize_{}_{}.json'.format(version, args.local_rank)
    synthesize(sub_strategics, save_file, output_dir)
