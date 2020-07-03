import json
import logging
import os
from datetime import datetime

import boxx
import rpctool
from tqdm import tqdm

levels = ['easy', 'medium', 'hard', 'averaged']


def get_cAcc(result, level):
    index = levels.index(level)
    return float(result.loc[index, 'cAcc'].strip('%'))


def check_best_result(output_folder, result, result_str, filename):
    current_cAcc = get_cAcc(result, 'averaged')
    best_path = os.path.join(output_folder, 'best_result.txt')
    if os.path.exists(best_path):
        with open(best_path) as f:
            best_cAcc = float(f.readline().strip())
        if current_cAcc >= best_cAcc:
            best_cAcc = current_cAcc
            with open(best_path, 'w') as f:
                f.write(str(best_cAcc) + '\n' + filename + '\n' + result_str)
    else:
        best_cAcc = current_cAcc
        with open(best_path, 'w') as f:
            f.write(str(current_cAcc) + '\n' + filename + '\n' + result_str)
    return best_cAcc


def rpc_evaluation(dataset, predictions, output_folder, generate_pseudo_labels=False, iteration=-1, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    pred_boxlists = []
    annotations = []
    correct = 0
    mae = 0  # mean absolute error
    has_density_map = predictions[0].has_field('density_map')
    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)

        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        bboxes = prediction.bbox.numpy()
        labels = prediction.get_field("labels").numpy()
        scores = prediction.get_field("scores").numpy()

        # -----------------------------------------------#
        # -----------------Pseudo Label------------------#
        # -----------------------------------------------#
        density = 0.0
        if has_density_map:
            ann = dataset.get_annotation(img_info['id'])
            density_map = prediction.get_field('density_map').numpy()
            density = density_map.sum()
            if round(density) == len(ann):
                correct += 1
            mae += abs(density - len(ann))
        if generate_pseudo_labels and has_density_map:
            image_result = {
                'bbox': [],
                'width': image_width,
                'height': image_height,
                'id': img_info['id'],
                'file_name': img_info['file_name'],
            }

            for i in range(len(prediction)):
                score = scores[i]
                box = bboxes[i]
                label = labels[i]
                if score > 0.95:
                    x, y, width, height = float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])
                    image_result['bbox'].append(
                        (int(label), x, y, width, height)
                    )
            if len(image_result['bbox']) >= 3 and len(image_result['bbox']) == round(density):
                annotations.append(image_result)
        # -----------------------------------------------#
        # -----------------------------------------------#
        # -----------------------------------------------#

        for i in range(len(prediction)):
            score = scores[i]
            box = bboxes[i]
            label = labels[i]

            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]

            pred_boxlists.append({
                "image_id": img_info['id'],
                "category_id": int(label),
                "bbox": [float(k) for k in [x, y, width, height]],
                "score": float(score),
            })

    if has_density_map:
        logger.info('Ratio: {}'.format(correct / len(predictions)))
        logger.info('MAE: {:.3f} '.format(mae / len(predictions)))

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if len(pred_boxlists) == 0:
        logger.info('Nothing detected.')
        with open(os.path.join(output_folder, 'result_{}.txt'.format(time_stamp)), 'w') as fid:
            fid.write('Nothing detected.')
        return 'Nothing detected.'

    if generate_pseudo_labels:
        logger.info('Pseudo-Labeling: {}'.format(len(annotations)))
        with open(os.path.join(output_folder, 'pseudo_labeling.json'), 'w') as fid:
            json.dump(annotations, fid)

    save_path = os.path.join(output_folder, 'bbox_results.json')
    with open(save_path, 'w') as fid:
        json.dump(pred_boxlists, fid)
    res_js = boxx.loadjson(save_path)
    ann_js = boxx.loadjson(dataset.ann_file)
    result = rpctool.evaluate(res_js, ann_js)
    logger.info(result)

    result_str = str(result)
    if iteration > 0:
        filename = os.path.join(output_folder, 'result_{:07d}.txt'.format(iteration))
    else:
        filename = os.path.join(output_folder, 'result_{}.txt'.format(time_stamp))

    if has_density_map:
        result_str += '\n' + 'Ratio: {}, '.format(correct / len(predictions)) + 'MAE: {:.3f} '.format(mae / len(predictions))
    with open(filename, 'w') as fid:
        fid.write(result_str)

    best_cAcc = check_best_result(output_folder, result, result_str, filename)
    logger.info('Best cAcc: {}%'.format(best_cAcc))
    metrics = {
        'cAcc': {
            'averaged': get_cAcc(result, 'averaged'),
            'hard': get_cAcc(result, 'hard'),
            'medium': get_cAcc(result, 'medium'),
            'easy': get_cAcc(result, 'easy'),
        }
    }
    if has_density_map:
        metrics.update({
            'Ratio': correct / len(predictions),
            'MAE': mae / len(predictions),
        })
    eval_result = dict(metrics=metrics)
    return eval_result
