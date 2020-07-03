# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import glob
import os
import xlwt

import cv2
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import pickle
import numpy as np

def load_obj(filename):
	if not os.path.exists(filename):
		return None
	with open(filename, 'rb') as f:
		tensor = pickle.load(f)
	tensor = np.asarray(tensor).astype(np.float32)
	return tensor


def main():
	parser = argparse.ArgumentParser(description="DPNet Demo")
	parser.add_argument(
		"--config-file",
		default="configs/e2e_mask_rcnn_R_101_FPN_1x.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--images_dir",
		required=True,
		type=str,
		help="path to images file",
	)
	parser.add_argument(
		"--save_dir",
		default='rpc_results',
		type=str,
		help="path to images file",
	)
	parser.add_argument(
		"--confidence-threshold",
		type=float,
		default=0.7,
		help="Minimum score for the prediction to be shown",
	)
	parser.add_argument(
		"--min-image-size",
		type=int,
		default=800,
		help="Smallest size of the image to feed to the model. "
			 "Model was trained with 800, which gives best results",
	)
	parser.add_argument(
		"--show-mask-heatmaps",
		dest="show_mask_heatmaps",
		help="Show a heatmap probability for the top masks-per-dim masks",
		action="store_true",
	)
	parser.add_argument(
		"--masks-per-dim",
		type=int,
		default=2,
		help="Number of heatmaps per dimension to show",
	)
	parser.add_argument(
		"opts",
		help="Modify model config options using the command-line",
		default=None,
		nargs=argparse.REMAINDER,
	)

	args = parser.parse_args()

	# load config from file and command-line arguments
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()
#509 391 788 670
#564 391 
	# prepare object that handles inference plus adds predictions on top of image
	coco_demo = COCODemo(
		cfg,
		confidence_threshold=0.9,#args.confidence_threshold,
		show_mask_heatmaps=args.show_mask_heatmaps,
		masks_per_dim=args.masks_per_dim,
		min_image_size=args.min_image_size,
	)
	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)
	'''
	img = cv2.imread('test_img/112.jpg')
	composite = coco_demo.run_on_opencv_image(img, False)

	cv2.imwrite(os.path.join(args.save_dir, '111.png'), composite)
	'''
	#images = load_obj('../../test_img/imageres_ori_0.obj')
	ori_images = load_obj('test_img/imageres_ori_0.obj')
	#images = images * 255
	#images = images.astype(np.uint8)
	#images = images.transpose(0,2,3,1)
	ori_images = ori_images * 255
	ori_images = ori_images.astype(np.uint8)
	ori_images = ori_images.transpose(0, 2, 3, 1)
	'''
	for i, img in enumerate(images):
		print('fake_', i)
		composite = coco_demo.run_on_opencv_image(img,False)
		cv2.imwrite(os.path.join(args.save_dir, '0_' + str(i) + '.png'), composite)
	'''
	for i, img in enumerate(ori_images):
		print('ori_', i)
		composite = coco_demo.run_on_opencv_image(img,False)
		cv2.imwrite(os.path.join(args.save_dir, 'ori_0_' + str(i) + '.png'), composite)


if __name__ == "__main__":
	main()
