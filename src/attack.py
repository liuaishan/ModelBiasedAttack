import torch
import torchvision
import sys
import cv2
from torchvision import transforms
import argparse

import os
import math
import numpy as np
from PIL import Image
from scipy import misc
import pickle
import pylab as pl
from target_model.maskrcnn_benchmark.config import cfg
from target_model.demo.predictor import COCODemo as Demo
import random
import numpy as np

#get logits and adv_image
def attack(id, patch, image, model, position, cuda=False, output_dir=None):
	#device = torch.device('cuda:0') if cuda else torch.device('cpu')
	cpu_device = torch.device('cpu')

	def pad_transform(patch, image_w, image_h, offset_x, offset_y):
		patch_y, patch_x = patch.shape[:2]
		padded_patch = np.pad(patch, ((offset_x - patch_x // 2, image_w - patch_x - offset_x + patch_x // 2), \
				(offset_y - patch_y // 2, image_h - patch_y - offset_y + patch_y // 2), (0, 0)), 'constant',constant_values = (0,0)) #left, right, top ,bottom
		mask = np.ones(patch.shape)
		padded_mask = np.pad(mask, ((offset_x - patch_x // 2, image_w - patch_x - offset_x + patch_x // 2), \
				(offset_y - patch_y // 2, image_h - patch_y - offset_y + patch_y // 2), (0, 0)), 'constant',constant_values = (0,0))
		return padded_patch, padded_mask
	
	offset_x, offset_y = position
	image_w, image_h = image.shape[:2]
	padded_patch, padded_mask = pad_transform(patch, image_w, image_h, offset_x, offset_y)
	patch, mask = padded_patch, padded_mask

	adv_image = np.multiply((1 - mask), image) + np.multiply(mask, patch)
	ori_image = image
	adv_image = np.array(adv_image, dtype=image.dtype)
	
	with torch.no_grad():
		ori_composite = model.run_on_opencv_image(image)
		adv_composite = model.run_on_opencv_image(adv_image)
	#print("FINISHED")	
	if output_dir is not None:
		cv2.imwrite(os.path.join(output_dir, str(id) + '.png'), ori_composite)
		cv2.imwrite(os.path.join(output_dir, str(id) + '.png'), adv_composite)
	#merge = np.hstack((ori_composite, adv_composite))
	#cv2.imshow("result", merge)
	#cv2.waitKey(0)

def main():
	parser = argparse.ArgumentParser(description="Demo")
	parser.add_argument(
		"--config-file",
		default="target_model/configs/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--image_path",
		required=True,
		type=str,
		help="path to image",
	)
	parser.add_argument(
		"--patch_path",
		required=True,
		type=str,
		help="path to patch",
	)
	parser.add_argument(
		"--x",
		default=None,
		help="x",
	)
	parser.add_argument(
		"--y",
		default=None,
		help="y",
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		help="path to save results",
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
	
	#build_target_model
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()
	model = Demo(
		cfg,
		confidence_threshold=args.confidence_threshold,
		show_mask_heatmaps=args.show_mask_heatmaps,
		masks_per_dim=args.masks_per_dim,
		min_image_size=args.min_image_size,
	)
	
	#patch
	patch = cv2.imread(args.patch_path)
	
	#image_dir
	pics = os.listdir(args.image_path)
	for pic in pics:
		#image
		image = cv2.imread(os.path.join(args.image_path, pic))
		#image = cv2.resize(image, (512, 512))
		#position
		x, y = image.shape[:2]
		if args.x is not None and args.y is not None:
			position = (args.x, args.y)
		else:
			position = (x // 2, y // 2)
		#begin attack
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)
		attack(pic.split('.')[0], patch, image, model, position, cuda=True, output_dir=args.output_dir)
	
	
if __name__=='__main__':
	main()
