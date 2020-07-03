# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import glob
import os
import xlwt

import cv2
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def get_pic_and_label(root,file_path):
	f = open(file_path, 'r')
	dic = {}
	for line in f:
		line = line.rstrip()
		words = line.split()
		path = words[0]
		label = int(words[1])
		dic[path] = label
	f.close()
	return dic

def write_excel(dic):
	wb = xlwt.Workbook()
	sheet = wb.add_sheet('sheet1')
	sheet.write(0, 0, 'name')
	sheet.write(0, 1, 'y')
	sheet.write(0, 2, 'real_label')
	sheet.write(0, 3, 'fake_label')
	sheet.write(0, 4, 'y_in_real_label')
	sheet.write(0, 5, 'y_in_fake_label')
	sheet.col(0).width = 256 * 35
	sheet.col(1).width = 256 * 10
	sheet.col(2).width = 256 * 35
	sheet.col(3).width = 256 * 35
	sheet.col(4).width = 256 * 15
	sheet.col(5).width = 256 * 15
	count = 1
	for name,info in dic.items():
		sheet.write(count, 0, name)
		sheet.write(count, 1, str(info['y']))
		sheet.write(count, 2, str(info['real_label']))
		sheet.write(count, 3, str(info['fake_label']))
		sheet.write(count, 4, str(info['y'] in info['real_label']))
		sheet.write(count, 5, str(info['y'] in info['fake_label']))
		count = count + 1
	wb.save('result.xls')

def main():
	parser = argparse.ArgumentParser(description="DPNet Demo")
	parser.add_argument(
		"--config-file",
		default="configs/predict.yaml",
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
		default=512,
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

	# prepare object that handles inference plus adds predictions on top of image
	coco_demo = COCODemo(
		cfg,
		confidence_threshold=0.7,#args.confidence_threshold,
		show_mask_heatmaps=args.show_mask_heatmaps,
		masks_per_dim=args.masks_per_dim,
		min_image_size=args.min_image_size,
	)
	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	image_paths = glob.glob(os.path.join(args.images_dir, '*.png'))

	test = False
	dic = {}
	pic_path = '/media/dsg3/datasets/rpc/train2019/'
	pic_info = "../../dataset.txt"
	name2label = get_pic_and_label(pic_path, pic_info)
	patched = True
	#patch = load_obj('/media/dsg3/caobowen/result/correct2/patch_pkl/134_.pkl')
	for image_path in tqdm(image_paths):
		img = cv2.imread(image_path)
		if img is None:
			print(image_path)
			continue
		if test:
			composite, labels = coco_demo.run_on_opencv_image(img,test)
		else:
			if patched:
				
				composite = coco_demo.run_on_opencv_image(img,test)
		name = image_path.split('/')[-1]
		if test:
			if name[0] != 'f':
				real_label = name2label[name]
				if not dic.__contains__(name):
					 dic[name] = {}
				dic[name]['y'] = real_label
				dic[name]['real_label'] = labels
			else:
				if not dic.__contains__(name[5:]):
					 dic[name[5:]] = {}
				dic[name[5:]]['fake_label'] = labels
		cv2.imwrite(os.path.join(args.save_dir, os.path.basename(image_path)), composite)
	#write_excel(dic)

if __name__ == "__main__":
	main()
