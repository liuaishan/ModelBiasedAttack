# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import sys
import numpy as np
from torchvision import transforms as T
from torch.autograd import Variable

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

class COCODemo(object):
	# RPC categories for pretty print
	CATEGORIES = ['__background__', '1_puffed_food', '2_puffed_food', '3_puffed_food', '4_puffed_food', '5_puffed_food', '6_puffed_food', '7_puffed_food',
				  '8_puffed_food', '9_puffed_food', '10_puffed_food', '11_puffed_food', '12_puffed_food', '13_dried_fruit', '14_dried_fruit', '15_dried_fruit',
				  '16_dried_fruit', '17_dried_fruit', '18_dried_fruit', '19_dried_fruit', '20_dried_fruit', '21_dried_fruit', '22_dried_food', '23_dried_food',
				  '24_dried_food', '25_dried_food', '26_dried_food', '27_dried_food', '28_dried_food', '29_dried_food', '30_dried_food', '31_instant_drink',
				  '32_instant_drink', '33_instant_drink', '34_instant_drink', '35_instant_drink', '36_instant_drink', '37_instant_drink', '38_instant_drink',
				  '39_instant_drink', '40_instant_drink', '41_instant_drink', '42_instant_noodles', '43_instant_noodles', '44_instant_noodles',
				  '45_instant_noodles', '46_instant_noodles', '47_instant_noodles', '48_instant_noodles', '49_instant_noodles', '50_instant_noodles',
				  '51_instant_noodles', '52_instant_noodles', '53_instant_noodles', '54_dessert', '55_dessert', '56_dessert', '57_dessert', '58_dessert',
				  '59_dessert', '60_dessert', '61_dessert', '62_dessert', '63_dessert', '64_dessert', '65_dessert', '66_dessert', '67_dessert', '68_dessert',
				  '69_dessert', '70_dessert', '71_drink', '72_drink', '73_drink', '74_drink', '75_drink', '76_drink', '77_drink', '78_drink', '79_alcohol',
				  '80_alcohol', '81_drink', '82_drink', '83_drink', '84_drink', '85_drink', '86_drink', '87_drink', '88_alcohol', '89_alcohol', '90_alcohol',
				  '91_alcohol', '92_alcohol', '93_alcohol', '94_alcohol', '95_alcohol', '96_alcohol', '97_milk', '98_milk', '99_milk', '100_milk', '101_milk',
				  '102_milk', '103_milk', '104_milk', '105_milk', '106_milk', '107_milk', '108_canned_food', '109_canned_food', '110_canned_food',
				  '111_canned_food', '112_canned_food', '113_canned_food', '114_canned_food', '115_canned_food', '116_canned_food', '117_canned_food',
				  '118_canned_food', '119_canned_food', '120_canned_food', '121_canned_food', '122_chocolate', '123_chocolate', '124_chocolate', '125_chocolate',
				  '126_chocolate', '127_chocolate', '128_chocolate', '129_chocolate', '130_chocolate', '131_chocolate', '132_chocolate', '133_chocolate', '134_gum',
				  '135_gum', '136_gum', '137_gum', '138_gum', '139_gum', '140_gum', '141_gum', '142_candy', '143_candy', '144_candy', '145_candy', '146_candy',
				  '147_candy', '148_candy', '149_candy', '150_candy', '151_candy', '152_seasoner', '153_seasoner', '154_seasoner', '155_seasoner', '156_seasoner',
				  '157_seasoner', '158_seasoner', '159_seasoner', '160_seasoner', '161_seasoner', '162_seasoner', '163_seasoner', '164_personal_hygiene',
				  '165_personal_hygiene', '166_personal_hygiene', '167_personal_hygiene', '168_personal_hygiene', '169_personal_hygiene', '170_personal_hygiene',
				  '171_personal_hygiene', '172_personal_hygiene', '173_personal_hygiene', '174_tissue', '175_tissue', '176_tissue', '177_tissue', '178_tissue',
				  '179_tissue', '180_tissue', '181_tissue', '182_tissue', '183_tissue', '184_tissue', '185_tissue', '186_tissue', '187_tissue', '188_tissue',
				  '189_tissue', '190_tissue', '191_tissue', '192_tissue', '193_tissue', '194_stationery', '195_stationery', '196_stationery', '197_stationery',
				  '198_stationery', '199_stationery', '200_stationery']

	def __init__(
			self,
			cfg,
			confidence_threshold=0.7,
			show_mask_heatmaps=False,
			masks_per_dim=2,
			min_image_size=224,
	):
		self.cfg = cfg.clone()
		self.model = build_detection_model(cfg)
		self.model.eval()
		self.device = torch.device('cuda:0')#(cfg.MODEL.DEVICE)
		self.model.to(self.device)#TODO:device
		self.min_image_size = min_image_size

		save_dir = cfg.OUTPUT_DIR
		checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
		_ = checkpointer.load(cfg.MODEL.WEIGHT)

		self.transforms = self.build_transform()

		mask_threshold = -1 if show_mask_heatmaps else 0.5
		self.masker = Masker(threshold=mask_threshold, padding=1)

		# used to make colors for each class
		self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

		self.cpu_device = torch.device("cpu")
		self.confidence_threshold = confidence_threshold
		self.show_mask_heatmaps = show_mask_heatmaps
		self.masks_per_dim = masks_per_dim

	def build_transform(self):
		"""
		Creates a basic transformation that was used to train the models
		"""
		cfg = self.cfg

		# we are loading images with OpenCV, so we don't need to convert them
		# to BGR, they are already! So all we need to do is to normalize
		# by 255 if we want to convert to BGR255 format, or flip the channels
		# if we want it to be in RGB in [0-1] range.
		if cfg.INPUT.TO_BGR255:
			#print("One")
			to_bgr_transform = T.Lambda(lambda x: x * 255)
		else:
			#print("two")
			to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

		normalize_transform = T.Normalize(
			mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
		)
		#print("normalize:{}".format(normalize_transform))
		transform = T.Compose(
			[
				#T.ToPILImage(),
				#T.Resize(self.min_image_size),
				#T.ToTensor(),
				to_bgr_transform,
				normalize_transform,
			]
		)
		return transform

	def run_on_opencv_image(self, image, y):
		"""
		Arguments:
			image (np.ndarray): an image as returned by OpenCV

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		predictions = self.compute_prediction(image)
		loss = self.get_loss(predictions, y, False)
		return loss
	
	def test(self, image, ori_label):
		"""
		Arguments:
			image (np.ndarray): an image as returned by OpenCV

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		predictions = self.compute_prediction(image)
		ori_correct, top_3 = self.get_loss(predictions, ori_label, True)
		return ori_correct, top_3
		
	def correct_num(self, image, y):
		"""
		Arguments:
			image (np.ndarray): an image as returned by OpenCV

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		predictions = self.compute_prediction(image)
		correct = []
		top_3 = []
		top_5 = []
		for i in range(len(predictions)):
			scores = predictions[i].get_field('scores')
			labels = predictions[i].get_field('labels')
			max_id = torch.argmax(scores)
			max_label = int(labels[max_id].numpy())
			
			index = torch.eq(labels, y[i])
			target_score = scores[index]
			if len(target_score) > 0:
				max_target = max(target_score)
			else:
				max_target = 0
			higher = scores[scores > max_target]
			if len(higher) <= 2:
				top_3.append(1)
			else :
				top_3.append(0)
			if len(higher) <= 4:
				top_5.append(1)
			else :
				top_5.append(0)	
			if (max_label == y[i]):
				correct.append(1)
			else:
				correct.append(0)
		return correct, top_3, top_5
		
	def get_loss(self, predictions, ori_label, test):
		if test:
			ori_correct = 0
			top_3 = 0
			for i in range(len(predictions)):
				scores = predictions[i].get_field('scores')
				labels = predictions[i].get_field('labels')
				max_id = torch.argmax(scores)
				max_label = int(labels[max_id].numpy())

				index = torch.eq(labels, ori_label[i])
				target_score = scores[index]
				if len(target_score) > 0:
					max_target = max(target_score)
				else:
					max_target = 0
				higher = scores[scores > max_target]
				if len(higher) <= 2:
					top_3 += 1
				if max_label == ori_label[i]:
					ori_correct += 1
			return ori_correct, top_3

		else:
			loss = torch.Tensor(np.array([0]))
			for i in range(len(predictions)):
				scores = predictions[i].get_field('scores')
				labels = predictions[i].get_field('labels')
				idx = torch.eq(labels, ori_label[i])
				target_scores = scores[idx]
				other_scores = scores[~idx]
				if len(other_scores) > 0:
					other_max = other_scores[torch.argmax(other_scores)]
				else:
					other_max = 0
				if len(target_scores) > 0:
					loss += -torch.sum(target_scores) + other_max
				else:
					loss += torch.sum(other_scores)
				#idx = torch.eq(labels, y[i])
				#target_scores = scores[idx]
				#other_scores = scores[~idx]
				#loss += torch.sum(other_scores[other_scores>0.6]) - torch.sum(target_scores)
			return loss

	def get_all_scores(self, predictions):
		scores = predictions[0].get_field('scores')
		labels = predictions[0].get_field('labels')
		labels = torch.LongTensor(labels)
		labels = labels - 1
		scores = torch.zeros(200).scatter_(0,labels,scores)
		output = scores.unsqueeze(0)
		for i in range(1, len(predictions)):
			scores = predictions[i].get_field('scores')
			labels = predictions[i].get_field('labels')
			labels = torch.LongTensor(labels)
			labels = labels - 1
			scores = torch.zeros(200).scatter_(0,labels,scores)
			scores = scores.unsqueeze(0)
			output = torch.cat([output, scores],0)
		return output

	def get_scores(self, all_predictions):
		"""
		Select only predictions which have a `score` > self.confidence_threshold,
		and returns the predictions in descending order of score

		Arguments:
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `scores`.

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		predictions = all_predictions[0]
		scores = predictions.get_field("scores")
		#print(scores)
		keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
		predictions = predictions[keep]
		scores = predictions.get_field("scores")
		_, idx = scores.sort(0, descending=True)
		top_predictions = predictions[idx]
		scores = top_predictions.get_field('scores')
		#print('scores1='+str(scores))
		labels = top_predictions.get_field('labels').long()
		scores = torch.mean(scores)
		#print('scores2='+str(scores))
		#print(labels)
		if torch.isnan(scores):
			all_scores = torch.Tensor([0])
		else:
			all_scores = scores.unsqueeze(0)
		real_labels = [labels.tolist()]
		labels = labels.unsqueeze(0)
		all_labels = torch.zeros(1,201).scatter_(1,labels,1)
		#print(real_labels)
		#print(type(scores))
		#all_scores = Variable(scores)
		#print(scores)
		#print(type(scores))
		#print(all_scores.size())
		#print(all_scores)
		
		for i in range(1,len(all_predictions)):
			predictions = all_predictions[i]
			scores = predictions.get_field("scores")
			keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
			predictions = predictions[keep]
			scores = predictions.get_field("scores")
			_, idx = scores.sort(0, descending=True)
			top_predictions = predictions[idx]
			scores = top_predictions.get_field('scores')
			#print('scores3='+str(scores))
			labels = top_predictions.get_field('labels').long()
			#print(labels)
			temp = torch.mean(scores)
			if torch.isnan(temp):
				temp = torch.Tensor([0])
			else:
				temp = temp.unsqueeze(0)
			#print('scores4='+str(temp))
			#print('scores5='+str(temp))
			labels = labels.unsqueeze(0)
			#print(labels.tolist())
			real_labels.append(labels.tolist())
			labels = torch.zeros(1,201).scatter_(1,labels,1)
			#print(all_scores)
			#print(temp)
			all_labels = torch.cat([all_labels,labels],0)
			all_scores = torch.cat([all_scores,temp],0)
			#print('all_scores1='+str(all_scores))
		return all_scores, all_labels, real_labels

	def compute_prediction(self, original_image):
		"""
		Arguments:
			original_image (np.ndarray): an image as returned by OpenCV

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		# apply pre-processing to image
		#print(len(original_image))
		image = [self.transforms(original_image[i]) for i in range(len(original_image))]
		'''
		with open('test.txt','a') as f:
			np.set_printoptions(threshold='nan')
			i = torch.LongTensor([[1,1],[1,2],[2,1],[2,2]])
			print(image[0].shape)
			a = image[0][0]
			b = image[0][1]
			c = image[0][2]
			f.write(str(a))
			f.write(str(b))
			f.write(str(c))
		sys.exit(0)
		'''
		# convert to an ImageList, padded so that it is divisible by
		# cfg.DATALOADER.SIZE_DIVISIBILITY
		image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
		#print(image_list)
		image_list = image_list.to(self.device)
		# compute predictions
		predictions = self.model(image_list)
		predictions = [o.to(self.cpu_device) for o in predictions]
		#print("predictions: {}".format(predictions))
		# always single image is passed at a time
		#prediction = predictions[0]

		# reshape prediction (a BoxList) into the original image size
		height = width = original_image[0].shape[-2]
		prediction = [predictions[i].resize((width, height)) for i in range(len(predictions))]

		return prediction

	def select_top_predictions(self, predictions):
		"""
		Select only predictions which have a `score` > self.confidence_threshold,
		and returns the predictions in descending order of score

		Arguments:
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `scores`.

		Returns:
			prediction (BoxList): the detected objects. Additional information
				of the detection properties can be found in the fields of
				the BoxList via `prediction.fields()`
		"""
		scores = predictions.get_field("scores")
		keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
		predictions = predictions[keep]
		scores = predictions.get_field("scores")
		_, idx = scores.sort(0, descending=True)
		top_predictions = predictions[idx]
		return_scores = top_predictions.get_field('scores')
		return_labels = top_predictions.get_field('labels')
		return_bboxes = top_predictions.bbox
		return top_predictions, return_scores, return_labels, return_bboxes

	def compute_colors_for_labels(self, labels):
		"""
		Simple function that adds fixed colors depending on the class
		"""
		colors = labels[:, None] * self.palette
		colors = (colors % 255).numpy().astype("uint8")
		return colors

	def overlay_boxes(self, image, predictions):
		"""
		Adds the predicted boxes on top of the image

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `labels`.
		"""
		labels = predictions.get_field("labels")
		#print(labels)
		boxes = predictions.bbox

		colors = self.compute_colors_for_labels(labels).tolist()

		for box, color in zip(boxes, colors):
			box = box.to(torch.int64)
			top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
			image = cv2.rectangle(
				image, tuple(top_left), tuple(bottom_right), tuple(color), 2
			)

		return image

	def overlay_mask(self, image, predictions):
		"""
		Adds the instances contours for each predicted object.
		Each label has a different color.

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `mask` and `labels`.
		"""
		masks = predictions.get_field("mask").numpy()
		labels = predictions.get_field("labels")

		colors = self.compute_colors_for_labels(labels).tolist()

		for mask, color in zip(masks, colors):
			thresh = mask[0, :, :, None]
			contours, hierarchy = cv2_util.findContours(
				thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
			)
			image = cv2.drawContours(image, contours, -1, color, 3)

		composite = image

		return composite

	def overlay_keypoints(self, image, predictions):
		keypoints = predictions.get_field("keypoints")
		kps = keypoints.keypoints
		scores = keypoints.get_field("logits")
		kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
		for region in kps:
			image = vis_keypoints(image, region.transpose((1, 0)))
		return image

	def create_mask_montage(self, image, predictions):
		"""
		Create a montage showing the probability heatmaps for each one one of the
		detected objects

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `mask`.
		"""
		masks = predictions.get_field("mask")
		masks_per_dim = self.masks_per_dim
		masks = L.interpolate(
			masks.float(), scale_factor=1 / masks_per_dim
		).byte()
		height, width = masks.shape[-2:]
		max_masks = masks_per_dim ** 2
		masks = masks[:max_masks]
		# handle case where we have less detections than max_masks
		if len(masks) < max_masks:
			masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
			masks_padded[: len(masks)] = masks
			masks = masks_padded
		masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
		result = torch.zeros(
			(masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
		)
		for y in range(masks_per_dim):
			start_y = y * height
			end_y = (y + 1) * height
			for x in range(masks_per_dim):
				start_x = x * width
				end_x = (x + 1) * width
				result[start_y:end_y, start_x:end_x] = masks[y, x]
		return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

	def overlay_class_names(self, image, predictions):
		"""
		Adds detected class names and scores in the positions defined by the
		top-left corner of the predicted bounding box

		Arguments:
			image (np.ndarray): an image as returned by OpenCV
			predictions (BoxList): the result of the computation by the model.
				It should contain the field `scores` and `labels`.
		"""
		scores = predictions.get_field("scores").tolist()
		labels = predictions.get_field("labels").tolist()
		labels = [self.CATEGORIES[i] for i in labels]
		boxes = predictions.bbox

		template = "{}: {:.2f}"
		for box, score, label in zip(boxes, scores, labels):
			x, y = box[:2]
			s = template.format(label, score)
			cv2.putText(
				image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
			)

		return image


import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
	"""Visualizes keypoints (adapted from vis_one_image).
	kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
	"""
	dataset_keypoints = PersonKeypoints.NAMES
	kp_lines = PersonKeypoints.CONNECTIONS

	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
	colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

	# Perform the drawing on a copy of the image, to allow for blending.
	kp_mask = np.copy(img)

	# Draw mid shoulder / mid hip first for better visualization.
	mid_shoulder = (
						   kps[:2, dataset_keypoints.index('right_shoulder')] +
						   kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
	sc_mid_shoulder = np.minimum(
		kps[2, dataset_keypoints.index('right_shoulder')],
		kps[2, dataset_keypoints.index('left_shoulder')])
	mid_hip = (
					  kps[:2, dataset_keypoints.index('right_hip')] +
					  kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
	sc_mid_hip = np.minimum(
		kps[2, dataset_keypoints.index('right_hip')],
		kps[2, dataset_keypoints.index('left_hip')])
	nose_idx = dataset_keypoints.index('nose')
	if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
		cv2.line(
			kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
			color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
	if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
		cv2.line(
			kp_mask, tuple(mid_shoulder), tuple(mid_hip),
			color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

	# Draw the keypoints.
	for l in range(len(kp_lines)):
		i1 = kp_lines[l][0]
		i2 = kp_lines[l][1]
		p1 = kps[0, i1], kps[1, i1]
		p2 = kps[0, i2], kps[1, i2]
		if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
			cv2.line(
				kp_mask, p1, p2,
				color=colors[l], thickness=2, lineType=cv2.LINE_AA)
		if kps[2, i1] > kp_thresh:
			cv2.circle(
				kp_mask, p1,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
		if kps[2, i2] > kp_thresh:
			cv2.circle(
				kp_mask, p2,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

	# Blend the keypoints.
	return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
