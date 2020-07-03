import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn

from utils import save_obj
from utils import get_pic_info
import torch.nn.functional as F
from dataloader_crop import RPC
from torchvision import transforms
from PIL import Image
import sys
import os
import time
import numpy as np
import pickle
import pylab as pl
from maskrcnn_benchmark.config import cfg
from target_model.demo.multi_size_loss import COCODemo
import random

class FPA(object):

	def __init__(self, batch_size=10, image_size=512, patch_size=32,
				 channel=3, gamma=0.95, learning_rate=0.001, epoch=50,
				 pic_path="path/to/rpc/train2019/", output_dir="./result/"):

		# hyperparameter
		self.batch_size = batch_size
		self.image_size = image_size
		self.patch_size = patch_size

		self.gamma = gamma
		self.learning_rate = learning_rate
		self.epoch = epoch
		
		self.pic_path = pic_path
		self.output_dir = output_dir
		self.train_pic_info = "../class_info/1_train.txt"
		self.test_pic_info = "../class_info/1_test.txt"
		
		self.device = torch.device('cuda:0')
		self.config_file = './target_model/configs/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map.yaml'
		self.opts = []
		self.confidence_threshold = 0.7
		self.min_image_size = 800
		self.target_model = self.build_target_model()
		self.cpu_device = torch.device('cpu')
		self.title = ''
		
	def build_dir(self):
		path = self.output_dir
		if not os.path.exists(path):
			os.mkdir(path)
		dirs = ['patch', 'patch_pkl', 'loss', 'accuracy', 'prob', 'patched_pic']
		for dir in dirs:
			name = path + dir
			if not os.path.exists(name):
				os.mkdir(name)
				
	def build_target_model(self):
		cfg.merge_from_file(self.config_file)
		cfg.merge_from_list(self.opts)
		cfg.freeze()
		model = COCODemo(
			cfg,
			confidence_threshold=self.confidence_threshold,
			show_mask_heatmaps=False,
			masks_per_dim=2,
			min_image_size=self.min_image_size,
		)
		return model
		
	def transform(self, image):

		preprocess = transforms.Compose([
			transforms.Resize((1024,1024)),
			#transforms.CenterCrop(224),
			transforms.ToTensor(),
		])
		image = preprocess(image)
		return image

	def save_patched_pic(self, pro, id, path, images, class_id, single=False):
		transform = transforms.ToPILImage(mode='RGB')
		for index, img in enumerate(images):
			image = transform(img.cpu())
			name = path[index].split('/')[-1]
			image.save(self.output_dir + 'patched_pic/' + self.title + '_' + str(pro) + '.png',quality=100)
			if single:
				break
			
	def save_patch(self, patch, acc, counter, id):
		transform = transforms.Compose(
			[
				transforms.ToPILImage(mode='RGB'),
			]
		)
		patch = transform(patch)
		name = self.title + str(counter) + '_' + str(acc) + '.png'
		patch.save(self.output_dir + 'patch/' + str(name),quality=100,sub_sampling=0)
		
				
	def pad_transform(self, patch, image_size, patch_size, x, y):
		
		'''
		randx = (random.random()-0.5) * 32
		randy = (random.random()-0.5) * 32
		'''
		offset_x = x
		offset_y = y
		
		pad = nn.ConstantPad2d((offset_x - self.patch_size // 2, image_size- patch_size - offset_x + self.patch_size // 2, offset_y - self.patch_size // 2, image_size-patch_size-offset_y + self.patch_size // 2), 0) #left, right, top ,bottom
		mask = torch.ones((3, patch_size, patch_size))
		return pad(patch), pad(mask)
		
	# train adversarial patch
	def train_op(self):
		self.build_dir()
		#load fused prior
		patch_cpu = Image.open('../134_0_0.jpg', 'r')
		patch_cpu = patch_cpu.resize((512, 512))
		patch_cpu = patch_cpu.crop((0, 0, 32 , 32))
		patch_cpu = transforms.ToTensor()(patch_cpu)
		patch_cpu.requires_grad_(True)
		
		optimizer = torch.optim.Adam([{'params': patch_cpu}], lr=self.learning_rate, weight_decay=1e-4, amsgrad=True)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,8,15], gamma=self.gamma)
		
		counter = 0
		best_acc = 1.0
		current_acc = 0.0
		pic_info = "../new_class_info/train/correct_train.txt"

		labelset, img_path, centerset, bboxset = get_pic_info(self.pic_path, pic_info)
		length = len(labelset)
		if length == 0:
			return
		set = RPC(labelset, img_path, centerset, bboxset, self.batch_size, self.image_size)
		dataset = torch.utils.data.DataLoader(dataset=set, batch_size=self.batch_size, shuffle=False)

		start_time = time.time()
		x = []
		attack_loss = []
		x_acc = []
		ori_accuracy = []
		target_accuracy = []
		pro_mean = []
		top_5 = []
		for epoch in range(self.epoch):
			scheduler.step()
			batch_iteration = len(labelset) / self.batch_size
			for id,(image, label, center, bbox, path, _) in enumerate(dataset):
				
				all_offset = center
				
				real_image = Variable(image).to(self.device)
				y = Variable(label)
					
				offset_x, offset_y = all_offset[0][0], all_offset[1][0]
				
				all_patch, all_mask = self.pad_transform(patch_cpu, self.image_size, self.patch_size, offset_x, offset_y)
				all_patch = all_patch.unsqueeze(0)
				all_mask = all_mask.unsqueeze(0)
				for i in range (1, len(real_image)):
					offset_x, offset_y = all_offset[0][i], all_offset[1][i]
					patch, mask = self.pad_transform(patch_cpu, self.image_size, self.patch_size, offset_x, offset_y)
					patch = patch.unsqueeze(0)
					mask = mask.unsqueeze(0)
					all_patch = torch.cat([all_patch, patch], 0)
					all_mask = torch.cat([all_mask, mask], 0)
				
				patch, mask = all_patch.to(self.device), all_mask.to(self.device)
				
				
				adv_image = torch.mul((1 - mask), real_image) + torch.mul(mask, patch)

				adv = adv_image.to(self.cpu_device)
				loss = self.target_model.run_on_opencv_image(adv, y)
		
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				patch_cpu.data.clamp_(0, 1)
				
				if counter % (length // self.batch_size) == 0:
					pl.figure()
					x.append(counter)
					attack_loss.append(loss)
					pl.plot(x,attack_loss)
					pl.savefig(self.output_dir + 'loss/' + self.title + '_loss.png')
					pl.figure()
					
					t = time.time() - start_time
					print("Epoch: [%2d] [%4d/%4d] time: %2dh:%2dm:%2ds" % (epoch, id, batch_iteration, t//3600, t//60, t%60))
					print("[Validation].......")
					print("learning_rate: %.8f" % scheduler.get_lr()[0])

					errAE = loss

					ori_acc, top3, top5 = self.test_op(patch_cpu)
					
					self.save_patched_pic(ori_acc, counter, path, adv_image, 'all', True)
					
					x_acc.append(counter)

					ori_accuracy.append(ori_acc)
					pl.plot(x_acc, ori_accuracy)
					pl.savefig(self.output_dir + 'accuracy/' + self.title + '_top1.png')
					
					pl.figure()
					
					pro_mean.append(top3)
					pl.plot(x_acc, pro_mean)
					pl.savefig(self.output_dir + 'accuracy/' + self.title + '_top3.png')
					pl.close('all')
					
					pl.figure()
					
					top_5.append(top5)
					pl.plot(x_acc, top_5)
					pl.savefig(self.output_dir + 'accuracy/' + self.title + '_top5.png')
					
					print("top1: %4.4f" % ori_acc)
					print("top3: %4.4f" % top3)
					print("top5: %4.4f" % top5)
					if ori_acc < best_acc:
						best_acc = ori_acc
						self.save_patch(patch_cpu.cpu().detach(), ori_acc, counter, 'all')
						save_obj(patch_cpu.cpu().detach().numpy(), self.output_dir + 'patch_pkl/' + self.title + '.pkl')
					print("current acc: %.4f, best acc: %.4f" % (ori_acc, best_acc))
					print('***************************')			
				counter += 1
				
	# test Generator
	def test_op(self, patch_cpu):
		test_pic_info = "../new_class_info/test/correct_test.txt"
		labelset, img_path, centerset, bboxset = get_pic_info(self.pic_path, test_pic_info)
		length = len(labelset)
		if length == 0:
			return 0
		set = RPC(labelset, img_path, centerset, bboxset, self.batch_size, self.image_size)
		test_dataset = torch.utils.data.DataLoader(dataset=set, batch_size=self.batch_size, shuffle=True)
		#print("lr = " + str(learning_rate))
		
		ori_correct_count = 0
		target_correct_count = 0
		top3 = 0.0
		top5 = 0.0
		for id,(image, label, center, bbox, path, _) in enumerate(test_dataset):
			
			all_offset = center
			y = Variable(label)
			real_image = Variable(image).to(self.device)
			offset_x, offset_y = all_offset[0][0], all_offset[1][0]
			
			all_patch, all_mask = self.pad_transform(patch_cpu, self.image_size, self.patch_size, offset_x, offset_y)
			all_patch = all_patch.unsqueeze(0)
			all_mask = all_mask.unsqueeze(0)
			for i in range (1, len(real_image)):
				offset_x, offset_y = all_offset[0][i], all_offset[1][i]
				patch, mask = self.pad_transform(patch_cpu, self.image_size, self.patch_size, offset_x, offset_y)
				patch = patch.unsqueeze(0)
				mask = mask.unsqueeze(0)
				all_patch = torch.cat([all_patch, patch], 0)
				all_mask = torch.cat([all_mask, mask], 0)
			
			patch, mask = all_patch.to(self.device), all_mask.to(self.device)
			
			adv_image = torch.mul((1 - mask), real_image) + torch.mul(mask, patch)
			
			adv_image = adv_image.to(self.cpu_device)
			with torch.no_grad():
				ori_correct, top_3, top_5 = self.target_model.test(adv_image, y)

			top3 += top_3
			top5 += top_5
			ori_correct_count += ori_correct
		ori_accu = float(ori_correct_count) / float(len(labelset))
		top3_acc = top3 / float(len(labelset))
		
		return ori_accu, top3_acc, top5 / float(len(labelset))

if __name__ == '__main__':
	a = FPA()
	a.title = '134_0_0'
	a.train_op()


