import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn

from utils import save_obj
from utils import load_obj
from utils import get_pic_info
import torch.nn.functional as F
from dataloader_center import RPC
from torchvision import transforms, models
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

# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs
	
class Fushion(object):

	def __init__(self, batch_size=50, image_size=512, patch_size=32, epsilon=1,
				 channel=3, gamma=0.95, learning_rate=0.005, epoch=50, 
				 basic_path='../xuanmai.jpg', pic_path="path/to/rpc/train2019/",
				 output_dir="./result/"):

		# hyperparameter
		self.batch_size = batch_size
		self.image_size = image_size
		self.patch_size = patch_size
		self.basic_path = basic_path

		self.epsilon = epsilon
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.epoch = epoch
		
		self.pic_path = pic_path
		self.output_dir = output_dir
		self.pic_info = '../wrong_samples/134.txt'
		
		self.device = torch.device('cuda:0')
		self.config_file = './target_model/configs/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map.yaml'
		self.opts = []
		self.confidence_threshold = 0.7
		self.min_image_size = 800
		self.target_model = self.build_target_model()
		self.cpu_device = torch.device('cpu')
		self.title = ''
		self.x = 2592
		self.y = 1944
		
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
		resnet = models.resnet101(pretrained=False)
		class_num = 200 #假设要分类数目是200
		channel_in = resnet.fc.in_features#获取fc层的输入通道数
		#然后把resnet-101的fc层替换成class_num类别的fc层
		resnet.fc = nn.Linear(channel_in,class_num)
		resnet.load_state_dict(torch.load("path/to/resnet_model.pth"))

		for param in resnet.parameters():
			param.requires_grad = False
		resnet.eval()
		resnet = resnet.to(self.device)
		return resnet
		
	def transform(self, image):

		preprocess = transforms.Compose([
			transforms.Resize((1024,1024)),
			#transforms.CenterCrop(224),
			transforms.ToTensor(),
		])
		image = preprocess(image)
		return image
	
	def gram_matrix(self, input):
		b, c, d = input.shape
		features = input.view(b, c * d)
		G = torch.mm(features, features.t())
		return G
		
	# fuse
	def train_op(self):
		self.build_dir()

		basic = Image.open(self.basic_path, 'r')
		self.x, self.y = basic.size
		x3,x4,x5,x6 = 1192.58, 782.87, 206.76, 214.61
		x3 = int(x3)
		x4 = int(x4)
		x5 = int(x5) + 1
		x6 = int(x6) + 1
		i = 200
		image = basic.crop((x3-i,x4-i,x3+x5+i,x4+x6+100))
		new_x, new_y = image.size
		image = transforms.ToTensor()(image)
		x3 = (self.x - new_x) // 2
		x4 = (self.y - new_y) // 2
		pad = nn.ConstantPad2d((x3, x3, x4, x4), 1)
		basic = pad(image)
		preprocess = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((self.pic_size,self.pic_size)),
			transforms.ToTensor(),
		])
		basic = preprocess(basic)
		basic.requires_grad_(True)
		
		optimizer = torch.optim.Adam([{'params': basic}], lr=self.learning_rate, weight_decay=1e-4, amsgrad=True)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 8, 15, 25], gamma=self.gamma)
		
		counter = 0
		best_acc = float('inf')
		current_acc = 0.0

		labelset, img_path, centerset, bboxset = get_pic_info(self.pic_path, self.pic_info)
		length = len(labelset)
		if length == 0:
			return
		set = RPC(labelset, img_path, centerset, bboxset, self.batch_size, self.pic_size,trans=transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5]))
		dataset = torch.utils.data.DataLoader(dataset=set, batch_size=self.batch_size, shuffle=True)

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
				basic_uns = basic.unsqueeze(0).to(self.device)
				image = Variable(image).to(self.device)
				y = Variable(label)
					
				basic_logit = self.target_model(basic_uns)
				basic_logit = F.softmax(basic_logit, dim=1)[0]
				
				loss1 = torch.sum(torch.mul(basic_logit, torch.log(basic_logit)))
				
				extract_list = ["layer4"]
				extract_result = FeatureExtractor(self.target_model, extract_list)
				basic_feature = extract_result(basic_uns)[0]
				image_feature = extract_result(image)[0]
				basic_gram = self.gram_matrix(basic_feature[0])
				img_gram = self.gram_matrix(image_feature[0])
				loss3 = F.mse_loss(basic_gram, img_gram)
				for i in range(1, len(image)):
					img_gram = self.gram_matrix(image_feature[i])
					loss3 += F.mse_loss(basic_gram, img_gram)
				loss3 = torch.log(loss3)
				loss = self.epsilon * loss1 + loss3
				
				print('loss1:%.4f,loss2:%.4f\nloss:%.4f' % (loss1, loss3, loss))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				basic.data.clamp_(-1, 1)
				
				if counter % (length // (self.batch_size)) == 0:
					l = self.test_op(basic)
					pl.figure()
					x.append(counter)
					attack_loss.append(l)
					pl.plot(x,attack_loss)
					pl.savefig(self.output_dir + 'loss/' + self.title + '_loss.png')
					pl.close('all')
					
					t = time.time() - start_time
					print("Epoch: [%2d] [%4d/%4d] time: %2dh:%2dm:%2ds" % (epoch, id, batch_iteration, t//3600, t//60, t%60))
					print("loss: %.4f" % l)
					print("learning_rate: %.8f" % scheduler.get_lr()[0])

					self.save_pic(basic.cpu().detach(), counter)
					save_obj(basic.cpu().detach().numpy(), self.output_dir + 'patch_pkl/' + self.title + '_' + str(counter) + '.pkl')
					if l < best_acc:
						best_acc = l
					print("current loss: %.4f, best loss: %.4f" % (l, best_acc))
					print('***************************')
					
				counter += 1

	def test_op(self, basic):
		labelset, img_path, centerset, bboxset = get_pic_info(self.pic_path, self.pic_info)
		set = RPC(labelset, img_path, centerset, bboxset, self.batch_size, self.pic_size)
		dataset = torch.utils.data.DataLoader(dataset=set, batch_size=self.batch_size, shuffle=False)
		
		with torch.no_grad():
			basic_uns = basic.unsqueeze(0).to(self.device)
			basic_logit = self.target_model(basic_uns)
			basic_logit = F.softmax(basic_logit, dim=1)[0]
			
			loss = torch.sum(torch.mul(basic_logit, torch.log(basic_logit)))
			for id,(image, label, center, bbox, path, _) in enumerate(dataset):
				basic_uns = basic.unsqueeze(0).to(self.device)
				image = Variable(image).to(self.device)
				y = Variable(label)
				
				extract_list = ["layer4"]
				extract_result = FeatureExtractor(self.target_model, extract_list)
				basic_feature = extract_result(basic_uns)[0]
				image_feature = extract_result(image)[0]
				basic_gram = self.gram_matrix(basic_feature[0])
				img_gram = self.gram_matrix(image_feature[0])
				loss3 = F.mse_loss(basic_gram, img_gram)
				for i in range(1, len(image)):
					img_gram = self.gram_matrix(image_feature[i])
					loss3 += F.mse_loss(basic_gram, img_gram)
				loss3 = torch.log(loss3)
				loss += loss3
		return loss / len(labelset)
				
			
		
if __name__ == '__main__':
	a = Fushion()
	a.title = 'fuse'
	a.train_op()


