#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.nn import Parameter
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from PIL import Image
import math
from torchvision import models
from collections import OrderedDict
from collections import namedtuple
#from utils import OHE_labels
import pickle
import numpy as np

	
class RPC(torch.utils.data.Dataset): #�����Լ����ࣺCelebA,�̳�torch.utils.data.Dataset
	def __init__(self, labelset, img_path, centerset, bboxset, batch_size, pic_size): #��ʼ��һЩ��Ҫ����Ĳ���
		self.img_path = img_path
		self.labelset = labelset
		self.centerset = centerset
		self.bboxset = bboxset
		self.batch_size = batch_size
		self.pic_size = pic_size
		
	def __getitem__(self, index):	 #����Ҫ�У����ڰ���������ȡÿ��Ԫ�صľ������ݣ�ѵ��ʱ��ÿ��batch������
		image = Image.open(self.img_path[index]).convert('RGB')
		x,y = image.size
		x3,x4,x5,x6 = self.bboxset[index]
		x3 = int(x3)
		x4 = int(x4)
		x5 = int(x5) + 1
		x6 = int(x6) + 1
		i = 200
		image = image.crop((x3-i,x4-i,x3+x5+i,x4+x6+100))
		image = transforms.ToTensor()(image)
		new_y, new_x = image.shape[1:]
		pad = nn.ConstantPad2d((x3-i,x-new_x-x3+i,x4-i,y-new_y-x4+i),1)
		image = pad(image)
		image = transforms.ToPILImage()(image)
	
		w1, h1 = image.size
		ori_shape = image.size
		image = self.transform(image)
		_, w2, h2 = image.shape
		image = np.array(image)
		path = self.img_path[index]
		label = self.labelset[index]
		center = self.centerset[index]
		x, y = center
		x = int(x * w2 / w1)
		y = int(y * h2 / h1)
		center = [x, y]
		bbox = self.bboxset[index] #x,y,w,h
		
		x1, y1, w, h = bbox
		'''
		x1 = int(x1 * w2 / w1)
		y1 = int(y1 * h2 / h1)
		w = int(w * w2 / w1)
		h = int(h * h2 / h1)
		'''
		bbox = (x1,y1,w,h)
		
		return image, label, center, bbox, path, ori_shape
 
	def __len__(self):	#����Ҫд�����ص������ݼ��ĳ��ȣ�Ҳ���Ƕ�����ͼƬ��Ҫ��loader�ĳ���������
		return len(self.img_path)

	def transform(self, image):
		preprocess = transforms.Compose([
			transforms.Resize((self.pic_size,self.pic_size)),
			#transforms.CenterCrop(224),
			transforms.ToTensor(),
		])
		image = preprocess(image)
		return image

