import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from utils import save_obj, load_obj
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
import pickle
import argparse
	
def load_model(device, path):
	model = models.alexnet(pretrained=False)
	model.classifier = nn.Sequential(
		nn.Linear(256 * 6 * 6, 4096),
		nn.ReLU(True),
		nn.Dropout(),
		nn.Linear(4096, 4096),
		nn.ReLU(True),
		nn.Dropout(),
		nn.Linear(4096, 200),
	)
	model.load_state_dict(torch.load(path))

	for param in model.parameters():
		param.requires_grad = False
	model = model.to(device)
	model.eval()
	return model

def gen_prototypes(class_id, output_dir):
	# model
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = load_model(device, "path/to/your/model.pth")
	image_size = 512
	patch_size = 32
	total_class = 200
	iteration = 10000
	pic_num = 50
	criterion = torch.nn.MultiMarginLoss(margin=10)
	
	output_dir_img = output_dir + 'fake_img/'
	output_dir_pkl = output_dir + 'fake_pkl/'
	
	if not os.path.exists(output_dir_img):
		os.mkdir(output_dir_img)
	if not os.path.exists(output_dir_pkl):
		os.mkdir(output_dir_pkl)
	target_class = class_id
	#print('target: ',target_class)
	subdir = str(target_class)+'/'
	if not os.path.exists(output_dir_img+subdir):
		os.mkdir(output_dir_img+subdir)
	if not os.path.exists(output_dir_pkl+subdir):
		os.mkdir(output_dir_pkl+subdir)
	
	for pic_id in range(pic_num):
		label = torch.LongTensor([target_class])
		label = Variable(label).to(device)
		thresholds = torch.Tensor(1).uniform_(0.55, 0.99).to(device)
		print('	generating the {}th pic with thresholds {}'.format(pic_id, thresholds.cpu().numpy()[0]))
		input = torch.Tensor(3, 512, 512).uniform_(0,1)
		input = transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])(input)
		
		input = input.to(device)
		input.requires_grad_(True)
		
		optimizer = torch.optim.Adam([{'params': input}], lr=0.001, weight_decay=1e-4, amsgrad=True)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[iteration//128, iteration//64, iteration//32, iteration//16, \
						iteration//8, iteration//4, iteration//2], gamma=0.5)
		flag = 0
		# training
		for j in range(0,iteration):
			tmp_input = transformation(input)
			logits = model(tmp_input.unsqueeze(0))
			res = F.softmax(logits, dim=1)
			prob = res[0][target_class]
			if prob >= thresholds[0]:
				print('success! prob:',prob)
				#save_obj(input.cpu().detach().numpy(), output_dir_pkl+subdir+str(pic_id)+'.pkl')
				tmp = (input*0.5) + 0.5
				tmp = transforms.ToPILImage()(tmp.cpu())
				tmp.save(output_dir_img+subdir+str(pic_id)+'.png', quality=100)
				flag = 1
				break
			loss = criterion(logits, label)#-logits[0][target_class]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			input.data.clamp_(-1,1)
			scheduler.step()
		if flag == 0:
			pic_id = pic_id - 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='generate prototypes')
	parser.add_argument('--class_id', default=None, help='generate prototypes for class x')
	parser.add_argument('--output_dir', default='./prototypes/', help='path to save prototypes')
	args = parser.parse_args()
	
	gen_prototypes(int(args.class_id), args.output_dir)
