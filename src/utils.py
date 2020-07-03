import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
import numpy as np
import torch
import torchvision
from torchvision import transforms
import pickle
import os
import random
#from sklearn.preprocessing import OneHotEncoder
from scipy import misc
from functools import reduce
import math
import cv2
from PIL import Image
from skimage import transform
	
current_iteration = 0

# change list of labels to one hot encoder
# e.g. [0,1,2] --> [[1,0,0],[0,1,0],[0,0,1]]
#def OHE_labels(Y_tr, N_classes):
	#OHC = OneHotEncoder()
	#Y_ohc = OHC.fit(np.arange(N_classes).reshape(-1, 1))
	#Y_labels = Y_ohc.transform(Y_tr.reshape(-1, 1)).toarray()
	#return Y_labels


#feed PIL image
def resize_image_with_crop_or_pad(image, target_height, target_width):
	(x,y) = image.size
	#x,y = image.shape[0],image.shape[1]
	if x < target_height or y < target_height:
		w = max(0, (target_width - x) // 2)
		w_odd = (target_width - x) % 2
		h = max(0, (target_height - y) // 2)
		h_odd = (target_height - y) % 2
		pad = torchvision.transforms.Pad((w + w_odd, h + h_odd, w, h))
		image = pad(image)
	CenterCrop = torchvision.transforms.CenterCrop((target_height, target_width))
	image = CenterCrop(image)
	return image

def randomly_overlay(image, patch, if_random=False, center=False):
	
	image_size = image.shape[-1]
	patch_size = patch.shape[-1]
	# randomly overlay the image with patch
	patch_mask = torch.ones(3, patch_size, patch_size)

	if if_random==True:
		angle = np.random.uniform(low=-180.0, high=180.0)
		location_x = int(np.random.uniform(low=0, high=int(image_size)-patch_size))
		location_y = int(np.random.uniform(low=0, high=int(image_size)-patch_size))
	else:
		if center:
			angle = 0
			location_x = image_size // 2
			location_y = image_size // 5 * 3
		else:
			angle = 0
			location_x = 52
			location_y = 48

	# move the patch and mask to the sama location
	patch = pad_to_bounding_box(patch, location_y, location_x, int(image_size), int(image_size))
	patch_mask = pad_to_bounding_box(patch_mask, location_y, location_x, int(image_size), int(image_size))
	print(patch_mask.shape)
	print(patch.shape)
	print(image.shape)
	# overlay the image with patch
	image_with_patch = (1 - patch_mask) * image + patch
	return image_with_patch

#feed PIL image
def pad_to_bounding_box(image, offset_y, offset_x, target_y, target_x):
	tensor2PIL = transforms.ToPILImage()
	PIL2tensor = transforms.ToTensor()
	transform = torchvision.transforms.Pad((offset_x, offset_y, 0, 0),fill=0)#left,top,right,bottom
	image = tensor2PIL(image)
	image = transform(image)
	(x, y) = image.size
	if target_y > y:
		transform = torchvision.transforms.Pad((0, 0, 0, target_y - y))
		image = transform(image)
	if target_x > x:
		transform = torchvision.transforms.Pad((0, 0, target_x - x, 0))
		image = transform(image)
	image = PIL2tensor(image)
	return image
		
# ZhangAnlan 2018.5.3
# param@num number of image/patch to load
# param@data_dir directory of image/patch
# returnVal@ return a pair of list of image/patch and corresponding labels i.e return image, label
# extra=True --> need to generate extra data, otherwise only preprocess
# N_classes, n_each=, ang_range, shear_range, trans_range and randomize_Var are parameters needed to generate extra data

def transform(image, size):

	preprocess = transforms.Compose([
		#transforms.Resize((size,size)),
		#transforms.CenterCrop(224),
		transforms.ToTensor(),
	])
	image = preprocess(image)
	return image

def get_patch(path, size):
	patch = Image.open(path).convert('RGB')
	#patch.show()

	patch = transform(patch, size)

	return patch
	
def get_pic_info(root,file_path):
	labelset = []
	pathset = []
	centerset = []
	bboxset = []
	f = open(file_path, 'r')
	for line in f:
		line = line.rstrip()
		words = line.split()
		path = words[0]
		pathset.append(root + path)
		labelset.append(int(words[1]))
		centerset.append([float(words[2]),float(words[3])])
		bboxset.append([float(words[4]),float(words[5]),float(words[6]),float(words[7])])
	f.close()
	labelset = np.array(labelset)
	centerset = np.array(centerset)
	bboxset = np.array(bboxset)
	
	return labelset, pathset, centerset, bboxset


# TV, distance between a pixel and its adjacent 2 pixels.
# In order to make patch more 'smooth'
# warning: be ware of 'inf'
# not TESTED!
def TV(patch, patch_size, batch_size):
	if not batch_size == patch.shape[0]:
		return None

	# TV for single image
	'''
	def single_image_TV(patch, patch_size):
		result = tf.Variable(tf.zeros([1, patch_size - 1, 3]))
		slice_result = tf.assign(result, patch[0: 1, 1:, 0: 3])
		for iter in range(1, patch_size - 1):
			temp = tf.assign(result,tf.add(tf.subtract(patch[iter:iter + 1, 1:, 0: 3], patch[iter:iter + 1, 0:-1, 0: 3]),
										   tf.subtract(patch[iter:iter + 1, 0:-1, 0: 3],patch[iter + 1:iter + 2, 0:-1, 0: 3])))
			slice_result = tf.concat([slice_result, temp], 0)

			return slice_result
	'''
	#assuming patch is [size*size*3] here
	#unchecked
	def single_image_TV(patch, patch_size):
		result = numpy.zeros((1,patch_size - 1, 3))
		result[1] = [x for x in range(1,patch_size)]
		result[2] = [x for x in range(0,3)]
		slice_result = torch.from_numpy(result)
		for iter in range(1, patch_size - 1):
				sub1 = patch[iter:iter + 1, 1:, 0: 3] + patch[iter:iter + 1, 0:-1, 0: 3]
				sub2 = patch[iter:iter + 1, 0:-1, 0: 3] + patch[iter + 1:iter + 2, 0:-1, 0: 3]
				temp = sub1 + sub2
				slice_result = torch.cat([slice_result, temp],0)
				return slice_result

	batch_image = patch[0]
	batch_image = single_image_TV(batch_image, patch_size)
	batch_image = torch.unqueeze(batch_image, 0)
	#batch_image = tf.expand_dims(batch_image, 0)
	for iter in range(1, batch_size):
		temp = single_image_TV(patch[iter], patch_size)
		temp = torch.unsqueeze(temp, 0)
		#temp = tf.expand_dims(temp, 0)
		batch_image = torch.cat([batch_image, temp], 0)
		#batch_image = tf.concat([batch_image, temp], 0)
		
	return l2_loss(batch_image)
	#return torch.nn.l2_loss(batch_image)

def l2_loss(x):
	#x = np.asarray(x)
	x = x.detach().numpy()
	loss = (np.linalg.norm(x) ** 2) / 2
	return loss

# total variance loss
def tv_loss(image, tv_weight):
	# total variation denoising
	shape = tuple(image.get_shape().as_list())
	tv_y_size = _tensor_size(image[:,1:,:,:])
	tv_x_size = _tensor_size(image[:,:,1:,:])
	tv_loss = tv_weight * 2 * ((l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) / tv_y_size) +
			(l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) / tv_x_size))

	return tv_loss
	
#unchecked!!
def _tensor_size(tensor):
	from operator import mul
	return reduce(mul, (d.value for d in tensor.get_shape()), 1)

# save tensor
def save_obj(tensor, filename):
	tensor = np.asarray(tensor).astype(np.float32)
	# print(b.eval())
	serialized = pickle.dumps(tensor, protocol=0)
	with open(filename, 'wb') as f:
		f.write(serialized)

# save patches
def save_patches(patches, filename):
	num = int(math.sqrt(int(patches.shape[0])))
	for i in range(num):
		for j in range(num):
			temp = resize_image_with_crop_or_pad(patches[i*num+j], int(patches.shape[1])+2, int(patches.shape[1])+2)
			if not j:
				row = temp
			else:
				row = torch.cat([row, temp], 1)
		if not i:
			show_patch = row
		else:
			show_patch = torch.cat([show_patch, row], 0)
		del row
	plt.figure(figsize=(5,5))
	plt.imshow(_convert(show_patch.eval()))
	plt.axis('off')
	plt.savefig(filename, dpi=200)
	plt.close()


# load tensor
def load_obj(filename):
	if not os.path.exists(filename):
		return None
	with open(filename, 'rb') as f:
		tensor = pickle.load(f)
	tensor = np.asarray(tensor).astype(np.float32)
	tensor = torch.Tensor(tensor)
	return tensor

def _convert(image):
	return (image * 255.0).astype(np.uint8)
	#return ((image + 1.0) * 127.5).astype(np.uint8)

# show image
def show_image(image):
	plt.axis('off')
	plt.imshow(_convert(image), interpolation="nearest")
	plt.show()

# plot accrucy
def plot_acc(acc, filename):
	plt.plot(acc)
	plt.ylabel('Accrucy')
	plt.savefig(filename, dpi=200)
	plt.close()

# show image with patch and accuracy
def plot_images_and_acc(image, result, acc, num, filename):
	size = int(math.ceil(math.sqrt(num)))
	fig = plt.figure(figsize=(5,5))
	fig.suptitle('Accuracy of misclassification: %4.4f' % acc, verticalalignment='top')
	for i in range(size):
		for j in range(size):
			if(i*size+j < num):
				temp = image[i*size+j]
				p = fig.add_subplot(size,size,i*size+j+1)
				# p = plt.subplot(size,size,i*size+j)
				p.imshow(_convert(temp.eval()))
				p.axis('off')
				if(result[i*size+j]!=0):
					p.set_title("Wrong", fontsize=8)
				else:
					p.set_title("Right", fontsize=8)
	# plt.title('Accuracy of misclassification: %4.4f' % acc)
	fig.savefig(filename, dpi=200)
