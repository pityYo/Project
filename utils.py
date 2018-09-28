import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np


def np_to_pil(img):
	arr = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
	#arr = np.clip(arr*255, 0, 255).astype(np.uint8)
	#arr = arr[..., np.newaxis]
	#print(arr.shape)
	new_img = Image.fromarray(arr)
	#new_img.save('screen0.png')
	return new_img

#(210, 160, 3)
def preprocess(img):
	arr = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
	arr = arr[55:215,...]   #for Nature
	new_img = Image.fromarray(arr)
	
	processor = transforms.Compose([
		#transforms.Scale(size = (84, 110), interpolation = 1),
		transforms.Resize(size = (84, 84), interpolation = 1),#for Nature
	])
	new_img = processor(new_img)
	#print(new_img.shape)
	#new_img = new_img[:,18:102]
	#new_img = new_img.crop((0, 20, 84, 104))
	#new_img.save('screen.png')
	return new_img


class ReplayBuffer:
	def __init__(self, CAPACITY):
		self.capacity = CAPACITY
		self.memory = []

	def store(self, transition):
		self.memory.append(transition)
		if len(self.memory) > self.capacity:
			del self.memory[0]

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
