import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random

class RamDuel(nn.Module):
	def __init__(self, N_ACTIONS):
		super(DuelNet, self).__init__()
		self.action_num = N_ACTIONS
		self.conv1 = nn.Linear(180, 256)
		self.conv2 = nn.Linear(256, 128)
		#self.conv3 = nn.Linear(128, 64)

		self.adv_fc1 = nn.Linear(128, 64)
		self.val_fc1 = nn.Linear(128, 64)

		self.adv_fc2 = nn.Linear(64, N_ACTIONS)
		self.val_fc2 = nn.Linear(64, 1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		#x = F.relu(self.conv3(x))
		##print(x.size()) # [1,64,7,7]
		x = x.view(x.size(0), -1)

		val = self.val_fc1(x)
		adv = self.adv_fc1(x)
		val = F.relu(val)
		adv = F.relu(adv)
		adv = self.adv_fc2(adv)
		#print('adv', adv)

		val = self.val_fc2(val).expand(x.size(0), self.action_num)
		#print(val)
		output = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_num)
		return output

class DuelNet(nn.Module):
	def __init__(self, N_ACTIONS):
		super(DuelNet, self).__init__()
		self.action_num = N_ACTIONS
		self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
		#init.kaiming_normal_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
		self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
		#init.kaiming_normal_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
		self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
		#init.kaiming_normal_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
		self.adv_fc1 = nn.Linear(7*7*64, 512)
		self.val_fc1 = nn.Linear(7*7*64, 512)

		self.adv_fc2 = nn.Linear(512, N_ACTIONS)
		self.val_fc2 = nn.Linear(512, 1)
		#self.initial()
		
	def initial(self):
		init.kaiming_normal(self.conv1.weight.data, a = 0, mode='fan_in')
		init.kaiming_normal(self.conv2.weight.data, a = 0, mode='fan_in')
		init.kaiming_normal(self.conv3.weight.data, a = 0, mode='fan_in')
		init.kaiming_normal(self.adv_fc1.weight.data, a = 0, mode='fan_in')
		init.kaiming_normal(self.val_fc1.weight.data, a = 0, mode='fan_in')
	
	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), 0.01, inplace = True)
		x = F.leaky_relu(self.conv2(x), 0.01, inplace = True)
		x = F.leaky_relu(self.conv3(x), 0.01, inplace = True)
		#print(x.size()) # [1,64,7,7]
		x = x.view(x.size(0), -1)

		val = self.val_fc1(x)
		adv = self.adv_fc1(x)
		val = F.leaky_relu(val, 0.01, inplace = True)
		adv = F.leaky_relu(adv, 0.01, inplace = True)
		adv = self.adv_fc2(adv)
		#print('adv', adv)

		val = self.val_fc2(val).expand(x.size(0), self.action_num)
		#print(val)
		output = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_num)
		return output

class ReplayBuffer():
	def __init__(self, buffer_size):
		self.buffer = []
		self.limit = buffer_size
		
	def store(self, s, a, r, _s):
		item = [s, a, r, _s]
		self.buffer.append(item)
		if len(self.buffer) > self.limit:
			del self.buffer[0]
		
	def size(self):
		return len(self.buffer)
	
	def sample(self, BATCH_SIZE):
		return random.sample(self.buffer, BATCH_SIZE)