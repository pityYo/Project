import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
import gym
from utils import *
import torchvision
from model import *
from argparse import ArgumentParser
from gym import wrappers
from Logger import *
from Target import *
# args

parser = ArgumentParser()
parser.add_argument('--dir',dest = 'dir', default = '', help = "folder for saving data")
#parser.add_argument('--ep',dest = 'ep', default = 7800, type=int, help = "ep of parameter")

args = parser.parse_args()
#print('args :')
#print('train : %d, load : %s, test : %s' % (args.train, args.pretrain, args.test))

####################################
##	initial && declare something  ##
####################################

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# hyper parameter

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
# gym
env = gym.make('Tennis-v0')
#env = wrappers.Monitor(env, './fin', force=True)

N_ACTION = env.action_space.n
print(env.action_space.n, ' actions')

# CUDA?
USE_CUDA = torch.cuda.is_available()
print('CUDA :',USE_CUDA)
#if USE_CUDA:

Q_eval = DuelNet(N_ACTION).cuda()
Q_target = DuelNet(N_ACTION).cuda()
SKIP = 3
'''
if args.pretrain:
	Q_eval.load_state_dict(torch.load('Duel.pth'))
'''	

#Loss_function = nn.MSELoss().cuda()
	#Optim = torch.optim.Adam(Q_eval.parameters(), lr = LR)
#Optim = torch.optim.RMSprop(Q_eval.parameters(), lr = LR, alpha = 0.95, eps = 0.01, momentum = 0.0)

#Buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
#record = csv.writer(open("./Tennis.csv", 'w'))

logEP = Logger(args.dir, 'EP_return', 'csv', EP)
logACT = Logger(args.dir, 'Act_prob', 'csv', Act_prob)

def choose_action(obs, first_frame):
	x = Variable(torch.Tensor(obs).view(1,3,84,84)).cuda()
	Q = Q_eval(x).data.cpu()
	#print(Q.numpy()[0][0])
	
	if first_frame:
		for i in range(18):
			logACT.logkv('act_%d'%(i), Q.numpy()[0][i])
		logACT.write()
	
	return Q.max(1)[0].numpy(), Q.max(1)[1].numpy()
		
def test_model(net_ep):
	#print('test model')
	for s in range(1):
		obs = env.reset() #np
		R = 0
		done = False
		step = 0
		pre_obs = np.stack([preprocess(obs)]*3, axis = 0)
		
		player = 0
		comp = 0
		first_frame = False
		while not done:
			[max_Q], action = choose_action(pre_obs, first_frame)
			#print(max_Q)
			first_frame = False
			
			tr = 0
			obs_4 = []
			for sk in range(SKIP):
				_obs, r, done, info = env.step(action)
				tr += r
				_obs = preprocess(_obs)
				obs_4.append(_obs)
				if done:
					for i in range(SKIP - sk - 1):
						obs_4.append(_obs)
					break
			if tr < 0:
				comp += 1	
			elif tr > 0:
				player += 1
			#obs_list.append(_obs)
			
			R += tr
			nxt_obs = np.stack(obs_4, axis = 0)
			result = None
			step += 1
			pre_obs = nxt_obs
			
			if done:
				print('net_ep: %5d, step: %4d, return: %f, P %d : %d C' % (net_ep, step, R, player,comp))
				#record.writerow([R, step])
				'''
				logEP.logkv('ep', net_ep)
				logEP.logkv('return', R)
				logEP.logkv('step', step)
				logEP.logkv('player', player)
				logEP.logkv('computer', comp)
				logEP.write()
				'''
				break


Q_eval.load_state_dict(torch.load('Duel%d.pth'%(args.ep)))
test_model(args.ep)


'''
for i in range(1,40):
	Q_eval.load_state_dict(torch.load('Duel%d.pth'%(i*200)))
	test_model(i*200)
'''
