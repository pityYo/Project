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
# args

parser = ArgumentParser()
parser.add_argument('-t',dest = "train", type = int, default = 0)
parser.add_argument('-l', '--load', dest = "pretrain", default = False, action='store_true')
parser.add_argument('--test', dest = "test", default = False, action = 'store_true')
args = parser.parse_args()
print('args :')
print('train : %d, load : %s, test : %s' % (args.train, args.pretrain, args.test))

####################################
##	initial && declare something  ##
####################################

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# hyper parameter
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 65000
HISTORY_FRAME_LEN = 4
TARGET_UPDATE_ITER = 1e4
EPSILON = 0.5 if args.pretrain else 1
EPSILON_DECAY = 0.995
EPSILON_LIMIT = 0.01
GAMMA = 0.99
LR = 0.00025
STEP_LIMIT = 50000#2000
SKIP = 3
EPOCH = 20000
START_TRAIN = 500

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
if args.pretrain:
	Q_eval.load_state_dict(torch.load('Duel.pth'))
	#Q_target.load_state_dict(torch.load('Target.pth'))
	
	
	
Loss_function = nn.MSELoss().cuda()
	#Optim = torch.optim.Adam(Q_eval.parameters(), lr = LR)
Optim = torch.optim.RMSprop(Q_eval.parameters(), lr = LR, alpha = 0.95, eps = 0.01, momentum = 0.0)
'''
else:
	#print('No cuda')
	Q_eval = DuelNet(N_ACTION)
	Q_target = DuelNet(N_ACTION)
	Loss_function = nn.MSELoss()
	#Optim = torch.optim.Adam(Q_eval.parameters(), lr = LR)
	Optim = torch.optim.RMSprop(Q_eval.parameters(), lr = LR, alpha = 0.9)
'''
Buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
record = csv.writer(open("./Tennis.csv", 'w'))
iter_record = csv.writer(open("./Tennis_LQ.csv","w"))


def check_serve(action, serve_time):
	fire = [1,10,11,12,13,14,15,16,17]
	if serve_time < 0:
		return -1
	elif action in fire:
		return 2
	else:
		return 1 if serve_time < 10 else 3

def choose_action(obs):
	global EPSILON, EPSILON_DECAY, EPSILON_LIMIT
	
	if Buffer.size() < 45000 and not args.test and not args.pretrain:
		#print('random')
		return np.array([-100]), np.array([np.random.randint(N_ACTION)])
	
	elif random.random() < EPSILON and not args.test:
		#print('random')
		EPSILON = max(EPSILON_LIMIT, EPSILON*EPSILON_DECAY)
		return np.array([-100]), np.array([np.random.randint(N_ACTION)])
	
	else:
		#print('haha')
		#x = Variable(obs).cuda()
		#if USE_CUDA:
		x = Variable(torch.Tensor(obs).view(1,3,84,84)).cuda()
		Q = Q_eval(x).data.cpu()
		return Q.max(1)[0].numpy(), Q.max(1)[1].numpy()
		'''
		else:
			x = Variable(torch.Tensor(obs).view(1,4,84,84))
			Q = Q_eval(x).data
			return Q.max(1)[0].numpy(),Q.max(1)[1].numpy()
			
			'''
def test_model(epoch):
	print('test model')
	for s in range(200):
		obs = env.reset() #np
		R = 0
		done = False
		step = 0
		pre_obs = np.stack([preprocess(obs)]*3, axis = 0)
		
		player = 0
		comp = 0
		
		while not done:
			[max_Q], action = choose_action(pre_obs)
			#print(max_Q)
			
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
				print('epoch: %5d, step: %4d, return: %f, P %d : %d C' % (epoch-200+s, step, R, player,comp))
				record.writerow([R, step])
				break
	
def train(epoch):
	if Buffer.size() < 45000:
		return
	else: 
		transition = Buffer.sample(BATCH_SIZE)
		#print('train')
	batch_s, batch_a, batch_r, batch_s_ = zip(*transition)
	batch_s = np.array(batch_s)
	batch_a = np.array(batch_a)
	batch_r = np.array(batch_r)
	batch_s_ = np.array(batch_s_)
	
	batch_s = Variable(torch.Tensor(batch_s)).cuda()
	batch_a = Variable(LongTensor(batch_a)).cuda()
	batch_r = Variable(FloatTensor(batch_r)).cuda()
	batch_s_ = Variable(torch.Tensor(batch_s_)).cuda()
	#
	Q_val = Q_eval(batch_s).gather(1,batch_a.view(-1,1))
	#max_Q = Q_val.item().max(1)[0]
	eval_a = Q_eval(batch_s_).data.max(1)[1]
	eval_a = Variable(eval_a.view(-1,1)).cuda()
	#print(eval_a.size())
	_Q = Q_target(batch_s_).gather(1,eval_a).detach()
	#print(batch_r.size())
	#print(_Q.size())
	y = batch_r + GAMMA * _Q
	#print(y.size())
	Loss = Loss_function(Q_val, y.view(-1,1))
	LD = Loss[0].data.cpu().numpy()
	Optim.zero_grad()
	
	Loss.backward()
	norm = nn.utils.clip_grad_norm(Q_eval.parameters(), max_norm = 10.0)
	Optim.step()

	#print(max_Q)
	#print(LD)
	return LD, norm

def check_end(player, comp):
	if player >= 4 and player-comp >= 2:
		return True
	elif comp >= 4 and comp - player >= 2:
		return True
	else:
		return False
	
def nice(step, p, c):
	if step > 700 or p+c > 10:
		return True
	

Return = []
REPLACE = 0

iter_n = 0

tot = 0
for epoch in range(1,args.train+1):
	obs = env.reset() #np
	R = 0
	done = False
	step = 0
	pre_obs = np.stack([preprocess(obs)]*3, axis = 0)
	player = 0
	comp = 0
	G = 1
	lc = 0
	lp = 0
	
	
	serve_time = 0
	non_serve = 0
	y_serve = 0
	serve_flag = 0
	#print('GG')
	while not done:
		
		[max_Q], action = choose_action(pre_obs)
		#print(max_Q)
		flag = check_serve(action, serve_time)
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
		
		if flag == 1:
			serve_time += 1
		elif flag == -1:
			serve_time = -1
		elif flag == 2:
			serve_flag = 1
			y_serve += 1*(0 if G % 2 == 0 else 1)
			serve_time = -1
		elif flag == 3:
			serve_flag = -1
			non_serve += 1*(0 if G % 2 == 0 else 1)
			serve_time = 0
		elif flag == -1:
			serve_time = -1
		
		if tr < 0:
			lc += 1
			comp += 1
			#G = 0
			serve_time = 0
			
		elif tr > 0:
			lp += 1
			player += 1
			#G = 0
			serve_time = 0
		
		if check_end(lp, lc):
			G += 1
			lp = 0
			lc = 0
		
		
		R += tr + 0.5*serve_flag*(0 if G % 2 == 0 else 1)
		nxt_obs = np.stack(obs_4, axis = 0)
		result = None
		if not args.test:
			#print('hi')
			Buffer.store(
					pre_obs.reshape(3,84,84), #np
					action,
					np.array([tr + serve_flag*0.5*(0 if G % 2 == 0 else 1)]),
					nxt_obs.reshape(3,84,84)
				)
			
			result = train(epoch)
		serve_flag = 0
		step += 1
		
		if result is not None and not args.test:
			LD , norm = result
			iter_record.writerow([iter_n, LD, max_Q, norm])
			
			#print('loss: %f, max_Q: %f'%(LD,max_Q), end = '\r')
			iter_n += 1
		
		pre_obs = nxt_obs
		#del obs_list[0]
		
		REPLACE += 1
		if REPLACE % TARGET_UPDATE_ITER == 0:
			Q_target.load_state_dict(Q_eval.state_dict())
		if done:
			break
		'''
		if done or check_end(player, comp):
			break
		'''
	#tot += R
	print('epoch %4d, step: %d,R: %.2f, %d:%d, serve: %d,%d' % (epoch,step, R,player,comp,y_serve,non_serve))
	
	if epoch % 200 == 0 and not args.test:
		torch.save(Q_eval.state_dict(), 'Duel%d.pth'%(epoch))
		torch.save(Q_target.state_dict(), 'Target%d.pth'%(epoch))
		
	if epoch % 1000 == 0 and not args.test:
		torch.save(Q_eval.state_dict(), 'FinDuel%d.pth'%(epoch))
		torch.save(Q_target.state_dict(), 'FinTarget%d.pth'%(epoch))
	
#print('test 1000 episode, avg return : %f' % (tot/1000))
# 