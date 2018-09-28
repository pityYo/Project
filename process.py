import os
from argparse import ArgumentParser
import numpy as np
import csv
import sys
#args
parser = ArgumentParser()
parser.add_argument('--file',dest = "file", default = None)
parser.add_argument('--dst', dest="tar", default = "./")
args = parser.parse_args()
print(os.path.exists(args.tar))

if args.file is not None:
	#print('haha')
	if os.path.exists(args.file):
		
		L = []
		Q = []
		norm = []
		str1 = "tensor("
		str2 = ", device='cuda:0')"
		
		with open(args.file, newline = '') as file:
			lines = csv.reader(line.replace('\0','') for line in file)
			for line in lines:
				#print(line)
				if len(line) == 4:
					id, loss, q, NORM = line
					L.append(float(loss))
					Q.append(float(q))
					#norm.append(NORM[7:-19])
				
			np.save('%s/loss.npy'%(args.tar), L)
			np.save('%s/maxQ.npy'%(args.tar), Q)
			#np.save('%s/norm.npy'%(args.tar), norm)
	
	else:
		print('no such file')
		sys.exit()
else:
	sys.exit()