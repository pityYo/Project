from collections import defaultdict
import os
import csv

class Logger(object):
	def __init__(self, dir, filename, filetype, items):
	
		self.log = defaultdict(float)
		self.dir = dir
		self.mkdir(dir)
		self.type = filetype
		self.item = items
		self.writer = self.make_writer(filename, filetype)
		self.CSVtitle(items)
		
		
	def mkdir(self, dir):
		if dir is None or dir is '':
			print('***   need folder for saving data   ***')
			
		folder = os.path.exists(dir)
		if not folder:
			os.makedirs(dir)
			print('make folder %s done ...' % (dir))
		
		else:
			print('folder %s has existed ...'%(dir) )
			
	def logkv(self, key, val):
		self.log[key] = val
		
	def CSVtitle(self, items):
		self.writer.writerow(items)
	
	def make_writer(self, filename, filetype):
		if filetype is 'csv' or filetype is 'CSV':
			return csv.writer(open(self.dir + '/' + filename + '.csv', 'w', newline = ''))
			
		else:
			print('no support %s ...'%(filetype))
			print('CSV for alternative file type ...')
			self.type = 'csv'
			return csv.writer(open(self.dir + '/' + filename + '.csv', 'w', newline = ''))
	def write(self):
		if self.type is 'csv' or self.type is 'CSV':
			self.writeCSV()
		
		else:
			self.writeCSV()
		
	def writeCSV(self):
		buf = []
		for item in self.item:
			buf.append(self.log[item])
		self.writer.writerow(buf)
		self.log.clear()
		