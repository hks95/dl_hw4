import numpy as np
import os
import pdb

class WSJ():
	""" Load the WSJ speech dataset
		
		Ensure WSJ_PATH is path to directory containing 
		all data files (.npy) provided on Kaggle.
		
		Example usage:
			loader = WSJ()
			trainX, trainY = loader.train
			assert(trainX.shape[0] == 24590)
			
	"""
  
	def __init__(self):
		self.dev_set = None
		self.train_set = None
		self.test_set = None
		self.path = os.getcwd() + '/data/'
  
	@property
	def dev(self):
		if self.dev_set is None:
			self.dev_set = load_raw(self.path, 'dev')
		return self.dev_set

	@property
	def train(self):
		if self.train_set is None:
			self.train_set = load_raw(self.path, 'train')
		return self.train_set
  
	@property
	def test(self):
		if self.test_set is None:
			self.test_set = (np.load(os.path.join(self.path, 'test.npy'), encoding='bytes'), None)
		return self.test_set
	
def load_raw(path, name):
	return (
		np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'), 
		np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes'),
		np.load(os.path.join(path, '{}_label_dict.npy'.format(name)))
	)