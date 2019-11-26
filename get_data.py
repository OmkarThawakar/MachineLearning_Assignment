import util
import learning_rate
from common_math        import sign
from common_math        import sigmoid
from SVM                import SVM
from LogisticRegression import LogisticRegression
from NaiveBayes         import NaiveBayes

import tree_util
from data import Data
from tree import Tree
from tree import Node

from scipy.sparse import csr_matrix
import numpy as np
import math
from   random import randint

def read_libsvm(fname, num_features=0):
	data = []
	y = []
	row_ind = []
	col_ind = []
	with open(fname) as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			elements = line.split()
			y.append(int(elements[0]))
			for el in elements[1:]:
				row_ind.append(i)
				c, v = el.split(":")
				col_ind.append(int(c))
				data.append(float(v))
	if num_features == 0:
		num_features = max(col_ind) + 1
	X = csr_matrix((data, (row_ind, col_ind)), shape=(len(y), num_features))

	return X.toarray(), np.array(y), num_features

epoch_cv   = 20
epoch_test = 20
limit_min  = 50 
limit_max  = 80

np.random.seed(231)
################################################################################
# Load CV Data
################################################################################
print("\nReading data ...")
DATA_DIR = 'data/data_semeion/folds/'
num_folds = 5

data_cv = []
label_cv = []
max_col_prior = 0

def get_data(data_type):
	# First get what is the maximum number of features across all folds
	fold_col_prior = 0
	for i in range(num_folds):
		_, _, col_prior = read_libsvm('data/data_{}/folds/fold{}'.format(data_type,i+1))
		if col_prior > fold_col_prior :
			fold_col_prior = col_prior
		else:
			pass

	for i in range(num_folds):
		data_fold, label_fold, max_col_prior = read_libsvm('data/data_{}/folds/fold{}'.format(data_type, i+1))
		data_cv.append (data_fold)
		label_cv.append(label_fold)


	################################################################################
	# Load Train and Test Data
	################################################################################
	DATA_DIR = 'data_previous/'
	data_tr, label_tr, train_col_prior = read_libsvm('data/data_{}/{}_data_train'.format(data_type,data_type))

	data_te, label_te, col_prior = read_libsvm('data/data_{}/{}_data_test'.format(data_type,data_type))
	

	################################################################################
	# Prepare validation splits
	################################################################################
	print("\nPreparing cross-validation splits ...")
	train_split_data  = []
	train_split_label = []
	test_split_data   = []
	test_split_label  = []

	# For each fold
	for j in range(num_folds):
		if(j==0):
			start = 1			
			train_data = data_cv[1]
			train_label = label_cv[1]

			test_data  = data_cv[0]
			test_label = label_cv[1]
		else:
			start = 0
			train_data = data_cv[0]
			train_label = label_cv[0]

			test_data  = data_cv[j]
			test_label = label_cv[j]

		# Train data and label
		for k in range(start+1,num_folds):
			if(k != j):		
				train_data  = np.concatenate([train_data,  data_cv[k]] , axis=0)
				train_label = np.concatenate([train_label, label_cv[k]], axis=0)

		train_split_data.append(train_data)
		train_split_label.append(train_label)

		test_split_data .append(test_data)
		test_split_label.append(test_label)

	if data_type == 'madelon':
		fold_col_prior = 256
	else:
		pass

	col_prior  = {'fold':fold_col_prior, 'train':train_col_prior}

	return train_split_data, train_split_label, test_split_data, test_split_label , data_tr, label_tr, data_te, label_te, col_prior  
