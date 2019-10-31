import sys
from arg_process import *
from test import *
from data_prep import *
from cnn_data_prep import *
from classifier import *
from compute_auc import *


params = {
# Application parameters
'tmp_dir':'./', # temporal results directory
'smallest_axe':0, # smallest axes length (pixels), both x and y
'largest_axe':0, # largest axes length (pixels), both x and y
'median':0, # median (percent)
'resize_to':640, # resize to image with given size

# Neural network parameter:
'batch_size': 32,
'learn_rate': 0.0001,
'decay': 0.001,
'epochs': 100,
'dropout': 0.75,
'filters': 8,
'dense': 256,
'activation': 'relu',
'patience': 10,
# Category to experiment on
'category': ['dog','cat','bear'],
# Seeds for train/test splits
'seeds': [1, 2, 3, 4, 5]
}





def check_input(argv):
	global params
	args, params = check_arguments(argv, params)
	if args['load_flag']:
		prepare_data(params)
	if args['unzip_preloaded']:
		unzip_preloaded()
	if args['test_flag']:
		test(params)
	if args['filter_flag']:
		filter_dataset(params)
	if args['resize_flag']:
		resize_patches_to_size(params)
	if args['run_cv']:
		perform_5_2_CV(params)
	if args['auc']:
		get_CV_AUC(params)
	


def main():
	check_input(sys.argv)


if __name__ == "__main__":
	main()