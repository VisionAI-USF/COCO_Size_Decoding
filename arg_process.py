

def check_if_test(argv):
    if argv.lower() == "--test":
        return True
    else:
        return False


def check_if_load(argv):
    if argv.lower() == '--load':
        return True
    else:
        return False

def check_if_filter(argv):
    if argv.lower() == '--filter':
        return True
    else:
        return False

def check_if_tmp_dir(argv):
    if '--tmp_dir:' in argv.lower():
        args = argv.split(':')
        res_value = args[1]
        if res_value[-1]!='/':
            res_value = res_value+'/'
        return True, res_value
    else:
        return False, 0

def check_if_category(argv):
    if '--category:' in argv.lower():
        args = argv.split(':')
        res_value = args[1]
        args = res_value.split(',')
        res_value = args
        return True, res_value
    else:
        return False, []

def check_if_smallest_axe(argv):
    if '--smallest_axe:' in argv.lower():
        args = argv.split(':')
        res_value = int(args[1])
        return True, res_value
    else:
        return False, 0

def check_if_largest_axe(argv):
    if '--largest_axe:' in argv.lower():
        args = argv.split(':')
        res_value = int(args[1])
        return True, res_value
    else:
        return False, 0

def check_if_median(argv):
    if '--median:' in argv.lower():
        args = argv.split(':')
        res_value = int(args[1])
        return True, res_value
    else:
        return False, 0

def check_if_resize(argv):
    if '--resize:' in argv.lower():
        args = argv.split(':')
        res_value = int(args[1])
        return True, res_value
    else:
        return False, 0


def check_if_unzip_preloaded(argv):
    if argv.lower() == '--unzip_preloaded':
        return True
    else:
        return False


def check_if_run_cv(argv):
    if argv.lower() == '--run_cv':
        return True
    else:
        return False

def check_if_auc(argv):
    if argv.lower() == '--auc':
        return True
    else:
        return False



def check_arguments(argv, params):
	argn = len(argv)
	result = {'test_flag':False, 'load_flag':False,'filter_flag':False,
    'resize_flag':False, 'resize_value':0, 'unzip_preloaded':False,'run_cv':False,
    'auc':False}
	for j in range(1,argn):
		result['test_flag'] = (check_if_test(argv[j]) or result['test_flag'])
		result['unzip_preloaded'] = (check_if_unzip_preloaded(argv[j]) or result['unzip_preloaded'])
		result['load_flag'] = (check_if_load(argv[j]) or result['load_flag'])
		result['filter_flag'] = (check_if_filter(argv[j]) or result['filter_flag'])
		result['run_cv'] = (check_if_run_cv(argv[j]) or result['run_cv'])
		result['auc'] = (check_if_auc(argv[j]) or result['auc'])
		flag, flag_value = check_if_resize(argv[j])
		if flag:
			result['resize_flag'] = flag
			params['resize_to'] = flag_value
            
		flag, flag_value = check_if_tmp_dir(argv[j])
		if flag:
			params['tmp_dir'] = flag_value

		flag, flag_value = check_if_smallest_axe(argv[j])
		if flag:
			params['smallest_axe'] = flag_value

		flag, flag_value = check_if_largest_axe(argv[j])
		if flag:
			params['largest_axe'] = flag_value

		flag, flag_value = check_if_median(argv[j])
		if flag:
			params['median'] = flag_value

		flag, flag_value = check_if_category(argv[j])
		if flag:
			params['category'] = flag_value
            
	return result, params
