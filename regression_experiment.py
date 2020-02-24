import os
import re
import glob
import json
import shutil
import cv2 as cv
import numpy as np
import keras as ks
import tensorflow as tf 
import more_itertools as mit
import sklearn.model_selection as sks
import keras.utils as ku
import keras.layers as kl
from keras import backend as bk
#from cnn_data_prep import *
from tensorflow import set_random_seed
import pickle




def get_regression_res(params):
    catName = params['category']
    accuracies = []

    for curCat in catName:
        fname = params['tmp_dir'] + '{}_regression.txt'.format(curCat)
        crossValidation(params, curCat)
        #with open(fname, 'w') as f:
        #    for item in cur_acc:
        #        f.write("%s\n" % item)
        #accuracies.append(cur_acc)
    #print(accuracies)





def data_generator(params, animal, category, n_batches, run, fold):
    while True:
        for i in range(n_batches):
            feat_file = params['tmp_dir'] + 'training_data_processed/%s/run_%s/fold_%s/%s_features_%s.npy' % (animal,run,fold, category, i)
            lbl_file = params['tmp_dir'] + 'training_data_processed/%s/run_%s/fold_%s/%s_labels_%s.npy' % (animal,run,fold, category, i)
            yield (
                np.load(feat_file),
                np.load(lbl_file)
            )



def check_classes(animal):
    ratios = []

    for dataset in ['train', 'valid', 'test']:
        zeroes, ones = 0, 0
        file_names = [
            x for x in os.listdir('training_data_processed/%s' %(animal))
            if x.startswith('%s_labels' % dataset)
        ]

        for file_name in file_names:
            batch = np.load('training_data_processed/%s/%s' % (animal, file_name))
            zeroes += sum(np.all(batch == [1, 0], 1))
            ones += sum(np.all(batch == [0, 1], 1))

        ratios.append(zeroes / ones)

    return ratios




def prepare_data(out_path, animal, batch_size, seed, val_seed, fold_num, params):

    # Parse image number from data in labels.txt
    def parse_image_num(image_name):
        return int(re.match('([0-9]+)_[0-9]+.jpg', image_name).groups()[0])

    # Create output directory for processed data
    #shutil.rmtree(out_path)
    #os.makedirs(out_path)

    # Determine features and labels paths based on specified animal
    features_path, labels_path = params['tmp_dir'] + r'training_data/%s' % animal, params['tmp_dir'] +  r'%s_labels.txt' % animal

    # Read labels from the file
    # Trim the path, leaving only images names
    # Parse labels and store them along with image names as tuples
    # Parse the original image number, since it will be needed for creating data splits
    # Group labels based on the original image numbers
    # Remove original image names, since they are not longer needed after grouping
    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
        labels = [x[x.rfind('/') + 1:] for x in labels]
        labels = [(lambda t: (t[0], float(t[1])))(x.split(':')) for x in labels]
        labels = [(parse_image_num(x[0]), x) for x in labels]
        labels = mit.map_reduce(labels, keyfunc=lambda x: x[0], valuefunc=lambda x: x[1])
        labels = [(k, v) for k, v in labels.items()]
        labels.sort(key=lambda x: x[0])
        labels = [x[1] for x in labels]

    # Perform test/train split and flatten the labels
    labels_train, labels_test = sks.train_test_split(labels, test_size=0.5, random_state=seed)

    # Assign train data to test data and vise versa for second fold
    if fold_num == 1:
        labels_train, labels_test = labels_test, labels_train

    # Split training data into training and validation sets
    labels_train, labels_valid = \
        sks.train_test_split(labels_train, test_size=0.2, random_state=val_seed)

    # Flatten arrays after the splits are created
    labels_train, labels_test, labels_valid = \
        [list(mit.flatten(x)) for x in [labels_train, labels_test, labels_valid]]

    nums_batches = []

    # Process train and validation data
    for name, labels in zip(['train', 'valid', 'test'], [labels_train, labels_valid, labels_test]):

        # Calculate number of batches and append it to the list to be returned
        num_batches = int(np.ceil(len(labels) / batch_size))
        nums_batches.append(num_batches)

        # Create batches of features and labels, and store them as .npy files
        for batch_num in range(num_batches):
            batch_labels = labels[batch_num * batch_size:(batch_num + 1) * batch_size]
            batch_features = np.array([
                cv.imread('%s/%s' % (features_path, x[0])) for x in batch_labels
            ]).astype(np.float16) / 255
            #  batch_labels = ku.to_categorical(np.array(batch_labels)[:, 1].astype(np.float), 2)
            batch_labels = np.array(batch_labels)[:, 1].astype(np.float)
            for category, data in zip(['features', 'labels'], [batch_features, batch_labels]):
                np.save('%s/%s_%s_%s' % (out_path, name, category, batch_num), data)

    return nums_batches


def save_test_labels(data_dir, out_path, n_test_batches):
    result = np.array([])
    for i in range(n_test_batches):
        fpath = data_dir + 'test_labels_{}.npy'.format(i)
        labels = np.load(fpath)
        if result.shape[0]==0:
            result = labels
        else:
            result = np.concatenate((result, labels),axis=0)
    fpath = out_path + 'test_labels.npy'    
    np.save(fpath, result)




def get_batches_num(src_path):
    train_features_list = glob.glob(src_path+'train_features_*')
    n_train_features = len(train_features_list)

    test_features_list = glob.glob(src_path+'test_features_*')
    n_test_features = len(test_features_list)

    valid_features_list = glob.glob(src_path+'valid_features_*')
    n_valid_features = len(valid_features_list)
    
    return n_train_features, n_valid_features, n_test_features




def crossValidation(params, category):
    bk.clear_session()
    resulting_acc = []

    for i, seed in zip(range(5), params['seeds']):
        print('Starting %s/5 run' % (i + 1))

        for j in range(2):
            print('Starting %s/2 fold' % (j + 1))

            # Create output directory for storing models for i-run j-fold, if does not exist
            model_path = params['tmp_dir'] + 'models/%s/%s/%s' % (category, i, j)
            auc_path = params['tmp_dir'] + 'auc_src/%s/%s/%s/' % (category, i, j)

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(auc_path):
                os.makedirs(auc_path)

            print('Preparing the data...')

            # Process and prepare the data
            src_dir = params['tmp_dir'] + 'training_data_processed/'
            path = src_dir + '{}/run_{}/fold_{}/'.format(category, i, j)
            if os.path.exists(src_dir):
                shutil.rmtree(src_dir)
                os.makedirs(path)
            else:
                os.makedirs(path)

            n_train_batches, n_valid_batches, n_test_batches = prepare_data(
                path, category, params['batch_size'], seed, i, j, params
            )

            save_test_labels(path, auc_path, n_test_batches)

            #ratios = check_classes(params['animal'])

            #print('Class ratios: %s' % ', '.join([str(x) for x in ratios]))
            print('Creating the model...')

            # Set allow_growth to true to avoid memory hogging
            ks.backend.tensorflow_backend.set_session(
                tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            )

            # Create model
            #   Input shape: (-1, 640, 640, 3)
            #   Output shape: (-1, 2)
            set_random_seed(seed*100+j)
            model = ks.models.Sequential(
                [kl.InputLayer(input_shape=(640, 640, 3))] +
                list(mit.flatten([
                    [
                        kl.Conv2D(
                            filters=params['filters'] * (2**n), kernel_size=(3, 3),
                            activation=params['activation'], padding='same'
                        ),
                        kl.MaxPooling2D(pool_size=(2, 2))
                    ] for n in range(7)
                ])) +
                [kl.Flatten()] +
                list(mit.flatten([
                    [
                        kl.Dense(params['dense'], activation=params['activation']),
                        kl.Dropout(rate=params['dropout'])
                    ] for n in range(2)
                ])) +
                [kl.Dense(1, activation='linear')]
            )

            # Compile the model
            model.compile(
                optimizer=ks.optimizers.Adam(lr=params['learn_rate'], decay=params['decay']),
                loss=ks.losses.mean_squared_error, metrics=['accuracy']
            )

            # Print model summary
            # model.summary()

            print('Starting training...')

            src_path = params['tmp_dir']+'training_data_processed/{}/run_{}/fold_{}/'.format(category, i, j)
            # n_train_batches, n_valid_batches, n_test_batches = get_batches_num(src_path)

            # Train model
            history = model.fit_generator(
                generator=data_generator(params ,category, 'train', n_train_batches,i,j),
                validation_data=data_generator(params, category, 'valid', n_valid_batches,i,j),
                steps_per_epoch=n_train_batches, validation_steps=n_valid_batches,
                epochs=params['epochs'], verbose=0,
                callbacks=[
                    ks.callbacks.ModelCheckpoint(
                        filepath='%s/%s.hdf5' % (model_path, category), monitor='val_loss',
                        verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1
                    ),
                    ks.callbacks.EarlyStopping(
                        monitor='val_loss', min_delta=0, patience=params['patience'],
                        verbose=0, mode='auto', restore_best_weights=True
                    )
                ]
            )

            print('Starting testing...')

            # Load the best model (based on validation data)
            # model = ks.models.load_model('%s/%s.hdf5' % (model_path, params['animal']))
            #_, test_accuracy = model.evaluate_generator(
            #    generator=data_generator(params, category, 'test', n_test_batches, i, j), steps=n_test_batches
            #)
            test_prediction = model.predict_generator(
                generator=data_generator(params, category, 'test', n_test_batches, i, j), steps=n_test_batches
            )
            fname = auc_path + 'cnn_predictions.npy'
            np.save(fname, test_prediction)


            #resulting_acc.append(test_prediction)
            #print('Test accuracy: %.4f' % test_accuracy)
            #print('Saving results...')

            # Save history
            #with open('%s/%s_history.json' % (model_path, category), 'w') as file:
            #    file.write(json.dumps(history.history, indent=4))

            # Save hyper-parameters
            #with open('%s/%s_params.json' % (model_path, category), 'w') as file:
            #    file.write(json.dumps(params, indent=4))

            # Save test accuracy
            #with open('%s/%s_accuracy.json' % (model_path, params['animal']), 'w') as file:
            #    file.write(json.dumps({
            #        'test_accuracy': '%.4f' % test_accuracy,
            #        'class_ratios': {
            #            'train': ratios[0],
            #            'valid': ratios[1],
            #            'test': ratios[2]
            #        }
            #    }, indent=4))
    bk.clear_session()
    shutil.rmtree(src_dir)
    return
    # TODO - statistical tests go here
    # Test accuracies are saved in the respective folders (e.g. models/bear/0/0/bear_accuracy.json)




