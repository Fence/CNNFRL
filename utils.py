#coding:utf-8
import os
import time
import pickle
import numpy as np


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def save_pkl(obj, path):
    with open(path, 'w') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path) as f:
        obj = pickle.load(f)
        return obj


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed


def ten_fold_split_idx(num_data, fname, k, random=True):
    """
    Split data for 10-fold-cross-validation
    Split randomly or sequentially
    Retutn the indecies of splited data
    """
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            indices = pickle.load(f)
            return indices
    n = num_data/k
    indices = []

    if random:
        tmp_idxs = np.arange(num_data)
        np.random.shuffle(tmp_idxs)
        for i in range(k):
            if i == k - 1:
                indices.append(tmp_idxs[i*n: ])
            else:
                indices.append(tmp_idxs[i*n: (i+1)*n])
    else:
        for i in xrange(k):
            indices.append(range(i*n, (i+1)*n))

    with open(fname, 'wb') as f:
        pickle.dump(indices, f)
    return indices


def index2data(indices, data):
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    if type(data) == dict:
        keys = data.keys()
        print('data.keys: {}'.format(keys))
        num_data = len(data[keys[0]])
        for i in xrange(len(indices)):
            valid_data = {}
            train_data = {}
            for k in keys:
                valid_data[k] = []
                train_data[k] = []
            for idx in xrange(num_data):
                for k in keys:
                    if idx in indices[i]:
                        valid_data[k].append(data[k][idx])
                    else:
                        train_data[k].append(data[k][idx])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)
    else:
        num_data = len(data)
        for i in xrange(len(indices)):
            valid_data = []
            train_data = []
            for idx in xrange(num_data):
                if idx in indices[i]:
                    valid_data.append(data[idx])
                else:
                    train_data.append(data[idx])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)

    return folds