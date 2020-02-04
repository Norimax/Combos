import numpy as np
import pandas as pd
import scipy.io as sio


def make_dataset(data_nm, train_size, use_classes='all',
                 balanced=True, composition=None):
    class_nms = get_class_nms(data_nm)
    if use_classes == 'all':
        use_labels = [i for i, cls in enumerate(class_nms)]
    else:
        use_labels = [i for ucls in use_classes for i, cls
                      in enumerate(class_nms) if cls == ucls]
        label_mapper = {label: new_label for new_label, label
                        in enumerate(use_labels)}
    if balanced:
        composition = np.ones(len(use_labels))
        composition = composition / composition.sum()

    X_train, y_train, X_test, y_test = load_data(data_nm)
    X_train_stack, y_train_stack = [], []
    X_test_stack, y_test_stack = [], []
    for label, c in zip(use_labels, composition):
        sample_size = int(train_size * c)
        X, y = sample_imgages_by_class(
            X_train, y_train, label, sample_size)
        X_train_stack.append(X)
        y_train_stack.append(y)

        X = X_test[y_test == label]
        y = y_test[y_test == label]
        X_test_stack.append(X)
        y_test_stack.append(y)
    X_train = np.concatenate(X_train_stack)
    y_train = np.concatenate(y_train_stack)
    X_test = np.concatenate(X_test_stack)
    y_test = np.concatenate(y_test_stack)

    if use_classes != 'all':
        y_train = pd.Series(y_train).replace(label_mapper).values
        y_test = pd.Series(y_test).replace(label_mapper).values
    dataset = {
        'data_nm': data_nm,
        'train_images': X_train, 'train_labels': y_train,
        'test_images': X_test, 'test_labels': y_test,
        'class_nms': np.array(class_nms)[use_labels].tolist()
    }

    for k, v in dataset.items():
        if k == 'data_nm':
            print('data_nm:', v)
        elif k == 'class_nms':
            d = {i: cls for i, cls in enumerate(v)}
            d_ = {i: c for i, c in enumerate(composition)}
            print('label and class:', d)
            print('compositions:', d_)
        else:
            print(f'{k} shape:', v.shape)
    return dataset


def get_class_nms(data_nm):
    dic = {
        'mnist': list(range(10)),
        'fashion_mnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                          'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                          'Ankle boot'],
        'notmnist': list('ABCDEFGHIJ'),
        'kmnist': list('おきすつなはまやれを'),
        'cifar10': ['airplay', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'],
        'svhn': list(range(1, 11)),
        'stl10': ['airplane', 'bird', 'car', 'cat', 'deer',
                  'dog', 'horse', 'monkey', 'ship', 'truck'],
        'coil10': ['car', 'fortune cat', 'anacin', 'baby powder',
                   'tylenol', 'vaseline', 'tea cup', 'pig bank',
                   'mug', 'convertible car']
    }
    return dic[data_nm]


def load_data(data_nm):
    data = sio.loadmat(f'data/{data_nm}.mat')
    X_train = data['train_images']
    y_train = data['train_labels'].reshape(-1)
    X_test = data['test_images']
    y_test = data['test_labels'].reshape(-1)
    return X_train, y_train, X_test, y_test


def sample_imgages_by_class(X, y, label, sample_size):
    np.random.seed(42)
    if (y == label).sum() < sample_size:
        print(f'the sample size of label {label} must be '
              f'less than {(y == label).sum()}')
        X_sample = X[y == label]
        y_sample = y[y == label]
    else:
        sample_idx = np.random.choice(np.arange(len(y))[y == label],
                                      sample_size, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
    return X_sample, y_sample
