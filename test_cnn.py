from datasets import make_dataset, get_class_nms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [
    'IPAPGothic', 'Takao', 'Hiragino Maru Gothic Pro']

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def test_cnn_by_compositions(data_nm, train_size, use_classes='all',
                             n_iter=3, **cnn_params):
    W = load_W(data_nm, use_classes=use_classes)
    c_complexed, c_simple = calc_compos_from_W(W)

    res = {'dataset': data_nm, 'train_size': train_size}
    for state, c in zip(('balanced', 'complexed', 'simple'),
                       (None, c_complexed, c_simple)):
        print('\nstate:', state)
        train_accs, test_accs = [], []
        for i in range(n_iter):
            print('\nepoch:', i + 1)
            args = {'data_nm': data_nm, 'train_size': train_size,
                    'use_classes': use_classes,
                    'composition': c}
            train_acc, test_acc = test_cnn(**args, **cnn_params)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        train_score = np.mean(train_accs)
        test_score = np.mean(test_accs)
        print(f'train by {state} {train_size}samples >> '
              f'train:{train_score} test:{test_score}')
        res[state] = test_score
    return res


def load_W(data_nm, use_classes='all',
           emb_method='tsne', emb_dim=2, n_neighbors=3):
    opt = f'{data_nm}_cae_{emb_method}{emb_dim}d_{n_neighbors}K'
    W = np.load(f'results/W_{opt}.dat', allow_pickle=True)
    if use_classes != 'all':
        class_nms = get_class_nms(data_nm)
        use_labels = [i for ucls in use_classes for i, cls
                      in enumerate(class_nms) if cls == ucls]
        W = W[use_labels, :][:, use_labels]
    return W


def calc_compos_from_W(W):
    prod = W.prod(0)
    c_complexed  = prod / prod.sum()
    prod_inv = 1 / prod
    c_simple = prod_inv / prod_inv.sum()
    return c_complexed, c_simple


def preprocess_dataset(dataset):
    n_classes = len(dataset['class_nms'])
    for tr_te in ('train', 'test'):
        imgs = dataset[f'{tr_te}_images']
        labels = dataset[f'{tr_te}_labels']
        if len(imgs[0].shape) == 2:
            imgs = imgs[:, :, :, np.newaxis]
        dataset[f'{tr_te}_images'] = imgs / 255.
        dataset[f'{tr_te}_labels'] = np.eye(n_classes)[labels]
    return dataset


def build_simple_cnn(img_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                    input_shape=img_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def build_resnet50(img_shape, n_classes, weights='imagenet'):
    input_tensor = Input(shape=img_shape)
    resnet50 = ResNet50(include_top=False, weights=weights,
                        input_tensor=input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnet50.output_shape[1:]))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(n_classes, activation='softmax'))

    img = Input(shape=img_shape)
    model = Model(img, top_model(resnet50(img)))
    return model


def test_cnn(data_nm, train_size, cv=5, use_classes='all',
             composition=None, class_weight=None,
             model='simple_cnn', batch_size=32, epochs=100):
    balanced = False if composition is not None else True
    dataset = make_dataset(data_nm, train_size,
                           use_classes=use_classes,
                           balanced=balanced,
                           composition=composition)
    y = dataset['train_labels']
    y_test = dataset['test_labels']
    dataset = preprocess_dataset(dataset)
    n_classes = len(np.unique(y))

    skf = StratifiedKFold(cv, shuffle=True, random_state=42)
    y_preds, models, histories = [], [], []
    for i, (train_idx, val_idx) in enumerate(
            skf.split(dataset['train_images'], y), 1):
        X_train = dataset['train_images'][train_idx]
        y_train = dataset['train_labels'][train_idx]
        X_val = dataset['train_images'][val_idx]
        y_val =  dataset['train_labels'][val_idx]
        if model == 'resnet50':
            model = build_resnet50(X_train[0].shape, n_classes)
        else:
            model = build_simple_cnn(X_train[0].shape, n_classes)
        model.compile(SGD(1e-3, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        hist = model.fit(
            X_train, y_train, validation_data=[X_val, y_val],
            batch_size=batch_size, epochs=epochs, verbose=0,
            class_weight=class_weight)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f'fold{i} loss:{val_loss} acc:{val_acc}')
        y_pred = np.argmax(model.predict(X_val), 1)
        y_pred = pd.Series(y_pred, index=val_idx)
        y_preds.append(y_pred)
        models.append(model)
        histories.append(hist.history)
    y_preds = pd.concat(y_preds).sort_index().values
    y_test_preds = np.array([
        model.predict(dataset['test_images']) for model in models])
    y_test_preds = y_test_preds.sum(0).argmax(1)
    train_acc = accuracy_score(y, y_preds)
    test_acc = accuracy_score(y_test, y_test_preds)
    print(f'train acc:{train_acc} test acc:{test_acc}')
    history_ = {k: np.mean([hist[k] for hist in histories], 0)
                for k in histories[0].keys()}
    plot_history(history_)
    return train_acc, test_acc


def plot_history(history, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    x = list(range(1, len(history['loss']) + 1))
    for i, k in enumerate(['loss', 'acc']):
        y_train = history[k]
        y_val = history[f'val_{k}']
        label_train = 'train ({:.3f})'.format(y_train[-1])
        label_val = 'val ({:.3f})'.format(y_val[-1])
        ax[i].plot(x, y_train, label=label_train)
        ax[i].plot(x, y_val, label=label_val)
        ax[i].set_xlabel('epoch', size=25)
        ax[i].set_ylabel(k, size=25)
        ax[i].legend(fontsize=15)
    if save_path != None:
        plt.savefig(save_path)
    plt.show()
    plt.close()
