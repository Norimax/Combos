import numpy as np
from tensorflow.keras.layers import Dense, Activation, Input, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['IPAPGothic', 'Takao', 'Hiragino Maru Gothic Pro']


class ConvolutionalAutoencoder:
    def __init__(self, img_shape, n_components):
        self.img_shape = self.r, self.c, self.ch = img_shape
        self.n_components = n_components

    def build_encoder(self):
        model = Sequential()
        model.add(Conv2D(
            32, (3, 3), padding='same', activation='tanh',
            input_shape=self.img_shape))
        model.add(Conv2D(
            64, (3, 3), padding='same', activation='tanh', strides=2))
        model.add(BatchNormalization())
        model.add(Conv2D(
            128, (3, 3), padding='same', activation='tanh', strides=2))
        model.add(BatchNormalization())
        if (self.r % 8 == 0) & (self.c % 8 == 0):
            model.add(Conv2D(
                128, (3, 3), padding='same', activation='tanh', strides=2))
            model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(self.n_components, activation='tanh'))
        return model

    def build_decoder(self):
        if (self.r % 8 == 0) & (self.c % 8 == 0):
            r_, c_ = int(self.r / 8), int(self.c / 8)
        else:
            r_, c_ = int(self.r / 4), int(self.c / 4)
        model = Sequential()
        model.add(Dense(r_ * c_ * 128, activation='tanh',
                        input_dim=self.n_components))
        model.add(Reshape((r_, c_, 128)))
        if (self.r % 8 == 0) & (self.c % 8 == 0):
            model.add(Conv2DTranspose(
                128, (3, 3), strides=2, activation='tanh', padding='same'))
            model.add(BatchNormalization())
        model.add(Conv2DTranspose(
            128, (3, 3), strides=2, activation='tanh', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(
            64, (3, 3), strides=2, activation='tanh', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(
            32, (3, 3), activation='tanh', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(
            self.ch, (3, 3), activation='sigmoid', padding='same'))
        return model

    def build(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        img = Input(self.img_shape)
        self.model = Model(img, self.decoder(self.encoder(img)))
        self.model.summary()

    def train(self, X, batch_size=32, epochs=[200, 100], verbose=0):
        self.model.compile(optimizer=Adam(1e-3), loss='mse')
        ep = epochs[0]
        self.model.fit(X, X, batch_size=batch_size,
                       epochs=ep, verbose=verbose)
        loss = self.model.evaluate(X, X)
        print(f'{ep}epoch loss:', loss)
        if loss > 5e-4:
            self.model.compile(optimizer=Adam(1e-4), loss='mse')
            ep = epochs[1]
            self.model.fit(X, X, batch_size=batch_size,
                           epochs=ep, verbose=verbose)
            loss = self.model.evaluate(X, X)
            print(f'{ep}epoch loss:', loss)

    def plot_reconstructed_img(self, img, model=None, save_path=None):
        if model == None:
            model = self.model
        reconst_img = model.predict(np.array([img]))[0]
        titles = ['Original', 'Reconstructed']
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        for i, img in enumerate([img, reconst_img]):
            if img.shape[-1] == 1:
                ax[i].imshow(img[:, :, 0], cmap='gray')
            else:
                ax[i].imshow(img)
            ax[i].set_title(titles[i], size=20)
            ax[i].axis('off')
        if save_path != None:
            plt.savefig(save_path)
        plt.show()
        plt.close()
