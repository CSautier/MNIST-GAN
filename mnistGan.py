#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:20:15 2019

@author: cstr
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 128

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=(x_train.astype(np.float32)-127.5)/127.5

def create_generator():
    input = layers.Input(shape=(784,))
    x= layers.Dense(784, activation="tanh")(input)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
#    x= layers.Dense(512, activation="tanh")(x)
#    x= layers.Dense(1024, activation="tanh")(x)
    x= layers.Dense(784, activation="tanh")(x)
    output=layers.Reshape((28,28))(x)
    model = Model(inputs=input, outputs=output) #the loss_function requires advantage and prediction, so we feed them to the network but keep them unchanged
    model.compile(optimizer=SGD(lr=0.08, momentum=0.9),loss='binary_crossentropy')
    model.summary()
    return model

def create_discriminator():
    input = layers.Input(shape=(28,28))
    x= layers.Reshape((28,28,1))(input)
    x = layers.Conv2D(filters=20, kernel_size=(8,8), activation='relu', padding='same')(x)
    x= layers.MaxPooling2D(pool_size=(3,3), strides=None, padding='valid', data_format=None)(x)
#    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same')(x)
    x= layers.MaxPooling2D(pool_size=(3,3), strides=None, padding='valid', data_format=None)(x)
#    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
    x= layers.MaxPooling2D(pool_size=(3,3), strides=None, padding='valid', data_format=None)(x)
#    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x= layers.Dense(256)(x)    
    x = layers.LeakyReLU(alpha=0.2)(x)
    
#    input = layers.Input(shape=(28,28))
#    x= layers.Reshape((28,28,1))(input)
#    x = layers.Conv2D(filters=32, kernel_size=(8,8), activation='relu', padding='same')(x)
#    x= layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None)(x)
#    x = layers.Conv2D(filters=60, kernel_size=(5,5), activation='relu', padding='same')(x)
#    x= layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None)(x)
#    x = layers.Flatten()(x)
#    x= layers.Dense(512)(x)
#    x = layers.LeakyReLU(alpha=0.2)(x)
#    x= layers.Dense(256)(x)
#    x = layers.LeakyReLU(alpha=0.2)(x)
    
    output=layers.Dense(1,activation="sigmoid")(x)
    model = Model(inputs=input, outputs=output) #the loss_function requires advantage and prediction, so we feed them to the network but keep them unchanged
    model.compile(optimizer=SGD(lr=0.04, momentum=0.9),loss='binary_crossentropy')
    model.summary()
    return model

def create_gan(generator, discriminator):
    discriminator.trainable=False
    gan_input = layers.Input(shape=(784,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.summary()
    gan.compile(loss='binary_crossentropy', optimizer='SGD')
    return gan

def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(0, 1, size=[examples, 784])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)

generator = create_generator()
discriminator = create_discriminator()
gan = create_gan(generator, discriminator)
    
for e in tqdm(range(20000)):
    if e%1000==0:
        plot_generated_images(e, generator)
    batch=np.concatenate([x_train[np.random.randint(low=0,high=x_train.shape[0],size=batch_size)], generator.predict(np.random.normal(0, 1, size=[batch_size,784]))])
    labels = np.zeros(2*batch_size)
    labels[:batch_size]=0.99
    discriminator.train_on_batch(batch, labels)

    batch=np.random.normal(0, 1, size=[batch_size,784])
    labels = np.ones(batch_size)
    gan.train_on_batch(batch, labels)

    