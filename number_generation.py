from keras.datasets import mnist
from keras import layers
from keras import Model
from keras.optimizers import SGD
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_generator():
    int_input = layers.Input(shape=(10,))
    y = layers.RepeatVector(10)(int_input)
    y = layers.Reshape((100,))(y)
    input = layers.Input(shape=(10,))
    x = layers.Dense(100, activation="tanh")(input)
    x = layers.Multiply()([x, y])
    x = layers.Dense(2048, activation="tanh")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation="tanh")(x)
    x = layers.Dense(784, activation="tanh")(x)
    output = layers.Reshape((28, 28))(x)
    model = Model(inputs=[input, int_input], outputs=output)
    model.compile(optimizer=SGD(lr=0.08, momentum=0.9), loss='binary_crossentropy')
    model.summary()
    return model


def create_discriminator():
    int_input = layers.Input(shape=(10,))
    input = layers.Input(shape=(28, 28))

    x = layers.Reshape((28, 28, 1))(input)
    x = layers.Conv2D(filters=40, kernel_size=(8, 8), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(filters=60, kernel_size=(5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=100, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(100)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(10, activation="sigmoid")(x)
    output = layers.Dot(-1)([x, int_input])
    model = Model(inputs=[input, int_input], outputs=output)
    model.compile(optimizer=SGD(lr=0.04, momentum=0.9), loss='binary_crossentropy')
    model.summary()
    return model


def create_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(10,))  # 785
    gan_int = layers.Input((10,))
    x = generator([gan_input, gan_int])
    gan_output = discriminator([x, gan_int])
    gan = Model(inputs=[gan_input, gan_int], outputs=gan_output)
    gan.summary()
    gan.compile(loss='binary_crossentropy', optimizer='SGD')
    return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, 10])  # 785
    int_input = []
    for i in range(examples):
        encoder = [0] * 10
        encoder[i % 10] = 1
        int_input.append(encoder)
    generated_images = generator.predict([noise, np.array(int_input)])
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' % epoch)
    plt.close()


if __name__ == '__main__':
    batch_size = 60
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    idx = np.argsort(y_train)
    x_train = x_train[idx]
    y_train = y_train[idx]
    xindex = [0]
    for i in range(59999):
        if y_train[i] != y_train[i + 1]:
            xindex.append(i + 1)
    xindex.append(60000)
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(generator, discriminator)
    for e in tqdm(range(100000)):
        if e % 1000 == 0 and e != 0:
            plot_generated_images(e, generator)
        int_to_train = np.random.randint(0, 10)
        encoder = [0] * 10
        encoder[int_to_train] = 1
        int_label = np.array(2 * batch_size * encoder).reshape(2 * batch_size, 10)
        int_generator = np.array(batch_size * encoder).reshape(batch_size, 10)
        batch = np.concatenate(
            [x_train[np.random.randint(low=xindex[int_to_train], high=xindex[int_to_train + 1], size=batch_size)],
             generator.predict([np.random.normal(0, 1, size=[batch_size, 10]), int_generator])])
        labels = np.zeros(2 * batch_size)
        labels[:batch_size] = 0.99
        discriminator.train_on_batch([batch, int_label], labels)

        batch = np.random.normal(0, 1, size=[batch_size, 10])  # 785
        labels = np.ones(batch_size)
        gan.train_on_batch([batch, np.array(int_generator)], labels)
