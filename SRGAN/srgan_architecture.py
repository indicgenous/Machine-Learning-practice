import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input, Add, UpSampling2D, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

def build_generator(input_shape):
    def residual_block(x):
        res = Conv2D(64, (3, 3), padding='same')(x)
        res = BatchNormalization(momentum=0.8)(res)
        res = Activation('relu')(res)
        res = Conv2D(64, (3, 3), padding='same')(res)
        res = BatchNormalization(momentum=0.8)(res)
        res = Add()([res, x])
        return res

    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), padding='same')(input_layer)
    x = Activation('relu')(x)
    res = x

    for _ in range(16):
        res = residual_block(res)

    x = Conv2D(64, (3, 3), padding='same')(res)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, input_layer])

    x = UpSampling2D(size=2)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    output_layer = Conv2D(3, (9, 9), padding='same', activation='tanh')(x)
    return Model(input_layer, output_layer)

def build_discriminator(input_shape):
    def conv_block(x, filters, strides=1):
        x = Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        return x

    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    for filters, strides in zip([64, 128, 128, 256, 256, 512, 512], [2, 1, 2, 1, 2, 1, 2]):
        x = conv_block(x, filters, strides)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    return Model(input_layer, output_layer)

def build_vgg():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(None, None, 3))
    vgg.trainable = False
    model = Model(vgg.input, vgg.layers[20].output)
    return model

# Define input shape
input_shape = (64, 64, 3)

# Build and compile the models
generator = build_generator(input_shape)
discriminator = build_discriminator((256, 256, 3))
vgg = build_vgg()

# Print model summaries
generator.summary()
discriminator.summary()
vgg.summary()