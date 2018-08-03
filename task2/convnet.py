import keras.backend as K
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model

import gated_conv


def vgg13(input_shape, n_classes):
    """Create a VGG13-style model.

    Args:
        input_shape (tuple): Shape of the input tensor.
        n_classes (int): Number of classes for classification.

    Returns:
        A Keras model of the VGG13 architecture.
    """
    input_tensor = Input(shape=input_shape, name='input_tensor')

    x = _conv_block(input_tensor, n_filters=64)
    x = _conv_block(x, n_filters=128)
    x = _conv_block(x, n_filters=256)
    x = _conv_block(x, n_filters=512)
    x = _conv_block(x, n_filters=512)

    x = GlobalAveragePooling2D()(x)

    x = Dense(n_classes, activation='softmax')(x)
    return Model(input_tensor, x, name='vgg13')


def gcnn(input_shape, n_classes):
    """Create a VGG13 model based on gated convolutions.

    Args:
        input_shape (tuple): Shape of the input tensor.
        n_classes (int): Number of classes for classification.

    Returns:
        A Keras model of the GCNN architecture.
    """
    input_tensor = Input(shape=input_shape, name='input_tensor')

    x = gated_conv.block(input_tensor, n_filters=64)
    x = gated_conv.block(x, n_filters=128)
    x = gated_conv.block(x, n_filters=256)
    x = gated_conv.block(x, n_filters=512)
    x = gated_conv.block(x, n_filters=512)

    x = GlobalAveragePooling2D()(x)

    x = Dense(n_classes, activation='softmax')(x)
    return Model(input_tensor, x, name='gcnn')


def crnn(input_shape, n_classes):
    """Create a convolutional recurrent neural network (CRNN) model.

    Args:
        input_shape (tuple): Shape of the input tensor.
        n_classes (int): Number of classes for classification.

    Returns:
        A Keras model of the CRNN architecture.
    """
    input_tensor = Input(shape=input_shape, name='input_tensor')

    x = _conv_block(input_tensor, n_filters=64)
    x = _conv_block(x, n_filters=128)
    x = _conv_block(x, n_filters=256)
    x = _conv_block(x, n_filters=512)
    x = _conv_block(x, n_filters=512)

    x = Lambda(lambda x: K.mean(x, axis=2))(x)
    x = Bidirectional(GRU(512, activation='relu',
                      return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(n_classes, activation='softmax')(x)
    return Model(input_tensor, x, name='crnn')


def gcrnn(input_shape, n_classes):
    """Create a CRNN model based on gated convolutions.

    Args:
        input_shape (tuple): Shape of the input tensor.
        n_classes (int): Number of classes for classification.

    Returns:
        A Keras model of the GCRNN architecture.
    """
    input_tensor = Input(shape=input_shape, name='input_tensor')

    x = gated_conv.block(input_tensor, n_filters=64)
    x = gated_conv.block(x, n_filters=128)
    x = gated_conv.block(x, n_filters=256)
    x = gated_conv.block(x, n_filters=512)
    x = gated_conv.block(x, n_filters=512)

    x = Lambda(lambda x: K.mean(x, axis=2))(x)
    x = Bidirectional(GRU(512, activation='relu',
                      return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(n_classes, activation='softmax')(x)
    return Model(input_tensor, x, name='crnn')


def _conv_block(x, n_filters, kernel_size=(3, 3), pool_size=(2, 2), **kwargs):
    """Apply two batch-normalized convolutions followed by max pooling.

    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of convolution filters.
        kernel_size (int or tuple): Convolution kernel size.
        pool_size (int or tuple): Max pooling parameter.
        kwargs: Other keyword arguments.

    Returns:
        tensor: The output tensor.
    """
    x = _conv_bn(x, n_filters, kernel_size, **kwargs)
    x = _conv_bn(x, n_filters, kernel_size, **kwargs)
    return MaxPooling2D(pool_size=pool_size)(x)


def _conv_bn(x, n_filters, kernel_size=(3, 3), **kwargs):
    """Apply a convolution operation followed by batch normalization.

    Args:
        x (tensor): Input tensor.
        n_filters (int): Number of convolution filters.
        kernel_size (int or tuple): Convolution kernel size.
        kwargs: Other keyword arguments.

    Returns:
        tensor: The output tensor.
    """
    x = Conv2D(n_filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               **kwargs)(x)
    return BatchNormalization(axis=-1)(x)
