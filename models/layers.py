from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np


class Layer(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class Convolution2D(Layer):
    def __init__(self,
                 kernel_shape,
                 kernel=None,
                 bias=None,
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 activation=None,
                 scope=''):
        Layer.__init__(self)

        self.kernel_shape = kernel_shape
        self.kernel = kernel
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope

    def build(self, input_tensor):
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            self.kernel = tf.Variable(tf.truncated_normal(self.kernel_shape, stddev=0.1), name='kernel')

        kernel_height, kernel_width, num_input_channels, num_output_channels = self.kernel.get_shape()
        if self.bias:
            assert self.bias.get_shape() == (num_output_channels, )
        else:
            self.bias = tf.Variable(tf.constant(0.1, shape=[num_output_channels]), name='bias')

        conv = tf.nn.conv2d(input_tensor, self.kernel, strides=self.strides, padding=self.padding)

        if self.activation:
            return self.activation(conv + self.bias)
        return conv + self.bias

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class DeConvolution2D(Layer):
    def __init__(self,
                 kernel_shape,
                 output_shape,
                 kernel=None,
                 bias=None,
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 activation=None,
                 scope=''):
        Layer.__init__(self)

        self.kernel_shape = kernel_shape
        self.output_shape = output_shape
        self.kernel = kernel
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope

    def build(self, input_tensor):
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            self.kernel = tf.Variable(tf.truncated_normal(self.kernel_shape, stddev=0.1), name='kernel')

        window_height, window_width, num_output_channels, num_input_channels = self.kernel.get_shape()
        if self.bias:
            assert self.bias.get_shape() == (num_output_channels, )
        else:
            self.bias = tf.Variable(tf.constant(0.1, shape=[num_output_channels]), name='bias')

        deconv = tf.nn.conv2d_transpose(input_tensor,
                                        self.kernel,
                                        output_shape=self.output_shape,
                                        strides=self.strides,
                                        padding=self.padding)

        if self.activation:
            return self.activation(deconv + self.bias)
        return deconv + self.bias

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class MaxPooling(Layer):
    def __init__(self,
                 kernel_shape,
                 strides,
                 padding,
                 scope=''):
        Layer.__init__(self)

        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.scope = scope

    def build(self, input_tensor):
        return tf.nn.max_pool(input_tensor, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class UnPooling(Layer):
    def __init__(self,
                 kernel_shape,
                 output_shape,
                 scope=''):
        Layer.__init__(self)

        self.kernel_shape = kernel_shape
        self.output_shape = output_shape
        self.scope = scope

    def build(self, input_tensor):
        num_channels = input_tensor.get_shape()[-1]
        input_dtype_as_numpy = input_tensor.dtype.as_numpy_dtype()
        kernel_rows, kernel_cols = self.kernel_shape
        kernel_value = np.zeros((kernel_rows, kernel_cols, num_channels, num_channels), dtype=input_dtype_as_numpy)
        kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value)

        unpool = tf.nn.conv2d_transpose(input_tensor,
                                        kernel,
                                        output_shape=self.output_shape,
                                        strides=(1, kernel_rows, kernel_cols, 1),
                                        padding='VALID')
        return unpool

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class Unfold(Layer):
    def __init__(self,
                 scope=''):
        Layer.__init__(self)

        self.scope = scope

    def build(self, input_tensor):
        num_batch, height, width, num_channels = input_tensor.get_shape()

        return tf.reshape(input_tensor, [-1, (height * width * num_channels).value])

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class Fold(Layer):
    def __init__(self,
                 fold_shape,
                 scope=''):
        Layer.__init__(self)

        self.fold_shape = fold_shape
        self.scope = scope

    def build(self, input_tensor):
        return tf.reshape(input_tensor, self.fold_shape)

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class FullyConnected(Layer):
    def __init__(self,
                 output_dim,
                 weights=None,
                 bias=None,
                 activation=None,
                 scope=''):
        Layer.__init__(self)

        self.output_dim = output_dim
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.scope = scope

    def build(self, input_tensor):
        num_batch, input_dim = input_tensor.get_shape()

        if self.weights:
            assert self.weights.get_shape() == (input_dim.value, self.output_dim)
        else:
            self.weights = tf.Variable(tf.truncated_normal((input_dim.value, self.output_dim), stddev=0.1),
                                       name='weights')

        if self.bias:
            assert self.bias.get_shape() == (self.output_dim, )
        else:
            self.bias = tf.Variable(tf.constant(0.1, shape=[self.output_dim]), name='bias')

        fc = tf.matmul(input_tensor, self.weights) + self.bias

        if self.activation:
            return self.activation(fc)
        return fc

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


def main():
    conv = Convolution2D([5, 5, 1, 32])


if __name__ == '__main__':
    main()
