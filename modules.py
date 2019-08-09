# -*- coding: utf-8 -*-
import tensorflow as tf


def conv(inputs, output_size, kernel_size=1, bias=None, activation=None, dilation_rate=1, name="conv", reuse=None):
        initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                     mode='FAN_AVG',
                                                                     uniform=True,
                                                                     dtype=tf.float32)
        initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                     mode='FAN_IN',
                                                                     uniform=False,
                                                                     dtype=tf.float32)
        regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
        with tf.variable_scope(name, reuse=reuse):
            shapes = inputs.shape.as_list()
            if len(shapes) > 4:
                raise NotImplementedError
            elif len(shapes) == 4:
                filter_shape = [1,kernel_size,shapes[-1],output_size]
                bias_shape = [1,1,1,output_size]
                strides = [1,1,1,1]
            else:
                filter_shape = [kernel_size,shapes[-1],output_size]
                bias_shape = [1,1,output_size]
                strides = 1
            conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
            kernel_ = tf.get_variable("kernel_",
                            filter_shape,
                            dtype=tf.float32,
                            regularizer=regularizer,
                            initializer=initializer_relu() if activation is not None else initializer())
            outputs = conv_func(inputs, kernel_, strides, "VALID", dilations=dilation_rate)
            if bias:
                outputs += tf.get_variable("bias_",
                            bias_shape,
                            regularizer=regularizer,
                            initializer=tf.zeros_initializer())
            if activation is not None:
                return activation(outputs)
            else:
                return outputs


def dense(inputs, hidden, use_bias=True, scope="dense"):
        with tf.variable_scope(scope):
            shape = tf.shape(inputs)
            dim = inputs.get_shape().as_list()[-1]
            out_shape = [shape[idx] for idx in range(
                len(inputs.get_shape().as_list()) - 1)] + [hidden]
            flat_inputs = tf.reshape(inputs, [-1, dim])
            W = tf.get_variable("W", [dim, hidden])
            res = tf.matmul(flat_inputs, W)
            if use_bias:
                b = tf.get_variable(
                    "b", [hidden], initializer=tf.constant_initializer(0.))
                res = tf.nn.bias_add(res, b)
            res = tf.reshape(res, out_shape)
            return res


def position_embedding(inputs, position_size=None):
        position_size = position_size or tf.shape(inputs)[-1]
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        position_j = 1. / tf.pow(10000., \
                     2 * tf.range(position_size / 2, dtype=tf.float32 \
                        ) / tf.cast(position_size, tf.float32))
        position_j = tf.expand_dims(position_j, 0)
        position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        position_i = tf.expand_dims(position_i, 1)
        position_ij = tf.matmul(position_i, position_j)
        position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
        position_embedding = tf.expand_dims(position_ij, 0) \
                             + tf.zeros((batch_size, seq_len, position_size))
        return position_embedding


def mask(inputs, seq_len, mode='mul'):
    if seq_len is None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12


def multihead_attention_encoder(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
    # 对Q、K、V分别作线性映射
    Q = tf.layers.dense(Q, nb_head * size_per_head, use_bias=False)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = tf.layers.dense(K, nb_head * size_per_head, use_bias=False)
    K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = tf.layers.dense(V, nb_head * size_per_head, use_bias=False)
    V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    # 计算内积，然后mask，然后softmax
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    # 输出并mask
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = mask(O, Q_len, 'mul')
    return O


def create_kernel_initializer(type='conv', stddev=0.1):
    if type == 'dense':
        # return tf.truncated_normal_initializer(stddev=stddev)
        return  # as default
    else:
        # return tf.contrib.layers.xavier_initializer()
        return  # as default


def create_bias_initializer(type='conv'):
    if type == 'dense':
        return tf.constant_initializer(1)
    else:
        return tf.constant_initializer(0)
