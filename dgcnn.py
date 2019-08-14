# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:15:52 2018

@author: Peter
"""
import tensorflow as tf
from modules import mask, position_embedding, multihead_attention_encoder, create_kernel_initializer, create_bias_initializer


class DGCNN(object):
    def __init__(self, config, embeddings, sequence_length, embedding_size):
        self.config = config  # Not used yet
        self.embeddings = embeddings
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.build_model(config)

    def build_model(self, config):
        self.q = tf.placeholder(tf.int32, [None, self.sequence_length])  # question
        self.e = tf.placeholder(tf.int32, [None, self.sequence_length])  # evidence
        self.y1 = tf.placeholder(tf.int32, [None, None])  # the label of answer start
        self.y2 = tf.placeholder(tf.int32, [None, None])  # the label of answer end
        self.lr = tf.placeholder(tf.float32)  # learning rate
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.Q, self.E = self.embedding_layer(self.q, self.e, self.embeddings)
        # self.P = position_embedding(self.e)
        self.encoded_q = self.conv1d_block(self.Q, 3, dilation_rate=1, scope='conv1d_block_q')
        self.encoded_q = self.attention_encoder(self.encoded_q)
        # self.encoded_q = multihead_attention_encoder(self.encoded_q, self.encoded_q, self.encoded_q, 8, 16)

        self.merged = tf.concat([self.E, self.encoded_q], 2)  # (batch_size, sequence_length, 2*embedding_size)
        self.merged = tf.layers.dense(self.merged, self.sequence_length)
        self.merged = self.conv1d_block(self.merged, 1, dilation_rate=1, scope='conv1d_block_merge')
        self.merged = self.conv1d_block(self.merged, 3, dilation_rate=1, scope='conv1d_block_dilation1')
        self.merged = self.conv1d_block(self.merged, 3, dilation_rate=2, scope='conv1d_block_dilation2')
        self.merged = self.conv1d_block(self.merged, 3, dilation_rate=4, scope='conv1d_block_dilation4')
        # self.merged = tf.layers.dropout(self.merged, rate=0.25, training=self.is_train)
        # self.merged = tf.layers.batch_normalization(self.merged, training=True)
        self.p1, self.p2 = self.output_layer(self.merged)

        self.train, self.loss = self.optimize(self.p1, self.p2, self.y1, self.y2, self.lr)
        self.acc1 = self.compute_accuracy(self.p1, self.y1)
        self.acc2 = self.compute_accuracy(self.p2, self.y2)

    def embedding_layer(self, q, e, embeddings):
        embeddings = tf.constant(embeddings, dtype=tf.float32)
        embed_q = tf.nn.embedding_lookup(embeddings, q)
        embed_q = tf.layers.dropout(embed_q, rate=0.25, training=self.is_train)
        embed_e = tf.nn.embedding_lookup(embeddings, e)
        embed_e = tf.layers.dropout(embed_e, rate=0.25, training=self.is_train)

        return embed_q, embed_e

    def conv1d_block(self, X, kernel_size, dilation_rate=1, padding='same', scope='conv1d_block'):
        """
        gated dilation conv1d layer
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            filters = X.get_shape().as_list()[-1]
            glu = tf.layers.conv1d(X, filters, kernel_size, dilation_rate=dilation_rate, padding=padding,
                                kernel_initializer=create_kernel_initializer('conv'),
                                 bias_initializer=create_bias_initializer('conv'))
            glu = tf.sigmoid(tf.layers.dropout(glu, rate=0.1, training=self.is_train))
            conv = tf.layers.conv1d(X, filters, kernel_size, dilation_rate=dilation_rate, padding=padding,
                                kernel_initializer=create_kernel_initializer('conv'),
                                 bias_initializer=create_bias_initializer('conv'))
            gated_conv = tf.multiply(conv, glu)
            gated_x = tf.multiply(X, 1 - glu)
            outputs = tf.add(gated_x, gated_conv)

            # mask
            outputs = tf.where(tf.equal(X, 0), X, outputs)

            # outputs = tf.layers.dropout(outputs, rate=0.25, training=self.is_train)
            # outputs = tf.layers.batch_normalization(outputs, training=True)
            return outputs

    def attention_encoder(self, X, hidden_size=128, scope='attention_encoder'):
        """
        attention encoder layer
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            T = X.get_shape().as_list()[1]
            h = tf.nn.tanh(tf.layers.dense(X, hidden_size, use_bias=False, kernel_initializer=create_kernel_initializer('dense'), name='h'))
            attention = tf.layers.dense(h, T, use_bias=False, kernel_initializer=create_kernel_initializer('dense'), name='att')

            # mask: from transformer
            padding_num = -2 ** 32 + 1  # multiply max number, let 0 index of timestep equal softmax 0
            masks = tf.sign(tf.reduce_sum(tf.abs(X), axis=-1))  # [N, T]
            masks = tf.tile(tf.expand_dims(masks, axis=1), [1, T, 1])  # [N, T, T]
            paddings = tf.ones_like(masks) * padding_num
            attention = tf.where(tf.equal(masks, 0), paddings, attention)

            # softmax
            attention = tf.nn.softmax(attention)

            outputs = tf.matmul(attention, X)

            # outputs = tf.layers.dropout(outputs, rate=0.25, training=self.is_train)
            # outputs = tf.layers.batch_normalization(outputs, training=True)
            return outputs

    def output_layer(self, X, hidden_size=128, scope='output_layer'):
        """
        the output layer
        utilize the attention mechanism as a pointer to select the start position and end position
        """
        X = tf.layers.flatten(X)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            s1 = tf.nn.tanh(tf.layers.dense(X, hidden_size, kernel_initializer=create_kernel_initializer('dense'), bias_initializer=create_bias_initializer('dense'), name='s1'))
            p1 = tf.layers.dense(s1, self.sequence_length, kernel_initializer=create_kernel_initializer('dense'), bias_initializer=create_bias_initializer('dense'), name='p1')
            s2 = tf.nn.tanh(tf.layers.dense(X, hidden_size, kernel_initializer=create_kernel_initializer('dense'), bias_initializer=create_bias_initializer('dense'), name='s2'))
            p2 = tf.layers.dense(s2, self.sequence_length, kernel_initializer=create_kernel_initializer('dense'), bias_initializer=create_bias_initializer('dense'), name='p2')
            return p1, p2

    def optimize(self, logit1, logit2, y1, y2, lr):
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit1, labels=y1))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit2, labels=y2))
        loss = loss1 + loss2
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train = optimizer.apply_gradients(zip(grads, tvars))

        return train, loss

    def compute_accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy
