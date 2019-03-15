# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:15:52 2018

@author: Peter
"""
import tensorflow as tf


class DGCNN(object):
    def __init__(self, embeddings, time_step, embedding_size, hidden_size):
        self.embeddings = embeddings
        self.time_step = time_step
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.q = tf.placeholder(tf.int32, [None, self.time_step])  # question
        self.e = tf.placeholder(tf.int32, [None, self.time_step])  # evidence
        self.y1 = tf.placeholder(tf.int32, [None, None])  # the label of answer start
        self.y2 = tf.placeholder(tf.int32, [None, None])  # the label of answer end
        self.lr = tf.placeholder(tf.float32)  # learning rate

        self.Q, self.E = self.embedding_layer(self.q, self.e, self.embeddings)

        self.encoded_q = self.conv1d_block(self.Q, self.embedding_size, 3, 1, pd='SAME')
        self.encoded_q = self.attention_encoder(self.encoded_q)

        self.merge = tf.concat([self.E, self.encoded_q], 2)  # (batch_size, 2*time_step, embedding_size)

        M = self.merge.get_shape().as_list()[-1]
        self.merge = tf.reshape(self.merge, [-1, M])
        self.merge = tf.layers.dense(self.merge, self.embedding_size)
        self.merge = tf.reshape(self.merge, [-1, self.time_step, self.embedding_size])

        self.merge = self.conv1d_block(self.merge, self.embedding_size, 1, 1)
        self.merge = self.conv1d_block(self.merge, self.embedding_size, 3, 1)
        self.merge = self.conv1d_block(self.merge, self.embedding_size, 3, 2)
        self.merge = self.conv1d_block(self.merge, self.embedding_size, 3, 4)
        self.merge = tf.layers.average_pooling1d(self.merge, pool_size=3, padding="SAME", strides=1)
        self.merge = tf.layers.flatten(self.merge)
        self.merge = tf.layers.dense(self.merge, 256)

        self.p1, self.p2 = self.output_layer(self.merge)

        self.train, self.loss = self.optimize(self.p1, self.p2, self.y1, self.y2, self.lr)
        self.acc_s = self.compute_accuracy(self.p1, self.y1)
        self.acc_e = self.compute_accuracy(self.p2, self.y2)

    def embedding_layer(self, q, e, embeddings):
        embeddings = tf.constant(embeddings, dtype=tf.float32)
        embed_q = tf.nn.embedding_lookup(embeddings, q)
        embed_e = tf.nn.embedding_lookup(embeddings, e)

        return embed_q, embed_e

    def conv1d_block(self, X, filters, kernel_size, dr, pd='SAME'):
        """
        gated dilation conv1d layer
        """
        glu = tf.sigmoid(tf.layers.conv1d(X, filters, kernel_size, dilation_rate=dr, padding=pd))
        conv = tf.layers.conv1d(X, filters, kernel_size, dilation_rate=dr, padding=pd)
        gated_conv = tf.multiply(conv, glu)

        gated_x = tf.multiply(X, 1 - glu)
        outputs = tf.add(gated_x, gated_conv)
        return outputs

    def attention_encoder(self, X, stddev=0.1):
        """
        attention encoder layer
        """
        M = X.get_shape().as_list()[1]
        N = X.get_shape().as_list()[2]
        reshaped_x = tf.reshape(X, [-1, N, M])
        attention = tf.layers.dense(reshaped_x, M, activation='softmax')
        attention = tf.reshape(attention, [-1, M, N])
        outputs = tf.multiply(X, attention)
        return outputs

    def output_layer(self, X):
        """
        the output layer
        utilize the attention mechanism as a pointer to select the start position and end position
        """
        p1 = tf.layers.dense(X, self.time_step)
        p2 = tf.layers.dense(X, self.time_step)

        return p1, p2

    def optimize(self, logit_s, logit_e, y1, y2, lr):
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_s, labels=y1))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_e, labels=y2))
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
