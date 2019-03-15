# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:57:59 2018

@author: Peter
"""

import tensorflow as tf
from dgcnn import DGCNN
from prepro import get_max_length, load_embedding, load_data, next_batch

if __name__ == '__main__':
    training_file = "D:/DataMining/QASystem/new_data/training.json"
    trained_model = "save/model.ckpt"
    embedding_file = "D:/DataMining/QASystem/wiki/wiki.zh.text.vector"
    embedding_size = 60  # Word embedding dimension
    epochs = 30
    batch_size = 128  # Batch data size
    hidden_size = 100  # Number of hidden layer neurons
    time_step = 100  # Sentence length
    keep_prob = 0.8
    learning_rate = 0.01
    lrdown_rate = 0.9
    gpu_mem_usage = 0.75
    gpu_device = "/gpu:0"

    time_step = get_max_length(training_file)
    embeddings, word2idx = load_embedding(embedding_file)
    questions, evidences, y1, y2 = load_data(training_file, word2idx, time_step)
    with tf.Graph().as_default(), tf.device(gpu_device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.variable_scope('Model'):
            model = DGCNN(embeddings, time_step, embedding_size, hidden_size)
            with tf.Session(config=session_conf).as_default() as sess:
                saver = tf.train.Saver()
                print("Start training")
                sess.run(tf.global_variables_initializer())
                for i in range(epochs):
                    print("The training of the %s iteration is underway" % (i + 1))
                    for batch_questions, batch_evidences, batch_y1, batch_y2 in next_batch(questions, evidences, y1, y2, batch_size):
                        feed_dict = {
                            model.e: batch_evidences,
                            model.q: batch_questions,
                            model.y1: batch_y1,
                            model.y2: batch_y2,
                            model.lr: learning_rate
                        }
                        _, loss, acc_s, acc_e = sess.run([model.train, model.loss, model.acc_s, model.acc_e], feed_dict)
                        print('LOSS: %s\t\tACC_S: %s\t\tACC_E: %s' % (loss, acc_s, acc_e))
                    learning_rate *= lrdown_rate
                    saver.save(sess, trained_model)
                print("End of the training")
