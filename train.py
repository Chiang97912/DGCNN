# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:57:59 2018

@author: Peter
"""
import time
import numpy as np
import tensorflow as tf
from dgcnn import DGCNN
from prepro import get_max_length, load_embedding, load_data, next_batch
from config import get_config


def main():
    training_file = "./new_data/training.json"
    validation_file = "./new_data/validation.ann.json"
    trained_model = "./checkpoints/model.ckpt"
    embedding_file = "D:/DataMining/QASystem/wiki/wiki.zh.text.vector"
    # embedding_file = "./wiki.zh.text.vector"
    embedding_size = 60  # Word embedding dimension
    epochs = 100
    batch_size = 64  # Batch data size
    sequence_length = 150  # Sentence length
    learning_rate = 0.0001
    lrdown_rate = 1
    gpu_mem_usage = 0.75
    gpu_device = "/gpu:0"
    cpu_device = "/cpu:0"

    config = get_config()  # Not used yet
    embeddings, word2idx = load_embedding(embedding_file)
    questions, evidences, y1, y2 = load_data(training_file, word2idx, sequence_length)
    questions_vali, evidences_vali, y1_vali, y2_vali = load_data(validation_file, word2idx, sequence_length)
    data_size = len(questions)
    permutation = np.random.permutation(data_size)
    questions = questions[permutation, :]
    evidences = evidences[permutation, :]
    y1 = y1[permutation]
    y2 = y2[permutation]
    with tf.Graph().as_default(), tf.device(gpu_device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.variable_scope('Model'):
            model = DGCNN(config, embeddings, sequence_length, embedding_size)
            with tf.Session(config=session_conf).as_default() as sess:
                saver = tf.train.Saver()
                print("Start training")
                sess.run(tf.global_variables_initializer())
                for i in range(epochs):
                    batch_number = 1
                    for batch_questions, batch_evidences, batch_y1, batch_y2 in next_batch(questions, evidences, y1, y2, batch_size):
                        start_time = time.time()
                        feed_dict = {
                            model.e: batch_evidences,
                            model.q: batch_questions,
                            model.y1: batch_y1,
                            model.y2: batch_y2,
                            model.lr: learning_rate,
                            model.is_train: True
                        }
                        _, loss, acc1, acc2 = sess.run([model.train, model.loss, model.acc1, model.acc2], feed_dict)
                        duration = time.time() - start_time
                        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAcc1 %2.3f\tAcc2 %2.3f' % (i + 1, batch_number * batch_size, data_size, duration, loss, acc1, acc2))
                        batch_number += 1
                    learning_rate *= lrdown_rate

                    # validation
                    start_time = time.time()
                    feed_dict = {
                        model.e: evidences_vali,
                        model.q: questions_vali,
                        model.y1: y1_vali,
                        model.y2: y2_vali,
                        model.is_train: False
                    }
                    loss, acc1, acc2 = sess.run([model.loss, model.acc1, model.acc2], feed_dict)
                    duration = time.time() - start_time
                    print('Validation: Time %.3f\tLoss %2.3f\tAcc1 %2.3f\tAcc2 %2.3f' % (duration, loss, acc1, acc2))

                    saver.save(sess, trained_model)

                print("End of the training")


if __name__ == '__main__':
    main()
