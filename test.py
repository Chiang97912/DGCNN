# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:57:59 2018

@author: Peter
"""
import numpy as np
import tensorflow as tf
from dgcnn import DGCNN
from prepro import get_max_length, load_embedding, load_data, next_batch
from config import get_config


def main():
    testing_file = "./new_data/test.ann.json"
    trained_model = "./checkpoints/model.ckpt"
    embedding_file = "D:/DataMining/QASystem/wiki/wiki.zh.text.vector"
    # embedding_file = "./wiki.zh.text.vector"
    embedding_size = 60  # Word embedding dimension
    batch_size = 64  # Batch data size
    sequence_length = 150  # Sentence length
    learning_rate = 0.01
    gpu_mem_usage = 0.75
    gpu_device = "/gpu:0"
    cpu_device = "/cpu:0"

    config = get_config()  # Not used yet
    embeddings, word2idx = load_embedding(embedding_file)
    questions, evidences, y1, y2 = load_data(testing_file, word2idx, sequence_length)
    with tf.Graph().as_default(), tf.device(gpu_device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.variable_scope('Model'):
            model = DGCNN(config, embeddings, sequence_length, embedding_size)
            with tf.Session(config=session_conf).as_default() as sess:
                saver = tf.train.Saver()
                print("Start loading the model")
                saver.restore(sess, trained_model)
                print("The model is loaded")
                acc1, acc2 = [], []
                for batch_questions, batch_evidences, batch_y1, batch_y2 in next_batch(questions, evidences, y1, y2, batch_size):
                    feed_dict = {
                        model.e: batch_evidences,
                        model.q: batch_questions,
                        model.y1: batch_y1,
                        model.y2: batch_y2,
                        model.is_train: False
                    }
                    acc1_, acc2_ = sess.run([model.acc1, model.acc2], feed_dict)
                    acc1.append(acc1_)
                    acc2.append(acc2_)
                    print('Acc1 %2.3f\tAcc2 %2.3f' % (acc1_, acc2_))
                print('Average: Acc1 %2.3f\tAcc2 %2.3f' % (np.mean(acc1), np.mean(acc2)))


if __name__ == '__main__':
    main()
