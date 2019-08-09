# -*- coding: utf-8 -*-
import re
import json
import tensorflow as tf


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        json_raw = config_file.read()
        # Processing // ... /n format non-json content
        json_str1 = re.sub(re.compile('(//[\\s\\S]*?\n)'), '', json_raw)
        # Processing /* ... */ format non-json content
        json_str2 = re.sub(re.compile('(/\*\*\*[\\s\\S]*?/)'), '', json_str1)

        config_dict = json.loads(json_str2)

    # convert the dictionary to a namespace using bunch lib
    config = Config(**config_dict)

    return config, config_dict


def get_config():
    flags = tf.app.flags
    flags.DEFINE_string("training_file", "./new_data/training.json", "")
    flags.DEFINE_string("validation_file", "./new_data/validation.ann.json", "")
    flags.DEFINE_string("testing_file", "./new_data/test.ann.json", "")
    flags.DEFINE_string("trained_model", "./checkpoints/model.ckpt", "")
    flags.DEFINE_string("embedding_file", "D:/DataMining/QASystem/wiki/wiki.zh.text.vector", "")

    flags.DEFINE_integer("embedding_size", 60, "Word embedding dimension")
    flags.DEFINE_integer("epochs", 50, "The number of epochs")
    flags.DEFINE_integer("batch_size", 64, "Batch data size")
    flags.DEFINE_integer("sequence_length", 150, "Sentence length")

    flags.DEFINE_float("learning_rate", 0.01, "")
    flags.DEFINE_float("lrdown_rate", 0.95, "")
    flags.DEFINE_float("gpu_mem_usage", 0.75, "")

    flags.DEFINE_string("gpu_device", "/gpu:0", "")
    flags.DEFINE_string("cpu_device", "/cpu:0", "")
    config = flags.FLAGS

    return config


if __name__ == '__main__':
    config, config_dict = get_config_from_json('config.json')
    print(config_dict)
