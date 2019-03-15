# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:57:15 2018

@author: Peter
"""

import numpy as np
import re
import json
from collections import defaultdict

def get_max_length(filename):
    max_question_len = 0
    max_evidence_len = 0
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            que_len = len(data['question_tokens'])
            evi_len = len(data['evidence_tokens'])
            if que_len > max_question_len:
                max_question_len = que_len
            
            if evi_len > max_evidence_len:
                max_evidence_len = evi_len
    if max_evidence_len > max_question_len:
        return max_evidence_len
    else:
        return max_question_len

def load_embedding(filename):
    embeddings = []
    word2idx = defaultdict(list)
    print("开始加载词向量")
    with open(filename,mode='r',encoding='utf-8') as f:
        for line in f:
            arr = line.split(" ")
            embedding = [float(val) for val in arr[1:len(arr)]]
            word2idx[arr[0]] = len(word2idx)
            embeddings.append(embedding)

    embedding_size = len(arr) - 1
    word2idx["UNKNOWN"] = len(word2idx)
    embeddings.append([0]*embedding_size)

    word2idx["NUM"] = len(word2idx)
    embeddings.append([0]*embedding_size)
    print("词向量加载完毕")
    return embeddings,word2idx

def sentence2index(sentence,word2idx,max_len):
    unknown = word2idx.get("UNKNOWN")
    num = word2idx.get("NUM")
    index = [unknown] * max_len
    i = 0
    for word in sentence:
        if word in word2idx:
            index[i] = word2idx[word]
        else:
            if re.match("\d+",word):
                index[i] = num
            else:
                index[i] = unknown
        if i >= max_len-1:
            break
        i += 1
    return index

def load_data(filename,word2idx,max_len):
    questions,evidences,y1,y2 = [],[],[],[]
    print("开始解析数据")
    with open(filename,'r') as f:
        for line in f:
            data = json.loads(line)
            question = data['question_tokens']
            questionIdx = sentence2index(question, word2idx, max_len)
            evidence = data['evidence_tokens']
            evidenceIdx = sentence2index(evidence, word2idx, max_len)
            start_index = data['answer_start']
            # end_index = data['answer_start'] + len(data['golden_answers']) - 1
            end_index = data['answer_end']
            as_temp = np.zeros(max_len)
            ae_temp = np.zeros(max_len)
            as_temp[start_index] = 1
            ae_temp[end_index] = 1
            questions.append(questionIdx)
            evidences.append(evidenceIdx)
            y1.append(as_temp)
            y2.append(ae_temp)
    print("解析数据完毕")
    return questions,evidences,y1,y2

def next_batch(questions,evidences,y1,y2,batch_size):
    data_len = len(questions)
    batch_num = int(data_len/batch_size)
    
    for batch in range(batch_num):
        result_questions,result_evidences,result_y1,result_y2 = [],[],[],[]
        
        for i in range(batch*batch_size,min((batch+1)*batch_size,data_len)):
            result_questions.append(questions[i])
            result_evidences.append(evidences[i])
            result_y1.append(y1[i])
            result_y2.append(y2[i])
        yield np.array(result_questions),np.array(result_evidences),np.array(result_y1),np.array(result_y2)