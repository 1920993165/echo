# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:03:06 2024

@author: 小怪兽
"""

import json
import copy
import time
import os
import requests
import copy
import csv
import matplotlib.pyplot as plt


#获得chatglm3结果
def get_conv(fn,idx):
    path=f'{fn}/{fn}-{idx}.json'
    with open(path, "r", encoding="utf-8") as file:
        conv = json.load(file)
    return conv

def count_acc(keys, sent1):
    for x in keys:
        if '|' in x:
            lst=x.split('|')
            flag=0
            for y in lst:
                if y in sent1:
                    flag=1
            if flag==0:
                return 0
        else:
            if x not in sent1:
                return 0
    # print(keys)
    # print(sent1)
    return 1


def exp2(fn):
    path='../test_data/exp2.json'
    with open(path, "r", encoding="utf-8") as file:
        testd = json.load(file)

    sent1_ls = []
    sent2_ls = []
    similarity_ls = []
    test_time_ls = []
    test_time_dict_s=[]
    test_time_dict_l=[]
    count=0
    for idx,one in enumerate(testd):
        print(f'progressing in {count}')
        conv=one['conversations']
        length_chat=len(conv)
        user_idx=0
        test_idx=1
        assi_idx=2
        chatglm3_conv=get_conv(fn,idx)
        chatglm3_conv=chatglm3_conv[0]['conversations']
        while assi_idx<length_chat:
            user_c=conv[user_idx]['content']
            time_c=conv[test_idx]['content']
            assi_c=conv[assi_idx]['content']
            if conv[test_idx]['test']==1:
                #获得
                sent1=assi_c
                sent2=chatglm3_conv[assi_idx]['content']
                key_word=conv[test_idx]['key-word']
                simi=count_acc(key_word,sent2)
                # print(simi)
                # 1/0
                test_level=conv[test_idx]['test-time']
                sent1_ls.append(sent1)
                sent2_ls.append(sent2)
                similarity_ls.append(simi)
                test_time_ls.append(test_level)
                if test_level=='s':
                    test_time_dict_s+=[simi,]
                else:
                    test_time_dict_l+=[simi,]
                # print('similarity',simi)
            user_idx+=3
            test_idx+=3
            assi_idx+=3
        count+=1


    # 假设的示例数据

    # 写入CSV文件
    csv_file_path = f"{fn}_acc.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["sent1", "sent2", "similarity", "test-level"])
        # 写入数据
        for i in range(len(sent1_ls)):
            writer.writerow([sent1_ls[i], sent2_ls[i], similarity_ls[i], test_time_ls[i]])
        
        # 写入总数据
        writer.writerow(['mean-easy'])
        writer.writerow([sum(test_time_dict_s)/len(test_time_dict_s)])
        writer.writerow(['mean-hard'])
        writer.writerow([sum(test_time_dict_l)/len(test_time_dict_l)])

import csv
import os
from itertools import islice

def read_last_lines(fn, num_lines=4):
    file_path=f"{fn}_acc.csv"
    with open(file_path, 'r') as file:
        # 跳过文件的前面部分，直接读取最后几行
        reverse_file = reversed(list(csv.reader(file)))
        last_lines = [row for row in islice(reverse_file, num_lines)]
    return last_lines


def draw_pic_easy(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo']):
    
    values=[]
    for idx,alg in enumerate(alg_ls):
        last_lines=read_last_lines(alg)
        values.append(float(last_lines[2][0]))
    colors = [
        '#1F77B4',  # 柔和的蓝色
        '#FF7F0E',  # 鲜明的橙色
        '#2CA02C',  # 鲜绿色
        '#D62728',  # 柔和的红色
        '#9467BD',  # 紫色
        '#8C564B',  # 棕色
        '#E377C2',  # 柔和的粉色
        '#7F7F7F'   # 灰色
    ]

    plt.figure(figsize=(12, 5))  # 可以根据需要调整大小
    bars = plt.bar(alg_ls, values, color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 2), ha='center', va='bottom')


    # 添加标题和轴标签
    plt.title('Short-term Temporal Reasoning')
    # plt.legend(loc="upper right")
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Models')
    plt.savefig('exp2_short_time.pdf', format='pdf')
    
def draw_pic_hard(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo']):
    values=[]
    for idx,alg in enumerate(alg_ls):
        last_lines=read_last_lines(alg)
        values.append(float(last_lines[0][0]))
    colors = [
        '#1F77B4',  # 柔和的蓝色
        '#FF7F0E',  # 鲜明的橙色
        '#2CA02C',  # 鲜绿色
        '#D62728',  # 柔和的红色
        '#9467BD',  # 紫色
        '#8C564B',  # 棕色
        '#E377C2',  # 柔和的粉色
        '#7F7F7F'   # 灰色
    ]

    plt.figure(figsize=(12, 5))  # 可以根据需要调整大小
    bars = plt.bar(alg_ls, values, color=colors)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 2), ha='center', va='bottom')
    # 添加标题和轴标签
    plt.title('Long-term Temporal Reasoning')
    # plt.legend(loc="upper right")
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Models')
    plt.savefig('exp2_long_time.pdf', format='pdf')

# exp2('chatglm3-6B')
# exp2('chatglm-turbo')

# exp2('GPT-4-1106')
# exp2('GPT-3.5-turbo')

# exp2('llama3-8b')

# exp2('echo-m1')
# exp2('echo-m2')
# exp2('echo-m3')

draw_pic_easy(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo','llama3-8b','echo-m3'])

draw_pic_hard(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo','llama3-8b','echo-m3'])

