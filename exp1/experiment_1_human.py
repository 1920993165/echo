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

def get_data(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def exp1_human(index):
    similarity_ls = []
    test_time_ls = []
    test_level_ls = []
    test_time_dict_easy={}
    test_time_dict_hard={}
    dirs=os.listdir('human')
    for fn in dirs:
        fn_path=f'human/{fn}'
        data=get_data(fn_path)
        test_time=data['test-time']
        test_level=data['test-level']
        score=data['score']
        score=score.split(',')[index]
        score=int(score)
        similarity_ls.append(score)
        test_time_ls.append(test_time)
        test_level_ls.append(test_level)
        if test_level=='easy':
            if test_time in test_time_dict_easy:
                test_time_dict_easy[test_time].append(score)
            else:
                test_time_dict_easy[test_time]=[score,]
        else:
            if test_time in test_time_dict_hard:
                test_time_dict_hard[test_time].append(score)
            else:
                test_time_dict_hard[test_time]=[score,]
        
    # for idx,one in enumerate(testd):
    #     print(f'progressing in {count}')
    #     conv=one['conversations']
    #     length_chat=len(conv)
    #     user_idx=0
    #     test_idx=1
    #     assi_idx=2
    #     chatglm3_conv=get_conv(fn,idx)
    #     chatglm3_conv=chatglm3_conv[0]['conversations']
    #     while assi_idx<length_chat:
    #         user_c=conv[user_idx]['content']
    #         time_c=conv[test_idx]['content']
    #         assi_c=conv[assi_idx]['content']
    #         if conv[test_idx]['test']==1:
    #             #获得
    #             sent1=assi_c
    #             sent2=chatglm3_conv[assi_idx]['content']
    #             simi=count_similarity(sent1,sent2)
    #             test_time=conv[test_idx]['test-time']
    #             test_level=conv[test_idx]['test-level']
    #             sent1_ls.append(sent1)
    #             sent2_ls.append(sent2)
    #             similarity_ls.append(simi)
    #             test_time_ls.append(test_time)
    #             test_level_ls.append(test_level)
    #             if test_level=='easy':
    #                 if test_time in test_time_dict_easy:
    #                     test_time_dict_easy[test_time].append(simi)
    #                 else:
    #                     test_time_dict_easy[test_time]=[simi,]
    #             else:
    #                 if test_time in test_time_dict_hard:
    #                     test_time_dict_hard[test_time].append(simi)
    #                 else:
    #                     test_time_dict_hard[test_time]=[simi,]
    #             # print('similarity',simi)
    #         user_idx+=3
    #         test_idx+=3
    #         assi_idx+=3
    #     count+=1



    # 假设的示例数据

    # 写入CSV文件
    csv_file_path = f"human_{index}.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["score", "test-time", "test-level"])
        # 写入数据
        for i in range(len(similarity_ls)):
            writer.writerow([similarity_ls[i], test_time_ls[i], test_level_ls[i]])
        
        # 写入总数据
        alltime=["jn","od","fd","om","fm","oy","fy","sd"]
        result_easy=[]
        result_hard=[]
        for one in alltime:
            lst_easy=test_time_dict_easy[one]
            result_easy.append(sum(lst_easy)/len(lst_easy))
            lst_hard=test_time_dict_hard[one]
            result_hard.append(sum(lst_hard)/len(lst_hard))
        writer.writerow([one+'-easy' for one in alltime]+['mean-easy'])
        writer.writerow(result_easy+[sum(result_easy)/len(result_easy)])
        writer.writerow([one+'-hard' for one in alltime]+['mean-hard'])
        writer.writerow(result_hard+[sum(result_hard)/len(result_hard)])

# for index in range(0,6):
#     exp1_human(index)

import csv
import os
from itertools import islice

def read_last_lines(fn, num_lines=4):
    file_path=f"{fn}.csv"
    with open(file_path, 'r') as file:
        # 跳过文件的前面部分，直接读取最后几行
        reverse_file = reversed(list(csv.reader(file)))
        last_lines = [row for row in islice(reverse_file, num_lines)]
    return last_lines

labels=["just now","one day","few days","one month","few months","one year","few years","several decades",'mean value']

aa=['human_0','human_1','human_2','human_3','human_4','human_5']
fn_list=['chatglm3-6B','llama3-8b','chatglm-turbo','echo-m3','GPT-3.5-turbo','GPT-4-1106']
mapp=dict()
for idx,x in enumerate(aa):
    mapp[x]=fn_list[idx]

def draw_pic_easy(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo']):
    plt.figure(figsize=(12, 5))  # 可以根据需要调整大小
    for idx,alg in enumerate(alg_ls):
        last_lines=read_last_lines(alg)
        values=last_lines[2]
        values=[float(one) for one in values]
        plt.plot(labels, values, marker='o',label=mapp[alg])  # 使用圆形标记

    # 添加标题和轴标签
    plt.title('Easy Episodic Memory')
    # plt.legend(loc="upper right")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Time scale')
    plt.savefig('exp1_easy_human.pdf', format='pdf')
    
def draw_pic_hard(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo']):
    plt.figure(figsize=(12, 5))  # 可以根据需要调整大小
    for idx,alg in enumerate(alg_ls):
        last_lines=read_last_lines(alg)
        values=last_lines[0]
        values=[float(one) for one in values]
        plt.plot(labels, values, marker='o',label=mapp[alg])  # 使用圆形标记

    # 添加标题和轴标签
    plt.title('Hard Episodic Memory')
    # plt.legend(loc="upper right")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Time scale')
    plt.savefig('exp1_hard_human.pdf', format='pdf')
# exp1('chatglm3-6B')
# exp1('chatglm-turbo')

# exp1('GPT-4-1106')
# exp1('GPT-3.5-turbo')

# exp1('llama3-8b')

# exp1('echo-m1')
# exp1('echo-m2')
# exp1('echo-m3')

# draw_pic_easy(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo','llama3-8b','echo-m3'])

# draw_pic_hard(alg_ls=['chatglm3-6B','chatglm-turbo','GPT-4-1106','GPT-3.5-turbo','llama3-8b','echo-m3'])

draw_pic_easy(alg_ls=['human_0','human_1','human_2','human_3','human_4','human_5'])

draw_pic_hard(alg_ls=['human_0','human_1','human_2','human_3','human_4','human_5'])