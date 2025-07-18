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



def statistic():
    path='../test_data/exp1.json'
    with open(path, "r", encoding="utf-8") as file:
        testd = json.load(file)
    test_time_dict_easy={}
    test_time_dict_hard={}
    
    dis=os.listdir('human')
    print(len(dis))

    1/0
        
    for idx,one in enumerate(testd):
        conv=one['conversations']
        length_chat=len(conv)
        user_idx=0
        test_idx=1
        assi_idx=2
        while assi_idx<length_chat:
            user_c=conv[user_idx]['content']
            time_c=conv[test_idx]['content']
            assi_c=conv[assi_idx]['content']
            if conv[test_idx]['test']==1:
                test_time=conv[test_idx]['test-time']
                test_level=conv[test_idx]['test-level']
                if test_level=='easy':
                    if test_time not in test_time_dict_easy:
                        test_time_dict_easy[test_time]=1
                    else:
                        test_time_dict_easy[test_time]+=1
                else:
                    if test_time not in test_time_dict_hard:
                        test_time_dict_hard[test_time]=1
                    else:
                        test_time_dict_hard[test_time]+=1
            user_idx+=3
            test_idx+=3
            assi_idx+=3



    # 假设的示例数据

    # 写入CSV文件
    csv_file_path = f"statistic_information.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["time", "easy", "hard",'all'])
        # 写入数据
        all_easy=0
        all_hard=0
        for key,item in test_time_dict_easy.items():
            hard_n=test_time_dict_hard[key]
            all_easy+=item
            all_hard+=hard_n
            writer.writerow([key, item, hard_n ,item+hard_n])

        writer.writerow(['all', all_easy, all_hard ,all_easy+all_hard])


labels=["just now","one day","few days","one month","few months","one year","few years","several decades",'mean value']

statistic()
