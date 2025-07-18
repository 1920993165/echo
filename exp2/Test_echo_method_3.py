# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:03:06 2024

@author: 小怪兽
"""

import json
import os
import copy
from transformers import AutoModel, AutoTokenizer
import torch
import json
import pytz
import re
import mdtex2html
import datetime
import torch
import csv
import json

def get_cur_time(format_t='%Y_%-m_%-d__%H_%M_%S'):
    from datetime import datetime
    # 获得当前时间
    beijing_timezone = pytz.timezone('Asia/Shanghai')                
    # 获取当前时间
    current_utc_time = datetime.utcnow()                
    # 将当前时间转换为北京时间
    current_beijing_time = current_utc_time.replace(tzinfo=pytz.utc).astimezone(beijing_timezone)
    # 格式化输出北京时间
    return current_beijing_time.strftime(format_t)

def find_day(year, month, day):
    # 创建一个日期对象
    date_object = datetime.date(int(year), int(month), int(day))
    # 获取英文的星期几
    weekday = date_object.strftime("%A")
    
    # 将英文星期几转换为对应的汉字数字
    weekday_mapping = {
        "Monday": "一",
        "Tuesday": "二",
        "Wednesday": "三",
        "Thursday": "四",
        "Friday": "五",
        "Saturday": "六",
        "Sunday": "日"
    }
    
    return weekday_mapping.get(weekday, "未知")

def get_time():
    formatted_time = get_cur_time('%Y年%m月%d日%H点%M分%S秒')
    match = re.search(r'(\d+)年(\d+)月(\d+)日.*?(\d+)点(\d+)分(\d+)秒', formatted_time)
    # 提取结果
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))
    sec = int(match.group(6))
    week=find_day(year, month, day)
    return  f'{year}年{month}月{day}日星期{week}{hour}点{minute}分{sec}秒',year,month,day,week,hour,minute,sec


MODEL_PATH = os.environ.get('MODEL_PATH', '/home/lwt/Model/echo26')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
if 'cuda' in DEVICE: # AMD, NVIDIA GPU can use Half Precision
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
else: # CPU, Intel GPU and other GPU can use Float16 Precision Only
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()

#获得prompt
path='../test_data/exp2.json'
with open(path, "r", encoding="utf-8") as file:
    testd = json.load(file)
# print(len(testd))


prompt='Below is a chat record between a user and a robot with episodic memory, which includes (user: user’s question, time record: conversation time, assistant: model’s answer). You will continue to play the role of the episodic memory robot, replying to the user’s new questions.\n\n'
# testd=testd[:5]
count=0
for one in testd:
    print(f'progressing in {count}')
    test_result=copy.deepcopy(one)
    conv=one['conversations']
    length_chat=len(conv)
    user_idx=0
    test_idx=1
    assi_idx=2
    content=''
    history=[]

    timeinfo=get_time()[0]
    assert isinstance(timeinfo,str)
    # print(timeinfo)

    while assi_idx<length_chat:
        user_c=conv[user_idx]['content']
        time_c=conv[test_idx]['content']
        assi_c=conv[assi_idx]['content']
        if conv[test_idx]['test']==1:
            # print('user_c : ',user_c)
            # print('history : ',history)
            response, _ = model.chat(tokenizer,
                                   user_c,
                                   timeinfo=time_c,
                                   history=history,
                                   max_length=8192,
                                   top_p= 0.8,
                                   temperature=0.95)
            # response = general(query, [], 8192, 0.8, 0.95)
            test_result['conversations'][assi_idx]['content']=response
        history.append({'role': 'user', 'content': user_c})
        history.append({'role': 'observation', 'content': time_c})
        history.append({'role': 'assistant', 'metadata': '', 'content': assi_c})
        user_idx+=3
        test_idx+=3
        assi_idx+=3
    if not os.path.exists('echo-m3'):
        os.mkdir('echo-m3')
    #保存结果
    with open("echo-m3/echo-m3-"+str(count)+".json", "w", encoding="utf-8") as file:
        json.dump([test_result], file, indent=4, ensure_ascii=False)
    count+=1
        
        
    



