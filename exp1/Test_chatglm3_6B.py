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


def general(inputs, history_l, max_length, top_p, temperature):
    
    url_l = "http://0.0.0.0:8000"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "prompt": inputs,
        "history": history_l,
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature,
    }
    res_language = requests.post(url_l, headers=headers, json=data)
    assert res_language.status_code==200
    response_data = res_language.json()
    response_text = response_data.get('response', '')
    history_l = response_data.get('history', '')

    return response_text

#获得prompt

path='test_data/testV3.json'
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
    while assi_idx<length_chat:
        user_c=conv[user_idx]['content']
        time_c=conv[test_idx]['content']
        assi_c=conv[assi_idx]['content']
        content=content+'user:'+user_c+'\n\n'
        content=content+'time record:'+time_c+'\n\n'
        if conv[test_idx]['test']==1:
            query=prompt+content+'assistant:'
            response = general(query, [], 8192, 0.8, 0.95)
            test_result['conversations'][assi_idx]['content']=response
        user_idx+=3
        test_idx+=3
        assi_idx+=3
        content=content+'assistant:'+assi_c+'\n\n'
    #保存结果
    with open("chatglm3-6B/chatglm3-6B-"+str(count)+".json", "w", encoding="utf-8") as file:
        json.dump([test_result], file, indent=4, ensure_ascii=False)
    count+=1
        
        
    



