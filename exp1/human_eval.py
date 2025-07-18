import os
import gradio as gr
import pytz
import re
import mdtex2html
import datetime
import torch
import csv
import json
import requests
import json
import datetime
import pytz
import re

path='exp1_human.json'
with open(path, "r", encoding="utf-8") as file:
    testd = json.load(file)

#计算有哪些没有处理的问题

human_dirs=os.listdir('human')



def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

#获得chatglm3结果
def get_conv(fn,idx):
    path=f'{fn}/{fn}-{idx}.json'
    with open(path, "r", encoding="utf-8") as file:
        conv = json.load(file)
    return conv


def get_model_res(fn_list,idx,assi_idx):
    tt=['模型A的输出:\n{}\n','模型B的输出:\n{}\n','模型C的输出:\n{}\n','模型D的输出:\n{}\n','模型E的输出:\n{}\n','模型F的输出:\n{}\n','模型G的输出:\n{}\n']
    res=''
    for count,fn in enumerate(fn_list):
        t_conv=get_conv(fn,idx)
        t_conv=t_conv[0]['conversations']
        # print(fn,t_conv[0])
        # print(len(t_conv))
        sent=t_conv[assi_idx]['content']
        res=res+tt[count].format(sent)
    
    return res

def save_json(file_path,data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f'Save {file_path} is ok!')


def get_histroy(conv,test_idx):
    i=0
    his=[]
    while i <=test_idx:
        one= conv[i]
        if one['role']=='user':
            his+=[(parse_text(one['content']), parse_text(''))]
        elif one['role']=='assistant':
            his[-1]=(his[-1][0], parse_text(one['content']))
        elif one['role']=='observation':
            his[-1]=(his[-1][0]+'\n'+one['content'], parse_text(''))
        else:
            print(one)
            1/0
        i+=1

    return his

def is_ok(res,fn_list):
    if len(res.split(','))==len(fn_list):
        return True
    return False

# def list2tuple(chatbot):
#     new_chat=[]
#     for one in chatbot:
#         print(one)
#         new_chat+=[(one[0],one[1])]
#     return new_chat

def predict(inputs, chatbot, max_length, top_p, temperature, history, user_start_time, past_key_values,stage,pro_data):
    fn_list=['chatglm3-6B','llama3-8b','chatglm-turbo','echo-m3','GPT-3.5-turbo','GPT-4-1106']
    if stage==0:
        for idx,one in enumerate(testd):
            conv=one['conversations']
            length_chat=len(conv)
            user_idx=0
            test_idx=1
            assi_idx=2
            while assi_idx<length_chat:
                print(test_idx)
                if conv[test_idx]['test']==1:
                    save_path=f'human/{str(idx)}-{str(test_idx)}.json'
                    if os.path.exists(save_path):
                        user_idx+=3
                        test_idx+=3
                        assi_idx+=3
                        continue
                    else:
                        print(f'{save_path} is not exist!')
                        res=get_model_res(fn_list,idx,assi_idx)
                        prompt=f'''下面是各个模型的输出，请给这些模型进行打分，分数范围为0-10\n{res}\n\n\n\n请严格按照下面的模板输入分数，例子如下：\n4,3,5,8,2,7'''
                        test_time=conv[test_idx]['test-time']
                        test_level=conv[test_idx]['test-level']
                        one['test-time']=test_time
                        one['test-level']=test_level
                        one['response']=res
                        pro_data=one
                        pro_data['save_path']=save_path
                        pro_data['idx']=str(idx)+str(test_idx)
                        chatbot=get_histroy(conv,test_idx)
                        # chatbot=list2tuple(chatbot)
                        chatbot[-1] = (chatbot[-1][0], parse_text(prompt))

                        stage+=1

                        return chatbot, history, user_start_time, past_key_values, stage, pro_data
                user_idx+=3
                test_idx+=3
                assi_idx+=3
                
    elif stage==1:
        inputs=inputs.strip()
        print(repr(inputs))
        if is_ok(inputs,fn_list):
            pro_data['score']=inputs
            save_json(pro_data['save_path'],pro_data)
            prompt=f'{pro_data["idx"]} 打分完毕，请刷新页面打分下一个对话哈'
            chatbot[-1] = (inputs, parse_text(prompt))
            stage=0
            pro_data=None
        else:
            prompt=f'''请严格按照下面的模板输入分数，例子如下：\n4,3,5,8,2,7'''
            print(inputs)
            chatbot[-1] = (inputs, parse_text(prompt))
    

    return chatbot, history, user_start_time, past_key_values, stage, pro_data


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None, 0, None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Echo Evaluation under human </h1>""")
    
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Shift + Enter 换行, Enter 提交",scale=9)#.style(container=False)
        submitBtn = gr.Button("Submit", variant="primary",scale=1)

    with gr.Row():     
        emptyBtn = gr.Button("Clear History")
        max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
        top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    user_start_time=gr.State(None)
    history = gr.State([])
    past_key_values = gr.State(None)    


    stage=gr.State(0)
    pro_data=gr.State(None)

    user_input.submit(predict, [user_input, chatbot, max_length, top_p, temperature, history, user_start_time, past_key_values,stage,pro_data],
                    [chatbot, history, user_start_time, past_key_values,stage,pro_data], show_progress=True)
    user_input.submit(reset_user_input, [], [user_input])
    
    
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, user_start_time, past_key_values,stage,pro_data],
                    [chatbot, history, user_start_time, past_key_values,stage,pro_data], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
 

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values,stage,pro_data], show_progress=True)

demo.queue().launch(share=True, server_name="0.0.0.0", server_port=9010, inbrowser=True)