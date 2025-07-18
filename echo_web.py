import os
from transformers import AutoModel, AutoTokenizer
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

base_url = "http://0.0.0.0:8501"

# MODEL_PATH = os.environ.get('MODEL_PATH', '/home/lwt/Model/echo26')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
# if 'cuda' in DEVICE: # AMD, NVIDIA GPU can use Half Precision
#     model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
# else: # CPU, Intel GPU and other GPU can use Float16 Precision Only
#     model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()

# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)

"""Override Chatbot.postprocess"""

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


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

def find_file_name(prefix,user_start_time,history_b):
    con=history_b[0]['content']
    if len(con)>20:
         con=con[:20]
    re_name=prefix+'/'+user_start_time+con+'.json'
    return re_name

def write2memory_data(user_time,chat_record,history_b,user_start_time):

    #找原本文件
    prefix='memory_data/train27'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    b_path=find_file_name(prefix,user_start_time,history_b)

    #history_b变化
    history_n=[]
    for x in history_b:
        #test [0,1]表示是否是测试点
        #test-time ["just now","one day","few days","one month","few months","one year","few years","several decades"]表示测试的记忆时间跨度
        #test-level ["easy","hard"]表示测试的难度
        # history_n.append({
        #         "role": x['role'],
        #         "test": 0,
        #         "test-time": "",
        #         "test-level": "",
        #         "content": x['content']
        #     })
        if x['role']=='assistant':
            history_n.append({
                    "role": x['role'],
                    "metadata": "",
                    "content": x['content']
                })
        else:
                history_n.append({
                    "role": x['role'],
                    "content": x['content']
                })
    #写brain_data数据
    with open(b_path, 'w', encoding="utf-8") as f:
        new_d={"conversations":history_n}
        json.dump(new_d, f, indent=4, ensure_ascii=False)

def predict(input, chatbot, max_length, top_p, temperature, history, user_start_time, past_key_values):
    if user_start_time is None:
        user_start_time=get_cur_time('%Y年%m月%d日%H时%M分%S秒')
    history+=[
        {'role': 'user', 'content': parse_text(input)}, 
        ]
    timeinfo=get_time()[0]
    chatbot.append((parse_text(input), ""))
    use_stream=True
    data = {
        "functions": None,  # 函数定义
        "model": 'echo',  # 模型名称
        "timeinfo": timeinfo, #当前时间
        "messages": history,  # 会话历史
        "stream": use_stream,  # 是否流式响应
        "max_tokens": 1000,  # 最多生成字数
        "temperature": 0.8,  # 温度
        "top_p": 0.8,  # 采样概率
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    assistant_reply=''
    if response.status_code == 200:
        if use_stream:
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        # print('0')
                        response_json = json.loads(decoded_line)
                        # print('1')
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        # print('2')
                        assistant_reply+=content
                        # print(content)
                        chatbot[-1] = (parse_text(input), parse_text(assistant_reply))
                        # print('3')
                        # yield chatbot, history, user_start_time, past_key_values
                        
                    except Exception as e:
                        # print(e)
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
    else:
        print("Error:", response.status_code)
        return None
    metadata, sub_content = assistant_reply.split("\n", maxsplit=1)
    sub_content=sub_content.strip()
    history+=[
        {'role': 'observation', 'content': timeinfo},
        {'role': 'assistant', 'metadata': metadata, 'content': sub_content}
        ]
    print(sub_content)
    
    user_time=get_cur_time('%Y年%m月%d日%H时%M分%S秒')

    write2memory_data(user_time,response,history,user_start_time)
    chatbot[-1] = (parse_text(input), parse_text(sub_content))
    yield chatbot, history, user_start_time, past_key_values


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Echo</h1>""")
    
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

    user_input.submit(predict, [user_input, chatbot, max_length, top_p, temperature, history, user_start_time, past_key_values],
                    [chatbot, history, user_start_time, past_key_values], show_progress=True)
    user_input.submit(reset_user_input, [], [user_input])
    
    
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, user_start_time, past_key_values],
                    [chatbot, history, user_start_time, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
 

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

demo.queue().launch(share=False, server_name="0.0.0.0", server_port=8502, inbrowser=True)
