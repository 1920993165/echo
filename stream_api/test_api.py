# 使用curl命令测试返回
# curl -X POST "http://172.23.132.104:8501/v1/chat/completions" \
# -H "Content-Type: application/json" \
# -d "{\"model\": \"chatglm3-6b\", \"messages\": [{\"role\": \"user\", \"content\": \"现在几点啦？\"}], \"stream\": false, \"max_tokens\": 100, \"temperature\": 0.8, \"top_p\": 0.8}"

# 使用Python代码测返回
import requests
import json
import datetime
import pytz
import re

base_url = "http://59.78.189.152:9010"

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

def create_chat_completion(model, messages, functions, use_stream=False, timeinfo='None'):
    data = {
        "functions": functions,  # 函数定义
        "model": model,  # 模型名称
        "timeinfo": timeinfo, #当前时间
        "messages": messages,  # 会话历史
        "stream": use_stream,  # 是否流式响应
        "max_tokens": 100,  # 最多生成字数
        "temperature": 0.8,  # 温度
        "top_p": 0.8,  # 采样概率
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    assistant_reply=''
    # print(1)
    if response.status_code == 200:
        # print(2)
        if use_stream:
            # print(3)
            # print(response)
            # 处理流式响应
            for line in response.iter_lines():
                # print(4)
                # print(line)
                if line:

                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        assistant_reply+=content
                        print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
    else:
        print("Error:", response.status_code)
        return None
    return assistant_reply


def simple_chat(use_stream=True):
    functions = None
    #demo1
    chat_messages=[
        {'role': 'user', 'content': '现在几点？'}, {'role': 'observation', 'content': '2024年6月5日星期三10点43分55秒'}, {'role': 'assistant', 'metadata': '', 'content': '现在是10点43分'},
        {'role': 'user', 'content': '今天星期几？'},
        ]
    #demo2
    # chat_messages=[
    #     {'role': 'user', 'content': '帮我记一下我的一篇关于大模型加载的参数介绍及参数推荐表的博客https://blog.csdn.net/a1920993165/article/details/134691021'}, 
    #     {'role': 'observation', 'content': '2024年5月13日星期一20点32分14秒'}, 
    #     {'role': 'assistant', 'metadata': '', 'content': '好的，我记住了。'}, 
    #     {'role': 'user', 'content': '今天星期几？'}, 
    #     ]
    #demo3
    chat_messages=[{'role': 'user', 'content': '现在几点？'}, {'role': 'observation', 'content': '2024年6月5日星期三10点43分55秒'}, {'role': 'assistant', 'metadata': '', 'content': '现在是10点43分'}
                   , {'role': 'user', 'content': '今天星期几？'}, {'role': 'observation', 'content': '2024年6月5日星期三11点12分25秒'}, {'role': 'assistant', 'metadata': '', 'content': '今天星期三'}
                   , {'role': 'user', 'content': '讲个鬼故事'},]
    timeinfo=get_time()[0]
    assistant_reply=create_chat_completion("echo", messages=chat_messages, functions=functions, use_stream=use_stream,timeinfo=timeinfo)
    metadata, sub_content = assistant_reply.split("\n", maxsplit=1)
    sub_content=sub_content.strip()
    chat_messages+=[
        {'role': 'observation', 'content': timeinfo},
        {'role': 'assistant', 'metadata': metadata, 'content': sub_content}
        ]
    print(chat_messages)

if __name__ == "__main__":
    # function_chat(use_stream=True)
    simple_chat(use_stream=True)
