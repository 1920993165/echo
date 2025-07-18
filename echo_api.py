from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from sse_starlette.sse import EventSourceResponse

app = FastAPI()


# 流式推理
def predict_stream(tokenizer, prompt, history, max_length, top_p, temperature):
    for response, new_history in model.stream_chat(tokenizer,
                                                   prompt,
                                                   history=history,
                                                   max_length=max_length if max_length else 8196,
                                                   top_p=top_p if top_p else 0.8,
                                                   temperature=temperature if temperature else 0.8):
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        yield json.dumps({
            'response': response,
            'history': new_history,
            'status': 200,
            'time': time,
            'sse_status': 1
        })
    log = "[" + time + "] " + "---来自流式推理的消息---" + "prompt:" + prompt + ", response:" + repr(response)
    print(log, flush=True)
    # 推理完成后，发送最后一包数据，sse_statu=2标识sse结束
    yield json.dumps({
        'response': response,
        'history': new_history,
        'status': 200,
        'time': time,
        'sse_status': 2
    })
    return torch_gc()


# 编码
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


# GC回收显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# sse流式方式
@app.post("/chatglm/server/text2text/sse")
async def create_item_sse(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    res = predict_stream(tokenizer, prompt, history, max_length, top_p, temperature)
    return EventSourceResponse(res)


if __name__ == '__main__':
    # cpu/gpu推理，建议GPU，CPU实在是忒慢了
    DEVICE = "cuda"
    DEVICE_ID = "0"
    model_path='/path/to/model/echo'
    CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8501, workers=1)

