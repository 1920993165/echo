from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import copy
import os

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct")


def llama3_chat(prompt):
    # prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

path='../test_data/testV3.json'
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
            response = llama3_chat(query)
            test_result['conversations'][assi_idx]['content']=response
        user_idx+=3
        test_idx+=3
        assi_idx+=3
        content=content+'assistant:'+assi_c+'\n\n'
    if not os.path.exists('llama3-8b'):
        os.mkdir('llama3-8b')
    #保存结果
    with open("llama3-8b/llama3-8b-"+str(count)+".json", "w", encoding="utf-8") as file:
        json.dump([test_result], file, indent=4, ensure_ascii=False)
    count+=1

