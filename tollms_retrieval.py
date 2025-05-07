import json
import os
import requests
from openai import OpenAI
from tqdm import tqdm

def llm_response(type,prompt):
    if type=='yiyan':
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-128k?access_token=" + "24.8700d21025b6c6b4fb7e622afcc217db.2592000.1744019702.282335-117939394"#get_access_token()
        
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "top_p": 0.8,
            "penalty_score": 1,
            "disable_search": True,
            "enable_citation": False,
            "response_format": "text"
        }, ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
        str_chunk_result = json.loads(response)["result"]
        return str_chunk_result
    elif type=='qwen-max':
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key='sk-cf09ff1a0fc147f68cd26ba52f2ec73f', 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-max-latest", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'user', 'content': prompt}
                ],
            temperature=0.7,
            top_p=0.8
            )
        
        return json.loads(completion.model_dump_json())["choices"][0]['message']['content']

ztype='yiyan'  # qwen-max yiyan

print('Start Q&A...')
for filename in os.listdir('chunk1forLLMs_top10'): 
    print('-'*10,filename,'-'*10)
    retrieval_save_list=[]     
    with open(os.path.join('chunk1forLLMs_top10', filename), 'r', encoding='utf-8') as file:  
        data_list = json.load(file)  
    for data in tqdm(data_list):
        try:
            retrieval_prompt=data['retrieval_list']  #.replace("Given the context information and not prior knowledge","Combining context information and prior knowledge")
            llm_ans=llm_response(ztype,retrieval_prompt)
            
            save = {}
            save['_id'] = data['_id']
            save['input'] = data['input']   
            save['llm_ans'] = llm_ans
            save['answers'] = data['answers']
            save['retrieval_list'] = retrieval_prompt
            retrieval_save_list.append(save)
        except:
            pass

        with open(os.path.join('newllm_ans', filename.replace('qwen2_7B_Chunks_300_merge',ztype)), 'w', encoding='utf-8') as json_file:
            json.dump(retrieval_save_list, json_file,indent=4)

# nohup python tollms_retrieval.py >> newllm_ans/two_dataset_yiyan.log 2>&1 &