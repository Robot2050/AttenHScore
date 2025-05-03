from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from perplexity_chunking import Judgeing
import math
import torch

model_name_or_path= 'model/vicuna-7b-v1.5'
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map,attn_implementation="eager",torch_dtype=torch.bfloat16)  
model.eval()
Jg=Judgeing(model, tokenizer)

with open('other_model/multifieldqa_zh_vicuna_7B_QA_top10.json', 'r', encoding='utf-8') as file:  
    qa_data1 = json.load(file)
with open('newllm_ans/multifieldqa_zh_qwen2_A14B_QA_top10.json', 'r', encoding='utf-8') as file:  
    qa_data2 = json.load(file)
    
filename ="judge_merge/atten/multifieldqa_zh_top10_qwen2_A14B.json" 
call_percentage=0.4

call_time=math.ceil(len(qa_data1)*call_percentage)
if_call_model=0
list_for_threshold=[] 
# threshold=3
# threshold_max=0.3

retrieval_save_list=[]
judge_list=[]
for i in range(len(qa_data1)):
    try:
        if if_call_model<call_time:
            first_sentence=qa_data1[i]["retrieval_list"]
            next_sentence=qa_data1[i]["llm_ans"]
            atten_score=Jg.get_ppl_for_next(first_sentence,next_sentence,entropy_chunk_size=10)
            # atten_score=Jg.get_perplexity_score(first_sentence,next_sentence)
            # atten_score=Jg.get_range_avg_score(first_sentence,next_sentence)
            if len(list_for_threshold)>5:
                threshold=sum(list_for_threshold)/len(list_for_threshold)
                if atten_score>=threshold: #atten 0.15  ppl 0.07
                    if_call_model+=1
                    llm_ans=qa_data2[i]["llm_ans"]
                    judge_list.append(1)
                else:
                    llm_ans=qa_data1[i]["llm_ans"]
                    judge_list.append(0)
            else:
                llm_ans=qa_data1[i]["llm_ans"]
                judge_list.append(0)
        
            list_for_threshold.append(atten_score)

            save = {}
            save['_id'] = qa_data1[i]['_id']
            save['input'] = qa_data1[i]['input']   
            save['llm_ans'] = llm_ans
            save['answers'] = qa_data1[i]['answers']
            save['retrieval_list'] = qa_data1[i]["retrieval_list"]
            save['atten_score'] = atten_score
            retrieval_save_list.append(save)
            print('call model: ',if_call_model,flush=True)
            
        else:
            judge_list.append(0)
            save = {}
            save['_id'] = qa_data1[i]['_id']
            save['input'] = qa_data1[i]['input']   
            save['llm_ans'] = qa_data1[i]['llm_ans']  
            save['answers'] = qa_data1[i]['answers']
            save['retrieval_list'] = qa_data1[i]["retrieval_list"]
            retrieval_save_list.append(save)
    except:
        judge_list.append(2)
        print('CUDA out of memory',flush=True)
    with open(filename, 'w', encoding='utf-8') as sfile:
        json.dump(retrieval_save_list, sfile, ensure_ascii=False, indent=4)
with open(filename.replace('multifieldqa_zh','judge_multifieldqa_zh'), 'w', encoding='utf-8') as sfile:
        json.dump(judge_list, sfile, ensure_ascii=False, indent=4)
print(filename,len(qa_data1),call_time,if_call_model)
    
# CUDA_VISIBLE_DEVICES=1,3 nohup python judge.py >> judge_merge/atten/multifieldqa_zh_top10_qwen2_A14B.log 2>&1 &