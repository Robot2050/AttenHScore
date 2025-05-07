# Running multiple datasets together  加入重排
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from llms.base import BaseLLM
import os
import torch
import math

class LLM_Chat(BaseLLM):
    def __init__(self, model_name='qwen_7b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = 'models/vicuna-7b-v1.5'   
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto",
                                                     trust_remote_code=True,attn_implementation="eager").eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }

    def request(self, query: str) -> str:
        query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids,pad_token_id=self.tokenizer.eos_token_id, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response
    
    def rerank_prompt(self,retrieval_prompt,query):                       
        response_text_list = retrieval_prompt.split('\n---------------------\n')
        response_text = response_text_list[1].split("\n\n")
        response_list = [text for text in response_text if not text.startswith("file_path: ")]
        
        rrf_rank_ppl=[]
        for index, paragraph in enumerate(response_list):
            gen_question_ppl=self.get_ppl_atten_for_next(self.model,self.tokenizer,paragraph+'\nBased on the above content, generate a question: ',query)
            rrf_rank_ppl.append(gen_question_ppl)
        sorted_lst = sorted(enumerate(rrf_rank_ppl), key=lambda x: x[1])  

        rerank_result=[]
        for i in range(len(response_list)):
            rerank_result.append(response_list[sorted_lst[i][0]])
        
        rerank_prompt=response_text_list[0]+'\n---------------------\n'+'\n\n'.join(rerank_result)+'\n---------------------\n'+response_text_list[2]
        return rerank_prompt
        
    def get_ppl_atten_for_next(self,model,tokenizer,first_sentence,next_sentence):
        tokenized_text_1 = tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
        tokenized_text_2 = tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
        input_ids=torch.cat([tokenized_text_1["input_ids"].to(model.device),tokenized_text_2["input_ids"].to(model.device)],dim=-1)
        attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(model.device),tokenized_text_2["attention_mask"].to(model.device)],dim=-1)
        with torch.no_grad():
            response = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
            )
            # past_key_values = response.past_key_values
        past_length=tokenized_text_1["input_ids"].to(model.device).shape[1]
        shift_logits = response.logits[..., past_length-1:-1, :].contiguous()  #模型的输出logits（即预测的类别分数）
        shift_labels = input_ids[..., past_length : ].contiguous()  #真实的目标标签（即输入ID中的下一个词）。现实中的值
        active = (attention_mask[:, past_length:] == 1).view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)  #使用交叉熵损失函数计算logits和标签之间的损失。
        
        atten = model(input_ids,output_attentions=True).attentions[-1][0]
        next_sentence_len=tokenized_text_2["input_ids"].to(model.device).shape[1]
        solver = "max"
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1:
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        atten_for_output=mean_atten[-next_sentence_len:].cpu().tolist()
        
        ppl_atten=0.0
        loss=loss.cpu().tolist()
        for i,a_loss in enumerate(loss):
            ppl_atten+=a_loss*math.exp(atten_for_output[i])
        return ppl_atten

        


llm = LLM_Chat(model_name='vicuna-7b', temperature=0.1, max_new_tokens=1280)    


    

print('Start Q&A...')
for filename in os.listdir('chunk1forLLMs_top10_test'): 
    print('-'*10,filename,'-'*10)
    retrieval_save_list=[]     
    with open(os.path.join('chunk1forLLMs_top10_test', filename), 'r', encoding='utf-8') as file:  
        data_list = json.load(file)  
    for data in data_list:
        try:                                    
            retrieval_prompt=data['retrieval_list']  
            retrieval_prompt=llm.rerank_prompt(retrieval_prompt,data['input'])
            
            llm_ans=llm.request(retrieval_prompt)
            llm_ans=llm_ans.replace('<|im_end|>','').strip()
            print(data['input'],'\n11111',llm_ans,flush=True)
            
            save = {}
            save['_id'] = data['_id']
            save['input'] = data['input']   
            save['llm_ans'] = llm_ans
            save['answers'] = data['answers']
            save['retrieval_list'] = retrieval_prompt
            retrieval_save_list.append(save)
        except Exception as e:  # 捕获所有其他异常类型
            print(f"发生了一个错误: {e}")

    with open(os.path.join('judge_experiment/other_model', filename.replace('qwen2_7B_Chunks_300_merge','vicuna_7B_QA_2')), 'w') as json_file:
        json.dump(retrieval_save_list, json_file,indent=4)

# CUDA_VISIBLE_DEVICES=6 nohup python tollms_retrieval_2.py >> qa_llms/vicuna_7B_top10_rerank/five_dataset_vicuna_7B_QA.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python tollms_retrieval_2.py >> judge_experiment/other_model/test_vicuna_7B_QA.log 2>&1 &