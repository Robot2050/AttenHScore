import argparse
from loguru import logger
from base import BaseRetriever
from embeddings.base import HuggingfaceEmbeddings
import json
import pandas as pd 
from transformers import AutoModelForCausalLM, AutoTokenizer
from llms.base import BaseLLM

class Qwen_7B_Chat(BaseLLM):
    def __init__(self, model_name='qwen_7b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = 'vicuna-7b-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto",
                                                     trust_remote_code=True).eval()
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
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='data/multifieldqa_zh.jsonl', help="Path to the dataset")  
parser.add_argument('--save_file', default='qa_nodie/multifieldqa_zh_qwen2_7B_Chunks.json', help="Path to the answer")

parser.add_argument('--embedding_name', default='BAAI/bge-base-zh-v1.5')
parser.add_argument('--embedding_dim', type=int, default=768)  # BAAI/bge-large-en-v1.5 1024

parser.add_argument('--docs_path', default='data/chunking/multifieldqa_zh_qwen2_7B_Chunks.json', help="Path to the retrieval documents")  
parser.add_argument('--construct_index', action='store_true', help="Whether to construct an index")
parser.add_argument('--add_index', action='store_true', default=False, help="Whether to add an index")
parser.add_argument('--collection_name', default="multifieldqa_zh_qwen2_7B_Chunks", help="Name of the collection")   
parser.add_argument('--retrieve_top_k', type=int, default=5, help="Top k documents to retrieve")

args = parser.parse_args()
logger.info(args)

llm = Qwen_7B_Chat(model_name='qwen_7b', temperature=0.1, max_new_tokens=1280)
embed_model = HuggingfaceEmbeddings(model_name=args.embedding_name)
print('Finish Loading...')
retriever = BaseRetriever(
        args.docs_path, embed_model=embed_model, embed_dim=args.embedding_dim,
        construct_index=args.construct_index, add_index=args.add_index,
        collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k
    )

print('Finish Indexing...')
retrieval_save_list=[]
with open(args.data_path, 'r', encoding='utf-8') as file:  
    # 逐行读取  
    for line in file: 
        data = json.loads(line) 
        try:
            retrieval_prompt=retriever.search_docs(data['input'])
            llm_ans=llm.request(retrieval_prompt)
            
            save = {}
            save['_id'] = data['_id']
            save['input'] = data['input']   
            save['llm_ans'] = llm_ans
            save['answers'] = data['answers']
            save['retrieval_list'] = retrieval_prompt
            retrieval_save_list.append(save)
            # if index==1:
            #     break
        except:
            pass

with open(args.save_file, 'w') as json_file:
    json.dump(retrieval_save_list, json_file,indent=4)
    
# CUDA_VISIBLE_DEVICES=7 nohup python retrieval.py --construct_index >> qa_nodie/2wikimqa_qwen15B_nodie_top5.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python retrieval.py >> qa_nodie/2wikimqa_qwen15B_nodie_top5.log 2>&1 &