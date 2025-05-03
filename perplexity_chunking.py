import torch
import torch.nn.functional as F
import numpy as np
import math
from nltk.tokenize import sent_tokenize
import jieba 

class Judgeing:
    def __init__(self, model, tokenizer) -> None:
        self.model=model
        self.tokenizer=tokenizer

    def get_ppl_for_next(self,first_sentence,next_sentence,past_key_values=None,entropy_chunk_size=10):
        tokenized_text_1 = self.tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
        tokenized_text_2 = self.tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
        input_ids=torch.cat([tokenized_text_1["input_ids"].to(self.model.device),tokenized_text_2["input_ids"].to(self.model.device)],dim=-1)
        attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(self.model.device),tokenized_text_2["attention_mask"].to(self.model.device)],dim=-1)
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        past_length=tokenized_text_2["input_ids"].to(self.model.device).shape[1]
        shift_logits = response.logits[..., -past_length-1:-1, :].contiguous()  #模型的输出logits（即预测的类别分数）
        token_probs = F.softmax(shift_logits, dim=-1)
        
        scores=[]
        for i,token_id in enumerate(tokenized_text_2["input_ids"][0]):
            prob=token_probs[0][i, token_id.item()].item()
            # print(i,token_id.item(),prob)
            scores.append(prob)

        atten = self.model(input_ids,output_attentions=True).attentions[-1][0]
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
        atten_for_output=mean_atten[-past_length:].cpu().tolist()
        
        def get_entropy(zscores,zatten_for_output):
            entropy = 0.0
            for i,conf in enumerate(zscores):
                each_atten_score=conf*np.log(conf)*math.exp(zatten_for_output[i])
                entropy += each_atten_score
            entropy = -1.0 * entropy
            return entropy
        entropy_chunk_size=entropy_chunk_size
        all_entropy=[]
        for i in range(0, len(scores), entropy_chunk_size):
            e_scores = scores[i:i + entropy_chunk_size]
            e_atten = atten_for_output[i:i + entropy_chunk_size]
            all_entropy.append(get_entropy(e_scores,e_atten))
        entropy=max(all_entropy)     
        
        
        return entropy

    def get_perplexity_score(self,first_sentence,next_sentence,past_key_values=None):
        tokenized_text_1 = self.tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
        tokenized_text_2 = self.tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
        input_ids=torch.cat([tokenized_text_1["input_ids"].to(self.model.device),tokenized_text_2["input_ids"].to(self.model.device)],dim=-1)
        attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(self.model.device),tokenized_text_2["attention_mask"].to(self.model.device)],dim=-1)
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        past_length=tokenized_text_2["input_ids"].to(self.model.device).shape[1]
        shift_logits = response.logits[..., -past_length-1:-1, :].contiguous()  #模型的输出logits（即预测的类别分数）
        token_probs = F.softmax(shift_logits, dim=-1)
        
        scores=[]
        for i,token_id in enumerate(tokenized_text_2["input_ids"][0]):
            prob=token_probs[0][i, token_id.item()].item()
            # print(i,token_id.item(),prob)
            scores.append(prob)
        perplexity = 0.0
        for conf in scores:
            perplexity += np.log(conf)
        perplexity = -1.0 * perplexity/len(scores)
        return perplexity
        
    def get_range_avg_score(self,first_sentence,next_sentence,past_key_values=None):
        tokenized_text_1 = self.tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
        tokenized_text_2 = self.tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
        input_ids=torch.cat([tokenized_text_1["input_ids"].to(self.model.device),tokenized_text_2["input_ids"].to(self.model.device)],dim=-1)
        attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(self.model.device),tokenized_text_2["attention_mask"].to(self.model.device)],dim=-1)
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        past_length=tokenized_text_2["input_ids"].to(self.model.device).shape[1]
        shift_logits = response.logits[..., -past_length-1:-1, :].contiguous()  #模型的输出logits（即预测的类别分数）
        token_probs = F.softmax(shift_logits, dim=-1)

        ranges=[]
        for logits in token_probs[0]:
            values, indices = torch.topk(logits, k=2, dim=-1)
            range_top2=torch.abs(torch.abs(values[0] - values[1])).cpu().item()
            ranges.append(range_top2)
        range_avg=sum(ranges)/len(ranges)
        return -range_avg
    
    def split_text_by_punctuation(self,text,language): 
        if language=='zh': 
            sentences = jieba.cut(text, cut_all=False)  
            sentences_list = list(sentences)  
            sentences = []  
            temp_sentence = ""  
            for word in sentences_list:  
                if word in ["。", "！", "？","；"]:  
                    sentences.append(temp_sentence.strip()+word)  
                    temp_sentence = ""  
                else:  
                    temp_sentence += word  
            if temp_sentence:   
                sentences.append(temp_sentence.strip())  
            
            return sentences
        else:
            full_segments = sent_tokenize(text)
            ret = []
            for item in full_segments:
                item_l = item.strip().split(' ')
                if len(item_l) > 512:
                    if len(item_l) > 1024:
                        item = ' '.join(item_l[:256]) + "..."
                    else:
                        item = ' '.join(item_l[:512]) + "..."
                ret.append(item)
            return ret
    def get_sentence_ppl_for_next(self,first_sentence,next_sentence,language='en',past_key_values=None):
        tokenized_text_1 = self.tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)

        segments = self.split_text_by_punctuation(next_sentence,language)
        segments = [item for item in segments if item.strip()]  
        len_sentences=[]
        input_ids_2=torch.tensor([[]], device=self.model.device,dtype=torch.long)  
        attention_mask_2 =torch.tensor([[]], device=self.model.device,dtype=torch.long)  
        for context in segments:
            tokenized_text = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
            input_id = tokenized_text["input_ids"].to(self.model.device)
            input_ids_2 = torch.cat([input_ids_2, input_id],dim=-1)
            len_sentences.append(input_id.shape[1])
            attention_mask_tmp = tokenized_text["attention_mask"].to(self.model.device)
            attention_mask_2 = torch.cat([attention_mask_2, attention_mask_tmp],dim=-1)
        
        input_ids=torch.cat([tokenized_text_1["input_ids"].to(self.model.device),input_ids_2],dim=-1)
        attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(self.model.device),attention_mask_2],dim=-1)
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        
        past_length=sum(len_sentences)
        shift_logits = response.logits[..., -past_length-1:-1, :].contiguous()  #模型的输出logits（即预测的类别分数）
        token_probs = F.softmax(shift_logits, dim=-1)
        
        scores=[]
        for i,token_id in enumerate(input_ids_2[0]):
            prob=token_probs[0][i, token_id.item()].item()
            scores.append(prob)

        atten = self.model(input_ids,output_attentions=True).attentions[-1][0]
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
        atten_for_output=mean_atten[-past_length:].cpu().tolist()
        
        def get_entropy(zscores,zatten_for_output):
            entropy = 0.0
            for i,conf in enumerate(zscores):
                each_atten_score=conf*np.log(conf)*math.exp(zatten_for_output[i])
                entropy += each_atten_score
            entropy = -1.0 * entropy
            return entropy

        all_entropy=[]
        index=0
        for i in range(len(len_sentences)):
            all_entropy.append(get_entropy(scores[index:index+len_sentences[i]],atten_for_output[index:index+len_sentences[i]]))
            index+=len_sentences[i]
        entropy=max(all_entropy)    
        print('all_entropy: ',all_entropy)   
 
        return entropy