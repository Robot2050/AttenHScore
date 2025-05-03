# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import MODEL_PATH


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float32):
    if model_name.startswith('facebook/opt'):
        model = OPTForCausalLM.from_pretrained(MODEL_PATH+model_name.split("/")[1], torch_dtype=torch_dtype)
    elif model_name == "microsoft/deberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")#, torch_dtype=torch_dtype)
    if model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf' or model_name == "Llama-2-13b-chat-hf" or model_name == "Meta-Llama-3-8B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype) #attn_implementation="eager"
    if model_name == "falcon-7b":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, trust_remote_code=True, torch_dtype=torch_dtype)
    if model_name == 'vicuna-7b-v1.5':
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype)
    # elif "opt" in model_name:
    #     model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype)
    elif model_name == 'roberta-large-mnli':
         model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")#, torch_dtype=torch_dtype)
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name.startswith('facebook/opt-'):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH+model_name.split("/")[1], use_fast=use_fast)
    elif model_name == "microsoft/deberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf' or model_name == "Llama-2-13b-chat-hf" or model_name == "Meta-Llama-3-8B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, use_fast=use_fast)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == "falcon-7b":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name), trust_remote_code=True, cache_dir=None, use_fast=use_fast)
    elif model_name == 'vicuna-7b-v1.5':
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name), trust_remote_code=True, cache_dir=None, use_fast=use_fast)
    return tokenizer