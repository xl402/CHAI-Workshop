import os
import re

from datasets import load_dataset
from tqdm import tqdm
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, SFTTrainer
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

BASE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"


def load_data(dataset_name):
    ds = load_dataset(dataset_name, split='train')
    return ds


def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = 'left'
    return tokenizer


def get_base_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).to('cuda')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model


def get_data_collator(response_template):
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return collator


if __name__ == '__main__':
    tag = 'Viral-ss-v1'
    ds = load_data(f'ChaiML/{tag}')
    ds = ds.select_columns(['text'])
    tokenizer = get_tokenizer(BASE_MODEL)
    model = get_base_model(BASE_MODEL)
    response_template =  "####\n"
    collator = get_data_collator(response_template)

    res = []
    for row in ds:
        _res = collator.torch_call([tokenizer(row['text'])])
        pct = (_res['labels'] == -100).numpy().mean()
        res.append(pct)
    print((np.array(res) == 1).mean())
