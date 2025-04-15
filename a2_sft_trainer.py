import os
import re

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import pipeline
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import numpy as np
import torch


"""
Goal
- Train a model with SFT and verify indeed it has learnt the dataset

Agenda
- What is data collator? How do I know this works?
- What is LoRA? How do I use this?
- Key hyper parameters, how do I calculate my batch size?
- How do I know my model was trained correctly?
"""


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


def get_base_model(model_id, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to('cuda')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model


def get_data_collator(response_template):
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return collator


def verify_data_collator(dataset, collator, text=None):
    res = []
    for row in dataset:
        _res = collator.torch_call([tokenizer(row['text'])])
        pct = (_res['labels'] == -100).numpy().mean()
        res.append(pct)
    print((np.array(res) == 1).mean())
    if text is not None:
        print(collator.torch_call([tokenizer(text)]))


def print_trainable_parameters(model):
    size = 0
    for name, param in model.named_parameters():
      if param.requires_grad:
          size += param.size().numel()
    print(f'Total number of trainable parameters: {size // 1e6} million')


def get_lora_base_model(model, lora_config):
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    return model


if __name__ == '__main__':
    BASE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
    MODEL_NAME = "WorkshopSFT"

    # Load dataset
    train_dataset = load_data(f'ChaiML/horror_data_formatted')
    train_dataset = train_dataset.select_columns(['text'])

    # Load tokenizer and base model
    tokenizer = get_tokenizer(BASE_MODEL)
    model = get_base_model(BASE_MODEL, tokenizer)

    # Define data collator for I/O training
    response_template =  "####\n"
    collator = get_data_collator(response_template)
    verify_data_collator(train_dataset, collator, "what is 1+1?\n####\nAssistant: 42!")

    # Load lora model
    lora_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_lora_base_model(model, lora_config)
    print_trainable_parameters(model)

    # Train model
    training_args = TrainingArguments(
        num_train_epochs=4,
        learning_rate=1e-05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        do_eval=True,
        per_device_eval_batch_size=1,
        adam_epsilon=1e-08,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1,
        seed=42,
        logging_steps=10,
        save_steps=1,
        eval_steps=50,
        save_strategy="epoch",
        output_dir=f"data/{MODEL_NAME}",
        hub_model_id="dpo",
        gradient_checkpointing=True,
        bf16=True,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=1600,
        dataset_text_field="text",
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model()

    # Push to Hub
    trained_model = model.merge_and_unload()
    tokenizer.push_to_hub(f'ChaiML/{MODEL_NAME}', private=True)
    trained_model.push_to_hub(f'ChaiML/{MODEL_NAME}', private=True)

    # Trained model verification -> Did it memorize the corpus?
    text_generator = pipeline("text-generation", model=trained_model.half(), tokenizer=tokenizer)
    prompt, expected_response = train_dataset['text'][0].split('\n####\n')
    generated_text = text_generator(
            prompt+'\n####\n',
            max_new_tokens=200,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id)
    generated_text = generated_text[0]['generated_text'].split('\n####\n')[1]
    print(f'Expected response: {expected_response}')
    print(f'Generated response: {generated_text}')
