import chaiverse as chai
import os

"""
Goal
- Select the best open-sourced foundation model for our purpose

Agenda
- Evaluate:
    llama 3 8b
    llama 3.1 8b
    mixtral 8x7b
    mistral nemo 13b
    mistral small 22b
- Understanding system prompt
- Submit models to get human-rated feedback & ELO score
"""

def submit_model(params):
    submitter = chai.ModelSubmitter(verbose=True)
    sub_id = submitter.submit(params)
    return sub_id


def get_llama3_formatter():
    llama_3_memory = "<|start_header_id|>system<|end_header_id|>\n\nYou are {bot_name}, write engaging responses\n\n"
    llama_3_bot_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>"
    llama_3_user_template = "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>"
    llama_3_response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:"
    formatter = chai.formatters.PromptFormatter(
            memory_template=llama_3_memory,
            prompt_template='',
            bot_template=llama_3_bot_template,
            user_template=llama_3_user_template,
            response_template=llama_3_response_template,
            truncate_by_message=True,
    )
    return formatter



if __name__ == '__main__':
    generation_params = {
            'frequency_penalty': 0.,
            'presence_penalty': 0.,
            'max_output_tokens': 64,
            'temperature': 1.,
            'top_k': 100,
            'top_p': 1.,
            'stopping_words': ['</s>', 'You:', '\n', '<|eot_id|>'],
            'max_input_tokens': 1024,
            'best_of': 8
            }
    chai.developer_login()
    llama_3_formatter = get_llama3_formatter()
    submission_params = {
            'model_repo': 'meta-llama/Meta-Llama-3-8B',
            'hf_auth_token': os.environ['HF_TOKEN'],
            'formatter': llama_3_formatter,
            'generation_params': generation_params,
            }
    sub_id = submit_model(submission_params)
