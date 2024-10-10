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
- Submit models to get human-rated feedback & ELO score
"""

def submit_model(params):
    submitter = chai.ModelSubmitter(verbose=True)
    sub_id = submitter.submit(params)
    return sub_id


if __name__ == '__main__':
    chai.developer_login()

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
    hf_repos = [
        'meta-llama/Meta-Llama-3-8B',
        'meta-llama/Llama-3.1-8B-Instruct',
        'mistralai/Mistral-Nemo-Instruct-2407',
        'mistralai/Mistral-Small-Instruct-2409'

    ]
    for model_repo in hf_repos:
        submission_params = {
                'model_repo': model_repo,
                'hf_auth_token': os.environ['HF_TOKEN'],
                'formatter': chai.formatters.VicunaFormatter(),
                'generation_params': generation_params,
                }
        sub_id = submit_model(submission_params)
