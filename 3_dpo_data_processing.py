from datasets import load_dataset


"""
Goal
- Format a dataset correctly for Direct Preference Optimization training

Agenda
- Take a look at the dataset, what does it look like?
- Format the conversation into target input and output format

----------------------------------------------------------------
Assistant: Hello, how are you doing today?
User: I'm doing well, thank you. How are you?
Assistant: I'm doing well, thank you. How can I help you today?
User: What is 1+1?

Chosen:
Assistant: 2!

Rejected:
Assistant: 42!
-----------------------------------------------------------------
"""


def print_color(text, color):
    colors = {'blue': '\033[94m',
              'cyan': '\033[96m',
              'green': '\033[92m',
              'yellow': '\033[93m',
              'red': '\033[91m'}
    assert color in colors.keys()
    print(f'{colors[color]}{text}\033[0m')


def load_data(dataset_name):
    ds = load_dataset(dataset_name, split='train')
    return ds.to_pandas().iloc[1:].reset_index(drop=True)


def sample(df):
    row = df.sample(1).iloc[0]
    chat_history = row.conversation_state['chat_history']
    for chat in chat_history[:-1]:
        sender, message = chat['sender'], chat['message']
        color = 'yellow' if sender == 'You' else 'cyan'
        print_color(f'{sender}: {message}', color)
    print_color(f"Chosen: {row.chosen_response}", 'green')
    print_color(f"Rejected: {row.rejected_response}", 'red')


def format_payload(row):
    chat_history = row.conversation_state['chat_history']
    formatted_chat_history = '\n'.join([f"{m['sender']}: {m['message'].strip()}" for m in chat_history[:-1]])
    chosen = row.chosen_response
    rejected = row.rejected_response
    prompt = formatted_chat_history[-5000:] + '\n####\n'
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}


if __name__ == '__main__':
    df = load_data('ChaiML/public_preference_data_20k')
