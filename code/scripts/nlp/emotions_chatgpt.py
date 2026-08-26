import pandas as pd
import time
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import openai
import json
import os
import sys

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

zero_shot_prompt = '''Koja emocija je izrazena u datom text-u? Vrati jednu od emocija: radost, tuga, poverenje, gadjenje, strah, bes, iznenadjenje, anticipacija
text: {}
'''''

COMPLETIONS_MODEL = "gpt-5-mini"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = openai.OpenAI()

def format_prompt(line):
    return zero_shot_prompt.format(line)

def classification(line):
    prompt = format_prompt(line)
    messages = [
        {"role": "system", "content": prompt},
    ]
    completion_response = openai.chat.completions.create(
                            messages=messages,
                            #temperature=0,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            model=COMPLETIONS_MODEL)
    label = completion_response.choices[0].message.content.replace('\n','')
    return label

def save_file(response, output_file):
    with open(output_file, 'a') as f:
        for line in response:
            f.write(json.dumps(line))
            f.write('\n')

if __name__ == "__main__":
    out_file = '../../results/emotions_chatgpt.jsonl'
    if os.path.exists('../../results/emotions_chatgpt.jsonl'):
        df = pd.read_json('../../results/emotions_chatgpt.jsonl', lines=True)
        df['id_str'] = df['id_str'].astype('str')
        records = list(df['id_str'].unique())
        
        data = pd.read_csv('../../data/np_without_duplicates.csv', dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)
        data = data[~data['id_str'].isin(records)].reset_index(drop=True)
        print('Left:', len(data))
    else:
        data = pd.read_csv('../../data/np_without_duplicates.csv', dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)
    
    for index, row in data.iterrows():
        print(index)
        id_str = row['id_str']
        text = row['text']
        emotion = classification(text)

        line = [{'id_str': id_str, 'emotion':emotion}]
        print(line)
        save_file(line, '../../results/emotions_chatgpt.jsonl')
