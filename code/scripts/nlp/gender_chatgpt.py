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

zero_shot_prompt = '''Kog pola je osoba sa ovim username-om? Odgovori sa jednom od ovih opcija zensko, musko, nepoznato
username: {}
'''''

COMPLETIONS_MODEL = "gpt-5.4"
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
                            temperature=0,
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


    data = pd.read_csv('../../data/np_without_duplicates.csv', dtype={'id_str':'str', 'parent_id_str':'str'})
    users = list(data['from_user'].unique())
    users = pd.DataFrame(users, columns=['username'])

    for index, row in users.iterrows():
        print(index)
        user = row['username']
        gender =  classification(user)
        line = [{'user': user, 'gender': gender}]
        save_file(line, '../../results/users_gender_chatgpt.jsonl')