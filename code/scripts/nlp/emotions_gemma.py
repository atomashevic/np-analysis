from langchain_ollama import ChatOllama
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json
import pandas as pd 
import time
import sys
import os
import asyncio

def save_file(response, output_file):
    with open(output_file, 'a') as f:
        for line in response:
            f.write(json.dumps(line))
            f.write('\n')

def import_data(data_file, out_file):

    #import data
    if os.path.exists(out_file):
        df = pd.read_json(out_file, lines=True)
        df['id'] = df['id'].astype('str')
        records = list(df['id'].unique())
        
        data = pd.read_csv(data_file, dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)
        data = data[~data['id_str'].isin(records)].reset_index(drop=True)
        print('Left:', len(data))
    else:
        data = pd.read_csv(data_file, dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)
        print('Left', len(data))

    return data


model_name = 'gemma3:12b'

llm = ChatOllama( model = model_name,
                      temperature = 0,
                      #request_timeout = 10,
                      #num_predict = 256,
                      #reasoning = True/False/None 
                    )

system_prompt = "Ti si ekspert u klasifikaciji emocija"
user_prompt = "Koja emocija je izrazena u datom text-u? Vrati samo jednu od emocija: radost, tuga, poverenje, gadjenje, strah, bes, iznenadjenje, anticipacija \n {text}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])

chain = prompt | llm 

#import data
out_file = '../../results/emotions_gemma.jsonl'
data_file = '../../data/np_without_duplicates.csv'
data = import_data(data_file, out_file)

for i in range(len(data)):

    text = data.iloc[i]['text']
    tid = data.iloc[i]['id_str']
    emotion = chain.invoke(text).content
    emotion = emotion.lower()

    #emotion = emotion.split(' ')[0].strip( ).lower()
    #emotion = emotion.strip('.')
    print(emotion.lower())
    emotions_list = ['radost', 'tuga', 'poverenje', 'gadjenje', 'strah', 'bes', 'iznenadjenje', 'anticipacija']
    for em in emotions_list:
        if emotion[:len(em)]==em:
            emotion = em
    #if emotion in emotions_list:
    print(emotion)
    result = [{'id':str(tid), 'emotion': str(emotion)}]
    save_file(result, out_file)
