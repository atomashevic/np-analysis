import pandas as pd
import time
import os
from tqdm import tqdm
import openai
import json
import os
import collections
import sys

def import_data(data_file, out_file):

    if os.path.exists(out_file):
        df = pd.read_json(out_file, lines=True)
        df['id_str'] = df['id_str'].astype('str')
        records = list(df['id_str'].unique())
        
        data = pd.read_csv(data_file, dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)
        data = data[~data['id_str'].isin(records)].reset_index(drop=True)

    else:
        data = pd.read_csv(data_file, dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)

    return data

def import_human_annotations(file = '../../results/ff_completed_samples_annotations.csv' ):

    translate = {
        'strah': 'fear',
        'poverenje': 'trust',
        'iznenađenje': 'surprise',
        'radost': 'joy',
        'gađenje': 'disgust',
        'bes': 'anger',
        'Emocionalno neutralno': 'neutral',
        'tuga': 'sadness',
        'iščekivanje': 'anticipation',
        'Ne mogu da razumem': 'unknown'
    }

    anotations = pd.read_csv(file, dtype={'tweet_id': 'str'}).rename(columns={'tweet_id': 'id_str'})
    annotations = anotations.groupby('id_str').label.apply(lambda x: list(x)).reset_index()
    annotations['N'] = annotations['label'].apply(lambda x: len(x))
    annotations['dist'] = annotations['label'].apply(lambda x: collections.Counter(x))
    annotations['label_human_emotions'] = annotations['dist'].apply(lambda x:  max(x, key=lambda k: x[k]))
    annotations['label_human_agreement'] = annotations.apply(lambda x: x['dist'][x['label_human_emotions']], axis=1)
    annotations['label_human_emotions'] = annotations['label_human_emotions'].apply(lambda x: translate[x])

    annotations = annotations[['id_str', 'label_human_emotions', 'label_human_agreement']]

    return annotations

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY_ANA") or os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY_ANA or OPENAI_API_KEY before running this script.")

zero_shot_prompt = ''' Vaš zadatak je da svakom tvitu dodelite jednu od emocija koja dominira u tvitu. Ukoliko tvit ne izražava nikakvu emociju, odaberite neutralno. Ukoliko ne možete da razumete tvit, odaberite nepoznato.
Emocije:
poverenje — osećaj sigurnosti, oslanjanja na nekoga
bes — gnev, ljutnja, ogorčenost
tuga — žalost, bol, gubitak
iznenađenje — šok, zapanjenost, neočekivanost
strah — uznemirenost, bojazan, pretnja
gađenje — odbojnost, odvratnost, moralno negodovanje
radost — sreća, olakšanje, nada
iščekivanje — napetost, očekivanje, praćenje razvoja situacije
neutralno - tvit ne izražava emociju
nepoznato - tvit ne može da se razume

tvit: {}
'''''

def format_prompt(line):
        return zero_shot_prompt.format(line)

def classification(line, TEMP, COMPLETIONS_MODEL):
    prompt = format_prompt(line)
    messages = [
        {"role": "system", "content": prompt},
    ]
    completion_response = openai.chat.completions.create(
                            messages=messages,
                            temperature=TEMP,
                            #top_p=1,
                            #frequency_penalty=0,
                            #presence_penalty=0,
                            model=COMPLETIONS_MODEL)
    label = completion_response.choices[0].message.content.replace('\n','')
    return label

def save_file(response, output_file):
    with open(output_file, 'a') as f:
        for line in response:
            f.write(json.dumps(line))
            f.write('\n')

if __name__ == "__main__":
    #gpt-5 supports only default temperature, 1.0
    for TEMP in [1.0]:
        for COMPLETIONS_MODEL in ["gpt-5.4"]:

            print('MODEL: %s, TEMP: %s'%(COMPLETIONS_MODEL, TEMP))

            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            client = openai.OpenAI()

            out_file = '../../results/emotions_gpt/emotions_%s_temp%s_all_tweets.jsonl'%(COMPLETIONS_MODEL, TEMP)
            data_file = '../../data/np_without_duplicates.csv'
            #students_file = '../../results/emotions_ff/ff_samples_18_25_annotations_with_majority.csv'

            data = import_data(data_file, out_file)
                
            #filter tweets for validation
            #students_tweets = list(import_human_annotations(file = students_file)['id_str'].unique())
            #data = data[data['id_str'].isin(students_tweets)].reset_index(drop=True)
            print('Left:', len(data))
    
            for index, row in tqdm(data.iterrows(), total=len(data)):
                #print(index)
                id_str = row['id_str']
                text = row['text']
                emotion = classification(text, TEMP, COMPLETIONS_MODEL)

                emotions_list = ['radost', 'tuga', 'poverenje', 'gađenje', 'strah', 'bes', 'iznenađenje', 'iščekivanje', 'nepoznato', 'neutralno']

                if emotion in emotions_list:
                    line = [{'id_str': id_str, 'emotion':emotion}]
                    #print(line)
                    save_file(line, out_file)
                else:
                    line = [{'id_str': id_str, 'emotion':'nepoznato'}]
                    save_file(line, out_file)
                    print('Error', emotion)