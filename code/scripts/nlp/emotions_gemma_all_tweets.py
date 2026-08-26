from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import pandas as pd 
import os
import collections
from tqdm import tqdm
import re

def save_file(response, output_file):
    with open(output_file, 'a') as f:
        for line in response:
            f.write(json.dumps(line))
            f.write('\n')

def import_human_annotations(file = '../../results/ff_completed_samples_annotations.csv' ):

    translate = {
        'strah': 'fear',
        'poverenje': 'trust',
        'iznenađenje': 'surprise',
        'radost': 'joy',
        'gađenje': 'disgust',
        'bes': 'anger',
        'Emocionalno neutralno': 'netral',
        'tuga': 'sadness',
        'iščekivanje': 'anticipation',
        'Ne mogu da razumem':'unknown'
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

def import_data(data_file, out_file):

    #import data
    if os.path.exists(out_file):
        df = pd.read_json(out_file, lines=True)
        df['id'] = df['id'].astype('str')
        records = list(df['id'].unique())
        
        data = pd.read_csv(data_file, dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)
        data = data[~data['id_str'].isin(records)].reset_index(drop=True)
    else:
        data = pd.read_csv(data_file, dtype={'id_str':'str', 'parent_id_str':'str'})
        data = data[data['post_type'].isin(['post', 'comment'])][['id_str', 'text']].reset_index(drop=True)

    return data


system_prompt = "Vi ste ekspert u klasifikaciji emocija"
user_prompt = ''' Vaš zadatak je da svakom tvitu dodelite jednu od emocija koja dominira u tvitu. Ukoliko tvit ne izražava nikakvu emociju, odaberite neutralno. Ukoliko ne možete da razumete tvit, odaberite nepoznato.
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

{tvit}
'''''

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])

def clean_emotion_text(emotion, emotions_list= ['radost', 'tuga', 'poverenje', 'gađenje', 'strah', 'bes', 'iznenađenje', 'iščekivanje', 'nepoznato', 'neutralno', 'straх']):

    emotion = emotion.lower()
    emotion = emotion.strip()
    
    for em in emotions_list:
        if re.search(em, emotion): #emotion[:len(em)]==em:
            emotion = em

    if emotion=='straх':
        emotion = 'strah'

    return emotion


def run_linear(data, chain, out_file='out.jsonl', emotions_list= ['radost', 'tuga', 'poverenje', 'gađenje', 'strah', 'bes', 'iznenađenje', 'iščekivanje', 'nepoznato', 'neutralno', 'straх']):

    for i in tqdm(range(len(data))):

        text = data.iloc[i]['text']
        tid = data.iloc[i]['id_str']

        emotion = chain.invoke(text).content
        emotion = clean_emotion_text(emotion, emotions_list=emotions_list)

        if emotion in emotions_list:
            res = [{'id':str(tid), 'emotion': str(emotion)}]
            save_file(res, out_file)
        else:
            print('Error', emotion)
            res = [{'id':str(tid), 'emotion': 'nepoznato'}]
            save_file(res, out_file)


def clean_batch(responses, emotions_list= ['radost', 'tuga', 'poverenje', 'gađenje', 'strah', 'bes', 'iznenađenje', 'iščekivanje', 'nepoznato', 'neutralno', 'straх']):
    results = []
    #print(len(responses))
    for ie in range(len(responses)):
        emotion = responses[ie].content
        emotion = clean_emotion_text(emotion, emotions_list=emotions_list)
        results.append(emotion)
    return results


def run_batches(data, chain, BATCH_SIZE=10, out_file = 'out.jsonl', emotions_list= ['radost', 'tuga', 'poverenje', 'gađenje', 'strah', 'bes', 'iznenađenje', 'iščekivanje', 'nepoznato', 'neutralno', 'straх']):

    inputs = []
    tids = []
    for i in tqdm(range(len(data))):
        text = data.iloc[i]['text']
        tid = data.iloc[i]['id_str']
        #content = {"text": text}
        inputs.append(text)
        tids.append(tid)
        
        if len(inputs) == BATCH_SIZE:
            #time.sleep(10)
            responses = chain.batch(inputs)
            #clean output for saving in file
            #print()
            results = clean_batch(responses, emotions_list=emotions_list)
            #print(results)
            for ie in range(len(results)):
                emotion = results[ie]
                t = tids[ie]

                if emotion in emotions_list:
                    res = [{'id':str(t), 'emotion': str(emotion)}]
                    save_file(res, out_file)
                else:
                    print('Error', emotion)
                    res = [{'id':str(t), 'emotion': 'nepoznato'}]
                    save_file(res, out_file)
                
            inputs = []
            tids = []
        
    if inputs:
        #time.sleep(10)
        #time.sleep(10)
        responses = chain.batch(inputs)
        #clean output for saving in file
        print()
        results = clean_batch(responses, emotions_list=emotions_list)
        print(results)
        for ie in range(len(results)):
            emotion = results[ie]
            t = tids[ie]
            if emotion in emotions_list:
                res = [{'id':str(t), 'emotion': str(emotion)}]
                save_file(res, out_file)
            else:
                print('Error', emotion)
                res = [{'id':str(t), 'emotion': 'nepoznato'}]
                save_file(res, out_file)

if __name__ == "__main__":
    
    for TEMP in [1.0, 0.0]:
        for MODEL_NAME in ['gemma4:31b']: #["gemma4:31b",]:  #"gemma3:12b"]:

            print('MODEL: %s, TEMP: %s'%(MODEL_NAME, TEMP))

            #MODEL_NAME = 'gemma3:12b'
            #TEMP = 0.0

            llm = ChatOllama( model = MODEL_NAME,
                                temperature = TEMP,
                                num_ctx= 2048, #8192, 
                                #request_timeout = 10,
                                #num_predict = 256,
                                #reasoning = True/False/None 
                                )
            
            chain = prompt | llm 

            #import data
            out_file = '../../results/emotions_gemma/emotions_%s_temp%s_all_tweets_1.jsonl'%(MODEL_NAME, TEMP)
            data_file = '../../data/np_without_duplicates.csv'
            #students_file = '../../results/emotions_ff/ff_samples_18_25_annotations_with_majority.csv'
            
            data = import_data(data_file, out_file)

            #filter tweets for validation
            #students_tweets = list(import_human_annotations(file = students_file)['id_str'].unique())
            #data = data[data['id_str'].isin(students_tweets)].reset_index(drop=True)
            print('Left:', len(data))
            print('Left:', len(data['id_str'].unique()))

            batches = False
            BATCH_SIZE = 10

            emotions_list= ['radost', 'tuga', 'poverenje', 'gađenje', 'strah', 'bes', 'iznenađenje', 'iščekivanje', 'nepoznato', 'neutralno', 'straх']

            if batches:
                print('Running in batches of size %s'%BATCH_SIZE)
                run_batches(data, chain, out_file=out_file, BATCH_SIZE=BATCH_SIZE, emotions_list=emotions_list)
            else:
                #run linear models
                print('Running linear')
                run_linear(data, chain, out_file=out_file, emotions_list=emotions_list)
