import pandas as pd
from cyrtranslit import to_latin
import re
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

def clean_tweet_text(text):

    #translate cyrilic to latin
    text = to_latin(str(text), "sr")
    
    #Removing URLs
    text = re.sub(r"http\S+", "", text)
    
    # Removal of mentions
    text = re.sub("@[^\s]*", "", text)

    #Removal of RT sign
    text = re.sub(r"^RT\s*:?\s*", "", text)
    
    #Remove #
    text = re.sub("#[^\s]*", "", text)
    
    # Removal of numbers
    #text = re.sub('[0-9]*[+-:]*[0-9]+', "", text)
    
    # Removal of punctuation
    text = re.sub(r'[^\w\s]', '', text)

    #Remove new line characters
    text = re.sub(r"[\n\r]+", ' ', text)

    #Remove single quotes
    text = re.sub("'", "", text)

    # Convert to lowercase
    text = text.lower()

    #strip blank spaces around text
    text = text.strip()

    #stem str
    #text = stem_str(text)

    return text

def preproces_dataset(data):
    data['cleaned_text'] = data['text'].apply(clean_tweet_text)
    data = data[['id_str', 'post_type', 'text', 'cleaned_text']]
    data = data.reset_index(drop=True)
    return data

if __name__ == "__main__":

    labels_index_emo = {0: "trust", 
                        1: "anger", 
                        2: "sadness", 
                        3: "surprise", 
                        4: "fear", 
                        5: "disgust", 
                        6: "joy", 
                        7: "anticipation",
                        8: "trust_evocation", 
                        9: "anger_evocation", 
                        10: "sadness_evocation", 
                        11: "surprise_evocation", 
                        12: "fear_evocation", 
                        13: "disgust_evocation", 
                        14: "joy_evocation", 
                        15: "anticipation_evocation"}
    
    # Download model
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")
    
    model = SentenceTransformer("procesaur/Emo355")
    print('model loaded')

    #import data
    data = pd.read_csv('../../data/np_without_duplicates.csv')
    data = preproces_dataset(data).reset_index()
    tweets = data['cleaned_text'].to_list()
    print('data loaded')

    #run model
    embeddings = model.encode(tweets)

    #save dataset
    results = pd.DataFrame(embeddings).reset_index()

    results = results.rename(columns = labels_index_emo)
    print(results.head())
    df = pd.merge(data, results)[['id_str', 'trust', 'anger', 'sadness', 'surprise', 'fear', 'disgust', 'joy', 'anticipation',
                                            'trust_evocation', 'anger_evocation', 'sadness_evocation', 'surprise_evocation', 'fear_evocation', 'disgust_evocation', 'joy_evocation', 'anticipation_evocation']]

    df.to_csv('../../results/np_emotions_emo355.csv', index=None)