import pandas as pd
from cyrtranslit import to_latin
import re
import torch
from transformers import pipeline

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

def analyse_text(data, models=["classla/bcms-bertic-frenk-hate"], models_labels = {"classla/bcms-bertic-frenk-hate": "bertic_hate"}):
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")

    data = preproces_dataset(data).reset_index()
    tweets = data['cleaned_text'].to_list()

    for model_name in models:

        print('calculating', model_name)

        pipe = pipeline("text-classification", model=model_name, device=device,)
        results = pipe(tweets, truncation=True,)
        results = pd.DataFrame(results).reset_index().rename(columns={'label': 'label_'+models_labels[model_name], 
                                                                      'score': 'score_'+models_labels[model_name] })

        data = pd.merge(data, results)

    return data

if __name__ == "__main__":

    models_labels = {"classla/bcms-bertic-frenk-hate": "bertic_hate",
                     "ICEF-NLP/bcms-bertic-senticomments-sr-polarity": "bertic_polarity",
                     "ICEF-NLP/bcms-bertic-senticomments-sr-subjectivity": "bertic_subjectivity", 
                     "ICEF-NLP/bcms-bertic-senticomments-sr-sixway": "bertic_sentiment_six",
                     "ICEF-NLP/bcms-bertic-senticomments-sr-fourway": "bertic_sentiment_four", 
                     "MilaNLProc/xlm-emo-t": "xlm_emotions"}
    
    models = ["classla/bcms-bertic-frenk-hate",
              "ICEF-NLP/bcms-bertic-senticomments-sr-polarity",
              "ICEF-NLP/bcms-bertic-senticomments-sr-subjectivity", 
              "ICEF-NLP/bcms-bertic-senticomments-sr-sixway", 
              "ICEF-NLP/bcms-bertic-senticomments-sr-fourway",
              "MilaNLProc/xlm-emo-t"]
    
    data = pd.read_csv('../data/np_without_duplicates.csv')
    #preprocess dataset

    data = analyse_text(data, models=models, models_labels=models_labels)
    data = data.drop(columns=['index'])
    data.to_csv('../data/np_text_features.csv', index=None)