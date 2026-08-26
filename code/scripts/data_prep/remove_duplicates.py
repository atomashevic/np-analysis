import pandas as pd
import re

def clean_data():

    data = pd.read_csv('../../data/np.csv')[['id_str', 'from_user', 'text', 'time', 'from_user_id_str', 'in_reply_to_status_id_str', 'in_reply_to_screen_name', 'in_reply_to_user_id_str', 'user_followers_count', 'user_friends_count',
        'user_location',]]
    data = data.drop_duplicates()
    data = data[data['in_reply_to_status_id_str'].isna()]
    data['post_type'] = data['text'].apply(lambda x: 'retweet' if x.startswith("RT") else 'post')

    replies = pd.read_csv('../../data/np.csv', dtype={'in_reply_to_status_id_str': 'str'}).drop_duplicates()[['id_str', 'from_user', 'text', 'time', 'from_user_id_str', 'in_reply_to_status_id_str', 'in_reply_to_screen_name', 'in_reply_to_user_id_str', 'user_followers_count', 'user_friends_count',
        'user_location',]]
    replies = replies[~replies['in_reply_to_status_id_str'].isna()]
    replies['post_type'] = 'comment'

    data = pd.concat([data, replies])
    data = data.sort_values(by='time')
    return data 

def clean_rt(text):
    text = re.sub(r"^RT\s*:?\s*", "", text, count=1)
    text = re.sub("@[^\s]*", "", text, count=1)
    text = text.strip()
    return text


def add_parent_id_str(data):

    posts = {}

    for index, row in data[data['post_type']=='post'].iterrows():
        idstr = row['id_str']
        text = row['text'].strip()
        posts[text] = idstr

    comments = {}

    for index, row in data[data['post_type']=='comment'].iterrows():
        idstr = row['id_str']
        text = row['text'].strip()
        comments[text] = idstr

    import re
    data['parent_id_str'] = ''
    for index, row in data.iterrows():
        idstr = str(row['id_str'])
        post_type = row['post_type']
        text = row['text']
        if post_type=='post':
            data.loc[index, 'parent_id_str'] = idstr
            #row['parent_id_str'] = idstr 
        elif post_type=='comment':
            data.loc[index, 'parent_id_str'] = str(row['in_reply_to_status_id_str'])

            #row['parent_id_str'] = row['in_reply_to_status_id_str']
        elif post_type == 'retweet':
            text = clean_rt(text)

            if text in posts.keys():
                data.loc[index, 'parent_id_str'] = str(posts[text])
            elif text in comments.keys():
                data.loc[index, 'parent_id_str'] = str(comments[text])
            else:
                row['parent_id_str'] = 'None'
        else:
            row['parent_id_str'] = 'None'

    return data

def add_target_id_str(data):

    posts = {}

    for index, row in data[data['post_type']=='post'].iterrows():
        idstr = row['id_str']
        text = row['text'].strip()
        posts[text] = idstr

    comments = {}

    for index, row in data[data['post_type']=='comment'].iterrows():
        idstr = row['id_str']
        text = row['text'].strip()
        comments[text] = idstr

    import re
    data['target_id_str'] = ''
    for index, row in data.iterrows():
        idstr = str(row['id_str'])
        post_type = row['post_type']
        text = row['text']
        if post_type=='post':
            data.loc[index, 'target_id_str'] = idstr
            #row['parent_id_str'] = idstr 
        elif post_type=='comment':
            data.loc[index, 'target_id_str'] = idstr

            #row['parent_id_str'] = row['in_reply_to_status_id_str']
        elif post_type == 'retweet':
            text = clean_rt(text)

            if text in posts.keys():
                data.loc[index, 'target_id_str'] = str(posts[text])
            elif text in comments.keys():
                data.loc[index, 'target_id_str'] = str(comments[text])
            else:
                row['target_id_str'] = 'None'
        else:
            row['target_id_str'] = 'None'

    return data

data = clean_data()
data = add_parent_id_str(data)
data = add_target_id_str(data)
data.to_csv('../../data/np_without_duplicates.csv', index=None)