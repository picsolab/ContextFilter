import nltk
import os
import sys
import numpy as np
import pandas as pd
import emoji
import csv
import json
import configparser
from nltk.corpus import wordnet as wn

def read_stop_word():
    lines = [line.rstrip('\r\n') for line in open('stopwordlist.txt')]
    stopword  = set()
    for l in lines:
        stopword.add(l)
    return stopword

def extract_emoji(text):
    emoji_list = []
    for i in range(len(text)):
        if text[i] in emoji.UNICODE_EMOJI:
            emoji_list.append(text[i])
    return emoji_list 

def replace_emoji(text, index):
    if index < len(text)-1:
        if text[index] in emoji.UNICODE_EMOJI:
            return replace_emoji(text[:index] + " emoji_label " + text[index+1:], index)
        else:
            return replace_emoji(text, index+1)
    else:
        if text[index] in emoji.UNICODE_EMOJI:
            return text[0:len(text)-1] + " emoji_label."
        else:
            return text

def text_has_emoji(text):
    if text is not None:
        for i in range(len(text)):
            if text[i] in emoji.UNICODE_EMOJI:
                return True
    return False
def extrat_hashtag(text_str):
    hash_tag_list = []
    for word in text_str.split():
        if word.startswith('#') and len(word) > 1:
            hash_tag_list.append((word, 'hashtag'))
            word = 'hashtag_label'
    return hash_tag_list
def extrat_url(text_str):
    url_list = []
    for word in text_str.split():
        if 'http' in word:
            url_list.append((word, 'url'))
            word = 'url_label'
    return url_list
def extract_mention(text_str):
    mention_list = []
    for word in text_str.split():
        if word.startswith('@') and len(word) > 1:
            mention_list.append((word, 'mention_label'))
            word = 'mention_label'
    return mention_list
def preprocess_twitter(text_str):
    cleaned_text = " ".join([word for word in text_str.split()
                            if not word.startswith('&')
                            and word != 'RT'
                            and 'http' not in word
                            and not word.startswith('@')
                            and not word.startswith('#')
                            and  word not in emoji.UNICODE_EMOJI])
    return cleaned_text
def preprocess_string(text_str):
    cleaned_text = " ".join([word for word in text_str.split()
                            if not word.startswith('&')
                            and word != 'RT'])
    cleaned_word = ""
    for word in cleaned_text.split():
        if word.startswith('#') and len(word) > 1:
            cleaned_word = cleaned_word + " "+ "hashtag_label"
        else:
            cleaned_word = cleaned_word + " " + word
    final_word = ""
    for word in cleaned_word.split():
        if 'http' in word:
            final_word = final_word +  " url_label"
        else:
            final_word = final_word + " " + word
    final_text = ""
    for word in final_word.split():
        if word.startswith('@') and len(word) > 1:
            final_text = final_text +  " mention_label"
        else:
            final_text = final_text + " " + word

    return final_text
def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'PERSON':
             entity_names.append((' '.join([child[0] for child in t]),'PERSON'))        
        if t.label() == 'ORGANIZATION':
             entity_names.append((' '.join([child[0] for child in t]), 'ORGANIZATION'))
        if t.label() =='LOCATION':
             entity_names.append((' '.join([child[0] for child in t]),  'LOCATION'))
        if t.label() == 'DATE':
             entity_names.append((' '.join([child[0] for child in t]),  'DATE'))
        if t.label() == 'TIME':
             entity_names.append((' '.join([child[0] for child in t]),  'TIME')) 
        if t.label() =='MONEY':
             entity_names.append((' '.join([child[0] for child in t]),  'MONEY'))
        if t.label() == 'PERCENT':
             entity_names.append((' '.join([child[0] for child in t]),  'PERCENT'))
        if t.label() == 'FACILITY':
             entity_names.append((' '.join([child[0] for child in t]),  'FACILITY'))
        if t.label() == 'GPE':
             entity_names.append((' '.join([child[0] for child in t]),  'GPE'))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

def print_files(file_path):
    file_list = []
    for file in os.listdir(file_path + "/"):
        if file.endswith(".txt"):
            #print(os.path.join("/times", file))
            path = os.path.join(file_path, file)
            file_list.append(path)
    return file_list
def check_all_nouns(current_word):
     nous = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
     for string in current_word:
         if (string.lower() in nous) == False:
             return False
     return True

def check_all_captialization(text_str):
    cleaned_text = " ".join([word for word in text_str.split()
                            if not word.startswith('&')
                            and not word.startswith('#')
                            and 'http' not in word
                            and word not in emoji.UNICODE_EMOJI
                            and word != 'RT'])
    for word in cleaned_text.split():
        if word.isupper() == False:
            return False
    return True

def check_all_start_word_caplitalization(text_str):
    cleaned_text = " ".join([word for word in text_str.split()
                            if not word.startswith('&')
                            and not word.startswith('#')
                            and 'http' not in word
                            and word not in emoji.UNICODE_EMOJI
                            and word != 'RT'])
    for word in cleaned_text.split():
        if str(word[0]).isupper() == False:
            return False
    return True
def convert_to_lower(text_str):
    result_str = ""
    for word in text_str.split():
        if not word.startswith('&') and not word.startswith('#') and 'http' not in word and word not in emoji.UNICODE_EMOJI and word != 'RT':
            result_str = result_str + word.lower()
        else:
            result_str = result_str + word
    return text_str
def file_reader(path):
    doc_complete=[]
    #f = open(path, 'r')
    #for line in f:
    #    doc_complete.append(line)
    #return doc_complete
    with open(path, 'r') as csvfile:
       readCSV = csv.reader(csvfile, delimiter=',' , quoting=csv.QUOTE_ALL, escapechar='\\')
       for row in readCSV:
           doc_complete.append(row)
    return doc_complete

def vectorize(sents,stopword):
    from sklearn.feature_extraction import DictVectorizer
    bow = []
    for sent in sents:
        local_cxt = nltk.word_tokenize(sent)
        word = {}
        for w in local_cxt:
            if w in stopword:
                continue
            if w in word:
                word[w] =  word[w] + 1
            else:
                word[w] = 1
        bow.append(word)
    vec = DictVectorizer()
    res = vec.fit_transform(bow).toarray()
    vocab = vec.get_feature_names()
    return res

def vectorize_tfidf(sents,stopword):
    from sklearn.feature_extraction.text import TfidfVectorizer
    bow = []
    for sent in sents:
        local_cxt = nltk.word_tokenize(sent.lower())
        word = ""
        for w in local_cxt:
            if w in stopword:
                continue
            else:
                word = word + " "+ w
        bow.append(word)
    vectorizer = TfidfVectorizer(min_df=1,max_features = 300)
    X = vectorizer.fit_transform(bow).toarray()
    idf = vectorizer.idf_
    print (vectorizer.get_feature_names())
    return X, vectorizer

def vectorize_tfidf_transform(sents,stopword,vectorizer):
    #from sklearn.feature_extraction.text import TfidfVectorizer
    bow = []
    for sent in sents:
        local_cxt = nltk.word_tokenize(sent.lower())
        word = ""
        for w in local_cxt:
            if w in stopword:
                continue
            else:
                word = word + " "+ w
        bow.append(word)
    #vectorizer = TfidfVectorizer(min_df=1,max_features = 300)
    X = vectorizer.transform(bow).toarray()
    #idf = vectorizer.idf_
    #print (vectorizer.get_feature_names())
    return X

def batch_predict(model, X, vectorizer, stopword):
    batch = 1000
    ep = (len(X)-1)//batch
    ans = []
    for i in range(ep+1):
        print ("epoch: "+str(i))
        end = (i+1)*batch if (i+1)*batch <= len(X) else len(X)
        cb = X[i*batch: end]
        vec_cb = vectorize_tfidf_transform(cb, stopword, vectorizer)
        res = model.predict(vec_cb).tolist()
        ans+=res
    return ans


def seed_tweet(topic, keyphrases, path):
    stopword = read_stop_word()
    print ("Reading Twitter Corpus...")
    text_twitter  = file_reader(path)
    print ("Totally tweets: "+ str(len(text_twitter)))
    count_num = 0
    pos_con,pos_lib,neg_con, neg_lib = [],[], [],[]
    neg_con_num,neg_lib_num = 0,0
    print ("build positive examples...")
    for tweet in text_twitter:
        current_twitter = tweet[2].lower()
        #if current_twitter.find("#NoBillNoBreak")>=0 or current_twitter.find("#DisarmHate")>=0:
        f_flag = False        
        for p in keyphrases:
            if current_twitter.find(p)>=0:
                if tweet[0] == 'con':
                    pos_con.append(current_twitter)
                else:
                    pos_lib.append(current_twitter)
                f_flag = True
                break
            #print (current_twitter)
        if f_flag == False:
            if  neg_con_num < 40000 and tweet[0] == "con":
                neg_con.append(current_twitter)
                neg_con_num += 1
            elif tweet[0] == "lib" and neg_lib_num < 40000:
                neg_lib.append(current_twitter)
                neg_lib_num += 1
    pos_lib_len = len(pos_lib)
    print ("lib len: "+ str(pos_lib_len))
    pos_con_len = len(pos_con)
    print ("con len: "+ str(pos_con_len))
    pos_con_len_train = (pos_con_len * 9)//10
    with open('./data/train_'+topic+'_con.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter='\t', escapechar='\\')
        for current_row in pos_con[:pos_con_len_train]:
            ans = [current_row]
            ans.append(1)
            spamwriter.writerow(ans)
        for current_row in neg_con[:38000]:
            ans = [current_row]
            ans.append(0)
            spamwriter.writerow(ans)

    with open('./data/test_'+topic+'_con.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t', escapechar='\\')
        for current_row in pos_con[pos_con_len_train:]:
            ans = [current_row]
            ans.append(1)
            spamwriter.writerow(ans)
            #spamwriter.writerow(current_row)
        for current_row in neg_con[38000:]:
            ans = [current_row]
            ans.append(0)
            spamwriter.writerow(ans)
   
    # lib
    pos_lib_len = len(pos_lib)
    print ("lib len: "+ str(pos_lib_len))
    pos_lib_len_train = (pos_lib_len * 9)//10
    with open('./data/train_'+topic+'_lib.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter='\t', escapechar='\\')
        for current_row in pos_lib[:pos_lib_len_train]:
            ans = [current_row]
            ans.append(1)
            spamwriter.writerow(ans)
        for current_row in neg_lib[:38000]:
            ans = [current_row]
            ans.append(0)
            spamwriter.writerow(ans)

    with open('./data/test_'+topic+'_lib.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t', escapechar='\\')
        for current_row in pos_lib[pos_lib_len_train:]:
            ans = [current_row]
            ans.append(1)
            spamwriter.writerow(ans)
            #spamwriter.writerow(current_row)
        for current_row in neg_lib[38000:]:
            ans = [current_row]
            ans.append(0)
            spamwriter.writerow(ans)

if __name__ == "__main__":
    config = configparser.RawConfigParser()
    config.read('parameter.cfg')
    seed_phrase = json.loads(config.get("content","seed_phrase"))
    new_phrase = json.loads(config.get("content","new_phrase"))
    topic1 = json.loads(config.get("content","topic"))
    keyphrases1 = seed_phrase+new_phrase
    print (topic1, keyphrases1)
    #topic2 = "img"
    #topic3 = "abo"
    #keyphrases1 = ["gun control","second amendment","gun violence","#disarmhate","gun laws","#wearorange","sense gun","amendment rights"]
    #keyphrases2 = ["immigration","immigrant","refugee","build a wall"]
    #keyphrases3 = ["abortion","pro-life","pro-choice"]
    path = "tweets_en_trainset.csv"
    seed_tweet(topic1,keyphrases1, path)
    #seed_tweet(topic2,keyphrases2, path)   
    #seed_tweet(topic3,keyphrases3, path)

