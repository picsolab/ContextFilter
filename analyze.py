import nltk

import csv
from nltk.corpus import wordnet as wn
import json
import configparser
import os


def read_stop_word():
    lines = [line.rstrip('\r\n') for line in open('stopwordlist.txt')]
    stopword  = set()
    for l in lines:
        stopword.add(l)
    return stopword

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

def analyze(keyphrases, path, topic=None, compare_pos_neg_counts=False):
    stopword = read_stop_word()
    print ("Reading Twitter Corpus...")
    text_twitter  = file_reader(path)
    print ("Totally tweets: "+ str(len(text_twitter)))
    count_num = 0
    res,neg = [],[]
    all_tweet = ""
    all_con = ""
    all_lib = ""
    y_p,y_n = [],[]
    con_num,lib_num,neg_num = 0,0,0
    print ("build positive examples...")
    for tweet in text_twitter:
        current_twitter = tweet[2].lower()
        #if current_twitter.find("#NoBillNoBreak")>=0 or current_twitter.find("#DisarmHate")>=0:
        f_flag = False
        for p in keyphrases:
            if current_twitter.find(p)>=0:
                res.append(current_twitter)
                all_tweet = all_tweet + " " + current_twitter
                if tweet[0] == 'con':
                    con_num +=1
                    all_con = all_con + " " + current_twitter
                else: all_lib = all_lib + " " + current_twitter
                y_p.append(1)
                f_flag = True
                break
            #print (current_twitter)
        if f_flag == False:
            neg.append(current_twitter)
            neg_num += 1
            #y_n.append(0)

    print ("Related tweets number: "+str(len(res)))
    print ("From Con: "+str(con_num))
    print ("From Lib: "+ str(len(res)-con_num))
    from nltk import word_tokenize
    from nltk.collocations import BigramCollocationFinder
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(word_tokenize(all_tweet))
    #print (finder.items()[0:25])
    ngram = list(finder.ngram_fd.items())
    ngram.sort(key=lambda item: item[-1], reverse=True)
    gram_filter = []    
    for gram in ngram[:1000]:
        if gram[0][0] +" "+ gram[0][1] in keyphrases or gram[0][1] == '#' or gram[0][0].lower() in stopword or gram[0][1].lower() in stopword:
            continue
        else:
            gram_filter.append(gram)
    #print("Gram filters: {}".format(gram_filter))

    if len(gram_filter)>20: gram_filter = gram_filter[:20]
    # check the frequency in the negative examples
    print ("New bi-grams found: ")
    for gram in gram_filter:
        count = 0
        for twitter in neg:
            if twitter.find(gram[0][0]+" "+gram[0][1])>=0 or twitter.find(gram[0][0]+gram[0][1])>=0: count = count + 1
        if 1<=(count//gram[1])<=10: 
            if gram[0][0] =='#':
                print (gram[0][0]+gram[0][1])    
            else: print (gram[0][0], gram[0][1])

    if compare_pos_neg_counts:
        with open(os.path.join("data", topic + "_compare_pos_neg_counts.csv"), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter=',', escapechar='\\')
            for gram in gram_filter:
                neg_count = 0
                for twitter in neg:
                    if twitter.find(gram[0][0] + " " + gram[0][1]) >= 0 or twitter.find(
                            gram[0][0] + gram[0][1]) >= 0: neg_count = neg_count + 1
                spamwriter.writerow([gram[0][0] + " " + gram[0][1], str(gram[1]), str(neg_count)])


if __name__ == "__main__":
    path = "tweets_en_trainset.csv"
    config = configparser.RawConfigParser()
    config.read('parameter.cfg')
    topic = json.loads(config.get("content", "topic"))
    keyphrases = json.loads(config.get("content", "seed_phrase"))
    print("seed_phrases are {}".format(keyphrases))
    #keyphrases1 = ["gun control","second amendment","gun violence" ]
    #keyphrases2 = ["immigration","immigrant","refugee","build a wall"]
    #keyphrases3 = ["abortion","pro-life","pro-choice"]
    #analyze(keyphrases1, path)
    analyze(keyphrases, path, topic, compare_pos_neg_counts=True)
    #analyze(keyphrases3, path)


