import torch
import torch.nn as nn
from torch import optim
import time, random
import os
import csv
from tqdm import tqdm
from lstm import LSTMSentiment
from bilstm import BiLSTMSentiment
from torchtext import data
import numpy as np
import argparse
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
import math
import pickle
import ConfigParser
import json

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

class TweetSearch:
    def __init__(self, topic, group):
        modelpath = "./runs/best_model_"+topic+"_"+group+".pth"
        self.topic = topic
        self.group = group
        self.USE_GPU = torch.cuda.is_available()
        self.EMBEDDING_DIM = 300
        self.HIDDEN_DIM = 150
        self.BATCH_SIZE = 1000
        self.id_field = data.Field(sequential=False,use_vocab=False)
        self.text_field = data.Field(lower=True)
        self.label_field = data.Field(sequential=False)
        self.train_iter, self.dev_iter, self.test_iter = self.load_sst(self.text_field, self.label_field, self.BATCH_SIZE)
        self.model = LSTMSentiment(embedding_dim=self.EMBEDDING_DIM, hidden_dim=self.HIDDEN_DIM, vocab_size=len(self.text_field.vocab), label_size=len(self.label_field.vocab)-1,\
                          use_gpu=self.USE_GPU, batch_size=self.BATCH_SIZE)
        self.model.load_state_dict(torch.load(modelpath))
        return

    def evaluate(self, model,sent):
        model.eval()
        model.batch_size = 1
        model.hidden = model.init_hidden()
        pred = model(Variable(sent.long(), requires_grad=False))
        pred_label = pred.data.numpy().tolist()
    #print (pred_label)
        return pred_label

    def load_sst(self, text_field, label_field, batch_size):
        train, dev, test = data.TabularDataset.splits(path='./data/', train='train_'+self.topic+'_'+self.group+'.csv',
                                                  validation='test_'+self.topic+'_'+self.group+'.csv', test=self.topic+'_'+self.group+'_evaluate_dataset.csv', format='tsv',
                                                  fields=[('text', text_field), ('label', label_field)])
        text_field.build_vocab(train, dev, test)
        label_field.build_vocab(train, dev, test)
        train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=-1)
    ## for GPU run
#     train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
#                 batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
        return train_iter, dev_iter, test_iter

    def preprocess(self):
        all_data = data.TabularDataset(path='tweet_slim.csv',
				format='csv',
				fields=[('grp', None),('id', self.id_field),('text', self.text_field)] , skip_header=True)
        data_iter = data.BucketIterator(all_data,
                batch_size=self.BATCH_SIZE, repeat=False, device=-1)
        return data_iter,all_data

    def load_test_data(self):
        tweets = []
        labels = []
        with open(os.path.join("data", self.topic+'_'+self.group+'_evaluate_dataset.csv'), 'r') as csvfile:
            readCSV = csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter='\t', escapechar='\\')
            for row in readCSV:
                tweets.append(row[0])
                labels.append(int(row[1]))

        with open(os.path.join("data", self.topic+'_'+self.group+'_evaluate_dataset_withID.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter='\t', escapechar='\\')
            for i in range(len(labels)):
                # id, tweet, label
                spamwriter.writerow([i, tweets[i], labels[i]])

        test_data = data.TabularDataset(path=os.path.join("data", self.topic+'_'+self.group+'_evaluate_dataset_withID.csv'),
                                        format='tsv',
                                      fields=[('id', self.id_field), ('text', self.text_field)])
        test_iter = data.Iterator(test_data, batch_size=self.BATCH_SIZE, repeat=False, device=-1, shuffle=False)
        return test_data, test_iter, tweets, labels

    def predict(self):
        test_data, test_iter, tweets, labels = self.load_test_data()
        print(test_data[0].text)
        print(test_data[0].id)
        print(test_data[1].text)
        print(test_data[1].id)
        print(test_data[2].text)
        print(test_data[2].id)
        self.model.eval()
        print ("begin to predict....")
        pred_res = []
        ids = []
        for batch in test_iter:
            sent = batch.text
            idd = batch.id
            ids += (idd.data.numpy().tolist())
            self.model.batch_size = len(idd.data)
            self.model.hidden = self.model.init_hidden()
            pred = self.model(sent)
            pred_label = pred.data.numpy()
            pred_res += pred_label[:, 1].tolist()

        ori_score = [math.exp(x) for x in pred_res]

        with open(os.path.join("data", 'predict_' + self.topic+'_'+self.group+'_evaluate_dataset.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter='\t', escapechar='\\')
            for tweet, label, score in zip(tweets, labels, ori_score):
                spamwriter.writerow([tweet, label, score])


    def search2(self):
        print ("loading data....")
        data,ori_data = self.preprocess()
        print (ori_data[0].text)
        print (ori_data[1].text)
        print (ori_data[2].text)
        self.model.eval()
        avg_loss = 0.0
        id_list = []
        pred_res = []
        id = 0
        print ("begin to search....")
        for batch in data:
            id = id +1
            if id %10 == 0:
                print (id)       	
            sent = batch.text
            idd = batch.id
            #id_list = np.append(id_list,idd.data.numpy() )
            id_list+=(idd.data.numpy().tolist())
            self.model.batch_size = len(idd.data)
            self.model.hidden = self.model.init_hidden()
            pred = self.model(sent)
            pred_label = pred.data.numpy()
            #pred_res = np.append(pred_res,pred_label[:,1])
            pred_res += pred_label[:,1].tolist()
        #print (id_list)
        #print (pred_res)
        #pos = pred_res>-0.69
        #id_list = id_list[pos].tolist()
        #pred_res = pred_res[pos].tolist()
        #ss = sorted(pred_res, reverse=True)
        #print ([math.exp(s) for s in ss[:10]])
        print ("finish searching.")
        print ("begin to ranking....")
        sort_index = [i[0] for i in sorted(enumerate(pred_res), key = lambda x:x[1], reverse = True)]
        sort_id = [id_list[id] for id in sort_index]
        ori_score = [math.exp(pred_res[id]) for id in sort_index]
        # Save sorted id and corresponding scores
        pickle.dump(sort_id, open("sortId_" + self.topic + "_" + self.group + ".pkl", "wb"))
        pickle.dump(ori_score, open("oriScore_" + self.topic + "_" + self.group + ".pkl", "wb"))
        #ori_score_pos = [s for s in ori_score if s>0.5]
        #plt.hist(ori_score_pos, normed=True, bins=100)
        #plt.ylabel('Probability')
        #plt.show()
        pos_ids, pos_h = self.get_tweets_id_score(threshold=0.5)
        return pos_ids, pos_h

    def show_scores_hist(self):
        ori_score = pickle.load(open("oriScore_" + self.topic + "_" + self.group + ".pkl", "rb"))
        print(ori_score[:10])
        hist = np.histogram(ori_score, np.linspace(0, 1, 11))
        print(hist)
        pickle.dump(hist, open("scoreHist_" + self.topic + "_" + self.group + ".pkl", "wb"))

    def get_tweets_id_score(self, threshold):
        ori_score = pickle.load(open("oriScore_" + self.topic + "_" + self.group + ".pkl", "rb"))
        sort_id = pickle.load(open("sortId_" + self.topic + "_" + self.group + ".pkl", "rb"))
        pos_ids = []
        pos_h = {}
        # pos_ids = [sort_id[idx] for idx, s in enumerate(ori_score) if s>0.5]
        for idx, s in enumerate(ori_score):
            if s > threshold:
                pos_ids.append(sort_id[idx])
                pos_h[sort_id[idx]] = s
        # print (sort_id[:10])
        # print (ori_score[:10])
        # print (ori_data[0].text)
        print (pos_ids)
        return pos_ids, pos_h

    def sample_tweets_id_score(self, num, path):
        """
        :num: randomly sample num of tweets, equally distributed in each 10 bin.
        :return:
        """
        # Get id for each group
        ids_by_group = {"con": set(), "lib": set()}
        with open(path, 'r') as csvfile:
            readCSV = csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter=',', escapechar='\\')
            first_line = True
            for row in readCSV:
                if first_line == True:
                    first_line = False
                    continue
                else:
                    ids_by_group[row[0]].add(int(row[1]))

        ori_score = pickle.load(open("oriScore_" + self.topic + "_" + self.group + ".pkl", "rb"))
        sort_id = pickle.load(open("sortId_" + self.topic + "_" + self.group + ".pkl", "rb"))
        bin = 10
        size = num/10
        all_pos_ids = []
        pos_ids = []
        pos_h = {}
        threshold = 0.9
        # Group id and scores by 10 bin
        for idx, s in enumerate(ori_score):
            if s >= threshold:
                id = sort_id[idx]
                if id in ids_by_group[self.group]:
                    pos_ids.append(id)
                    pos_h[id] = s
            else:
                print("There are {} tweets with a score more than {} in group {}".format(len(pos_ids), threshold, self.group))
                all_pos_ids.append(pos_ids)
                pos_ids = []
                threshold -= 1.0/bin
        print("There are {} tweets with a score more than {} in group {}".format(len(pos_ids), threshold, self.group))
        all_pos_ids.append(pos_ids)
        # Sample the number "size" of tweets in each sub-group
        sample_pos_ids = []

        for pos_ids in all_pos_ids:
            sampled_ids = random.sample(pos_ids, size)
            sample_pos_ids += sampled_ids

        random.shuffle(sample_pos_ids)

        return sample_pos_ids, pos_h

    def get_id_already(self, file_path):
        id_already = []
        with open(file_path, 'r') as csvfile:
            readCSV = csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter=',', escapechar='\\')
            for row in readCSV:
                status = row[2].strip()
                id = status.split("/")[-1]
                id_already.append(int(id))
        print("id_ready: {}".format(id_already))
        return id_already

    def sample_tweets_id_score2(self, num, path, id_already):
        """
        :num: randomly sample num of tweets, equally distributed in each 10 bin.
        :id_already: is the list of ID which is already sampled last time
        :return:
        """
        # Get id for each group
        ids_by_group = {"con": set(), "lib": set()}
        with open(path, 'r') as csvfile:
            readCSV = csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter=',', escapechar='\\')
            first_line = True
            for row in readCSV:
                if first_line == True:
                    first_line = False
                    continue
                else:
                    ids_by_group[row[0]].add(int(row[1]))

        ori_score = pickle.load(open("oriScore_" + self.topic + "_" + self.group + ".pkl", "rb"))
        sort_id = pickle.load(open("sortId_" + self.topic + "_" + self.group + ".pkl", "rb"))
        bin = 5
        size = num/bin
        all_pos_ids = []
        pos_ids = []
        pos_h = {}
        threshold = 0.95
        # Group id and scores by 10 bin
        for idx, s in enumerate(ori_score):
            if s >= threshold:
                id = sort_id[idx]
                if id in ids_by_group[self.group] and id not in id_already:
                    pos_ids.append(id)
                    pos_h[id] = s
            else:
                print("There are {} tweets with a score more than {} in group {}".format(len(pos_ids), threshold, self.group))
                all_pos_ids.append(pos_ids)
                pos_ids = []
                threshold -= 0.05
                if threshold < 0.74:  # TODO: modify the lowest threshold for sampling
                    break
        # Sample the number "size" of tweets in each sub-group
        sample_pos_ids = []

        for pos_ids in all_pos_ids:
            sampled_ids = random.sample(pos_ids, size)
            sample_pos_ids += sampled_ids

        random.shuffle(sample_pos_ids)

        return sample_pos_ids, pos_h
        
    def read_tweet_by_id(self, path, ids, ids_h, topic, group, sample=False):
        print (len(ids))
        id_set = set(ids)
        print(len(id_set))
        ans = {}
        with open(path, 'r') as csvfile:
            readCSV = csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter=',', escapechar='\\')
            first_line = True
            for row in readCSV:
                #print (row)
                if first_line ==  True:
                    first_line = False
                    continue
                if int(row[1]) in id_set and row[0] == group:
                    ans[int(row[1])] = row
        print(len(ans))
        if sample:
            tweet_name = 'sampled_tweet_'
        else:
            tweet_name = 'tweet_'
        with open(tweet_name+topic+'_'+group+'.csv', 'w') as csvfile:
            spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter=',', escapechar='\\')
            for x in ids:
                if x in ans:
                    t = ans[x]
                    tweet = [t[0], t[2], 'https://twitter.com/a/status/'+t[1], ids_h[int(t[1])]]
                    #print (ans[x])
                    #print ("")
                    spamwriter.writerow(tweet)



config = ConfigParser.RawConfigParser()
config.read('parameter.cfg')
topic = json.loads(config.get("content","topic"))
#topic = "gun"
group = "con"
print("Processing {} in group {}".format(topic, group))
ts = TweetSearch(topic, group)

if True:
    # If you want to predict on a test dataset (here I used the annotated data)
    ts.predict()
if True:
    # If you want to get all tweets which are related to the topic by a specific threshold.
    ids, ids_h = ts.search2()
    ts.show_scores_hist()
    ts.read_tweet_by_id("tweet_slim.csv", ids, ids_h, topic, group)
if True:
    # If you want to get stratified-sampled tweets according to their confidence score
    ids, ids_h = ts.sample_tweets_id_score(400, "tweet_slim.csv")
    ts.read_tweet_by_id("tweet_slim.csv", ids, ids_h, topic, group, sample=True)

if True:
    # If you want to resample stratified-sampled tweets and don't want repeated tweets as the first sample
    id_already_con = ts.get_id_already("sampled_tweet_" + topic + "_" + group + ".csv")
    ids, ids_h = ts.sample_tweets_id_score2(400, "tweet_slim.csv", id_already_con)
    ts.read_tweet_by_id("tweet_slim.csv", ids, ids_h, topic,  group, sample=True)


group2 = "lib"
print("Processing {} in group {}".format(topic, group2))
ts = TweetSearch(topic, group2)

if True:
    # If you want to predict on a test dataset (here I used the annotated data)
    ts.predict()
if True:
    # If you want to get all tweets which are related to the topic by a specific threshold.
    ids, ids_h = ts.search2()
    ts.show_scores_hist()
    ts.read_tweet_by_id("tweet_slim.csv", ids, ids_h, topic, group2)
if True:
    # If you want to get stratified-sampled tweets according to their confidence score
    ids, ids_h = ts.sample_tweets_id_score(400, "tweet_slim.csv")
    ts.read_tweet_by_id("tweet_slim.csv", ids, ids_h, topic, group2, sample=True)

if True:
    # If you want to resample stratified-sampled tweets and don't want repeated tweets as the first samples
    id_already_con = ts.get_id_already("sampled_tweet_" + topic + "_" + group2 + ".csv")
    ids, ids_h = ts.sample_tweets_id_score2(400, "tweet_slim.csv", id_already_con)
    ts.read_tweet_by_id("tweet_slim.csv", ids, ids_h, topic,  group2, sample=True)

