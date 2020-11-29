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
from sklearn.metrics import precision_recall_fscore_support
import ConfigParser
import json


torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        #print (sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def train_epoch(model, train_iter, loss_function, optimizer):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def evaluate(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        print ("label")
        #print (label.data.numpy().tolist())
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        #print (pred.data.numpy().tolist())
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    precision, recall, f1_score, _ = precision_recall_fscore_support(truth_res,pred_res)
    #reg_f1_scores.append(f1_score[1])
    print ("F1 score: ", f1_score[1])
    print ("Precision: ", precision[1])
    print ("Recall: ", recall[1])
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc


def load_sst(text_field, label_field, batch_size, topic, group, test_on_annotated_data=False):
    if test_on_annotated_data:
        test_file = topic+'_'+group+'_evaluate_dataset.csv'
    else:
        test_file = 'test_'+topic+'_'+group+'.csv'
    train, dev, test = data.TabularDataset.splits(path='./data/', train='train_'+topic+'_'+group+'.csv',
                                                  validation='test_'+topic+'_'+group+'.csv', test=test_file, format='tsv',
                                                  fields=[('text', text_field), ('label', label_field)])
    print (len(train))
    for i in range(10):
        print (train[i].text,train[i].label)
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=-1)
    ## for GPU run
#     train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
#                 batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


# def adjust_learning_rate(learning_rate, optimizer, epoch):
#     lr = learning_rate * (0.1 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer

from torch.autograd import Variable
def evaluate1(model,sent):
    model.eval()
    model.batch_size = 1
    model.hidden = model.init_hidden()
    pred = model(Variable(sent.long(), requires_grad=False))
    pred_label = pred.data.max(1)[1].numpy()
    #print (pred_label)
    return pred_label[0]

def evaluate2(model,sent):
    model.eval()
    model.batch_size = 1
    model.hidden = model.init_hidden()
    pred = model(Variable(sent.long(), requires_grad=False))
    pred_label = pred.data.numpy().tolist()
    #print (pred_label)
    return pred_label


def train_root(topic, group, test_on_annotated_data=False):
    args = argparse.ArgumentParser()
    args.add_argument('--m', dest='model', default='lstm', help='specify the mode to use (default: lstm)')
    args = args.parse_args()

    EPOCHS = 1
    USE_GPU = torch.cuda.is_available()
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 150

    BATCH_SIZE = 50
    timestamp = str(int(time.time()))
    best_dev_acc = 0.0


    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE, topic, group, test_on_annotated_data=test_on_annotated_data)

    print (text_field.vocab.stoi["zzzzzzlove"])
    model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)


    if USE_GPU:
        model = model.cuda()


    print('Load word embeddings...')
# # glove
# text_field.vocab.load_vectors('glove.6B.100d')

# word2vector
    word_to_idx = text_field.vocab.stoi
    pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
    pretrained_embeddings[0] = 0
    word2vec = load_bin_vec('./data/GoogleNews-vectors-negative300-SLIM.bin', word_to_idx)
    for word, vector in word2vec.items():
        pretrained_embeddings[word_to_idx[word]-1] = vector

# text_field.vocab.load_vectors(wv_type='', wv_dim=300)

    model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
# model.embeddings.weight.data = text_field.vocab.vectors
# model.embeddings.embed.weight.requires_grad = False


    best_model = model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.NLLLoss()

    print('Training...')
    #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
    print("Writing to {}\n".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_path = out_dir + '/best_model_'+topic+'_'+group + '.pth'
    for epoch in range(EPOCHS):
        avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch)
        tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))

        #if epoch == 1:
        #    ans = []
        #    test = doc_complete[:1000]
        #    for x in test:
        #        tl = []
        #        for w in x:
        #            tl.append(text_field.vocab.stoi[w])
        #        t = evaluate2(model,torch.Tensor(tl).view(-1,1))
        #        ans.append(t[0][1])
        #    sort_idx = [i[0] for i in sorted(enumerate(ans), key= lambda x:x[1], reverse = True)]
        #    tpo100 = sort_idx[:100]
        #    pos_tweet = [test[t] for t in tpo100]
        #    for idx,p in enumerate(pos_tweet):
        #        p = " ".join(p)
        #        print (str(idx)+" : "+ p)
        #        print (" ")
    

    #evaluate1(model,torch.Tensor(tl).view(-1,1))
        dev_acc = evaluate(model, dev_iter, loss_function, 'Dev')
        test_acc = evaluate(model, test_iter, loss_function, 'Test')
        if dev_acc > best_dev_acc:
            if best_dev_acc > 0:
                os.system('rm '+ model_path)
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), model_path)
            print("Best model is saved.")

    model.load_state_dict(torch.load(model_path))
    test_acc = evaluate(model, test_iter, loss_function, 'Final Test')


#topic = ["gun","img","abo"]
config = ConfigParser.RawConfigParser()
config.read('parameter.cfg')
topic = [json.loads(config.get("content","topic"))]
group = ["con","lib"]
for t in topic:
    for g in group:
        train_root(t, g, test_on_annotated_data=True)


