import csv


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

if __name__ == "__main__":
    #doc_complete = pd.read_csv('/home/victorzhz/sentisenseExtract/tweets90_en.csv',error_bad_lines=False)
    #text_twitter = doc_complete['text']
    #nous = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
    print ("Reading Twitter Corpus...")
    text_twitter  = file_reader('tweets_en_trainset.csv')
    print ("Totally tweets: "+ str(len(text_twitter)))
    with open('tweet_slim.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile,quoting=csv.QUOTE_ALL, delimiter=',', escapechar='\\')
        ct = 0
        for tweet in text_twitter:
            ct  = ct +1
            if ct %1000000 ==0:
                print (ct)
            ans = [tweet[0],tweet[4], tweet[2]]
            spamwriter.writerow(ans)
