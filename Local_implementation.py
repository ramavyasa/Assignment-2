import re
import math

def sigmoid(x):
        #print(x)
        if x < -500:
                return 0
        return 1 / (1 + math.exp(-x))

weight = dict()
s = set()
final = dict()
idf = dict()
tfidf = dict()
lr = 0.001
meu = 0.05
n = 1
classes = set()
epochs = 1

def tf_idf():
        count = 0
        f = open("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt",'r')  #open file
        #f = open('data','r')
        for i in f.readlines():
                count += 1
                words = i.split(' ')    #split the line at white space
                key = words[0].split(',')       #collect all classes to which doc belong
                new_words = [''.join(words[i:i+n]) for i in range(1,len(words) - (n - 1))]              #n-gram model
                for k in key:
                        classes.add(k)          #make a list of all possible classes
                for w in new_words:             #for each word in the doc do this
                        wc = re.sub(r'[^\w+\s+]','',w)          #clear unnessasary stuff
                        #print(w,wc)
                        if wc not in final:             #if seeing this word for the first time do this
                                s.add(wc)
                                final[wc] = dict()
                        for k in key:
                                if k not in final[wc]:          #if the particular word has not been seen in that class
                                        final[wc][k] = 1
                                else:
                                        final[wc][k] += 1   #else increment the count
        
        for w in s:             #find idf
                idf[w] = math.log(count / len(final[w]))
                weight[w] = dict()
        
        for w in s:             #initialize tfidf and weights
                tfidf[w] = dict()
                for cla in classes:
                        tfidf[w][cla] = 0
                        weight[w][cla] = 0
        
        for w in final:
                i = idf[w]
                for c in final[w]:
                        tfidf[w][c] += math.log(final[w][c] + 1) * i             #calculate tfidf

        print("No. of Documents in training set =",count)
        #print(tfidf['the'])
        f.close()
def train(cnt,lr):
        for cnt1 in range(1,2):
                count = 0
                c = 0
                t = 0
                error = 0
                count_cla = 0
                f = open("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt",'r')
                for i in f.readlines():
                        count += 1
                        #if count%5000 == 0:
                        #        print(count)
                        words = i.split(' ')
                        max_prob = 0
                        max_class = ""
                        key = words[0].split(',')
                        new_words = [''.join(words[i:i+n]) for i in range(1,len(words) - (n - 1))]
                        for cla in classes:             #for every class update the weights                
                                if cla in key:
                                        y = 1
                                else:
                                        y = 0
                                x = 0                           #initialse feature function
                                for w in new_words:
                                        wc = re.sub(r'[^\w+\s+]','',w)
                                        x += weight[wc][cla] * tfidf[wc][cla]
                                #print(x,end = ' ')
                                p = sigmoid(x)          #calculate prob
                                if p > max_prob:                #if prob is more than current prob
                                        max_prob = p
                                        max_class = cla
                                #print(p)
                                if y == 1 and p != 0:
                                        error += -math.log(p)
                                elif p != 1:
                                        error += -math.log(1-p)
                                for w in new_words:
                                        wc = re.sub(r'[^\w+\s+]','',w)
                                        #if weight[wc][cla] > 0:
                                        weight[wc][cla] = weight[wc][cla] + (y - p) * lr * tfidf[wc][cla] - 2 * lr * meu * weight[wc][cla]      #update weights
                        if max_class in key:
                                c += 1
                        t += 1
                f.close()
                print("training accuracy for epoch",cnt,"=", c/t*100)
                print("loss =",error)
def test(cnt):
        count = 0
        #print(weight)
        c = 0
        t = 0
        f = open("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_test.txt",'r')
        for i in f.readlines():
                count += 1
                #print(count)
                words = i.split(' ')
                key = words[0].split(',')
                new_words = [''.join(words[i:i+n]) for i in range(1,len(words) - (n - 1))]
                y = 1
                l = list()
                max_prob = 0
                max_class = ""
                for cla in classes:             #for every class update the weights                
                        x = 0                           #initialse feature function
                        for w in new_words:
                                wc = re.sub(r'[^\w+\s+]','',w)
                                if wc in s:
                                        x += weight[wc][cla] * tfidf[wc][cla]
                        #print(x,end = ' ')
                        p = sigmoid(x)          #calculate prob
                        if p > max_prob:                #if prob is more than current prob
                                max_prob = p
                                max_class = cla
                if max_class in key:
                        c += 1
                t += 1

        print("test accuracy for",cnt,"epoch =",float(c) / t * 100)
        f.close()

tf_idf()
print("Learning rate is constant")
print(n,"- gram model")
for cnt in range(0,epochs):
	train(cnt,lr)
	test(cnt)
#print(len(s))        
