from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer


stopWords = text.ENGLISH_STOP_WORDS
engStemmer = SnowballStemmer("english")

# Load the training data from the datasets
category = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
trainDatasets = fetch_20newsgroups(subset = 'train', categories = category, shuffle = True, random_state = 42, remove = ('headers','footers','quotes') )

# Load the test data
testCate = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
testDatasets = fetch_20newsgroups(subset = 'test', categories = category, shuffle = True, random_state = 42, remove = ('headers','footers','quotes') )


# Sort the data into 20 groups with different labels, save the result and calculate the length of each group
index = list()
length = list()
data = list()

for m in range(4):
    tempIndex = list()
    tempIndex.append(list(np.where(trainDatasets.target == m))[0])
    index.append(tempIndex)
    tempData = list()
    for n in index[m][0]:
        tempData.append(trainDatasets.data[n])
    data.append(tempData)
    length.append(len(tempData))

vectorization = CountVectorizer(min_df = 1)
tfidfVec = TfidfTransformer()


# Trim each category to the same number of items
for i in range(4):
    if i != 1:
        data[i][len(data[1]):] = []


# Create new list with the trimed list
dataTrim = list()
for i in range(4):
    dataTrim.extend(data[i])


# Exclude stop words and stems
dataExc = list()
for i in range(len(dataTrim)):
    tp_tr = dataTrim[i]       #tp is short for temporary, tr is short for train
    tp_tr = re.sub("[^a-zA-Z]"," ",tp_tr)
    tp_tr = tp_tr.lower()
    words_tr = tp_tr.split()
    excStop = [w for w in words_tr if not w in stopWords]
    excStem = [engStemmer.stem(w1) for w1 in excStop]
    tp_tr = " ".join(excStem)
    dataExc.append(tp_tr)


#exclude the stop words, punctuations and different stems of a word for testing set
data_ts = testDatasets.data[:]
dataExcTest = list()
for i in range(len(data_ts)):
    tp_ts = data_ts[i]  #tp short for temporary, ts short for test
    tp_ts = re.sub("[^a-zA-Z]"," ",tp_ts)
    tp_ts = tp_ts.lower()
    words_ts = tp_ts.split()
    excStopTest = [w for w in words_ts if not w in stopWords]
    excStemTest = [engStemmer.stem(w1) for w1 in excStopTest]
    tp_ts = " ".join(excStemTest)
    dataExcTest.append(tp_ts)


# Singular value decomposition
X_tr = vectorization.fit_transform(dataExc[:])
X_tr_tfidf = tfidfVec.fit_transform(X_tr)
X_ts = vectorization.fit_transform(dataExcTest[:])
X_ts_tfidf = tfidfVec.fit_transform(X_ts)


U,s,V = np.linalg.svd(X_tr_tfidf.toarray())
s1 = s[0:50]
U1 = U[:,0:50]
V1 = V[0:50,:]
D_tr = np.dot(X_tr_tfidf.toarray(), V1.T)
D_ts = np.dot(X_ts_tfidf.toarray(), V1.T)

label_tr = np.ones([len(data[1])*4, 1])
for i in range(0,4):
    label_tr[len(data[1])*i : len(data[1])*(i+1)] = i * np.ones([len(data[1]),1])
tr_out = np.hstack((D_tr, label_tr))

label_ts = np.zeros([len(D_ts), 1])
for i in range(len(D_ts)):
    label_ts[i] = testDatasets.target[i]
ts_out = np.hstack((D_ts, label_ts))

np.savetxt('multitrain_out.csv', tr_out, delimiter = ',', fmt = '%.8f')
np.savetxt('multitest_out.csv', ts_out, delimiter = ',', fmt = '%.8f')
