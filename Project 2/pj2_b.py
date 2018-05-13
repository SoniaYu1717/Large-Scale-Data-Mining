from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.stem.snowball import SnowballStemmer

stopWords = text.ENGLISH_STOP_WORDS
engStemmer = SnowballStemmer("english")

# Load the training data from the datasets
category = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
trainDatasets = fetch_20newsgroups(subset = 'train', categories = category, shuffle = True, random_state = 42, remove = ('headers','footers','quotes') )


# Load the test data
testCate = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
testDatasets = fetch_20newsgroups(subset = 'test', categories = category, shuffle = True, random_state = 42, remove = ('headers','footers','quotes') )

# Sort the data into 20 groups with different labels, save the result and calculate the length of each group
index = list()
length = list()
data = list()

for m in range(20):
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


# Make each class evenly distributed
for i in range(8):
	if i != 3:
		data[i][len(data[3]) :] = []


# Put the preprocessed data into a new list
dataTrim = list()
for j in range(8):
	dataTrim.extend(data[j])


# Exclude unnecessary elements in the list (stop words, punctuations and so on)
dataExc = list()
for i in range(len(dataTrim)):
	tp = dataTrim[i]   #tp stands for temporary
	tp = re.sub("[^a-zA-Z]", " ", tp)
	tp = tp.lower()
	words = tp.split()
	excStop = [w for w in words if not w in stopWords]
	excStem = [engStemmer.stem(w1) for w1 in excStop]
	tp = " ".join(excStem)
	dataExc.append(tp)


# Exclude unnecessary elements of a word for test dataset
dataTest = testDatasets.data[:]
dataExcTest = list()
for j in range(len(dataTest)):
	tempTest = dataTest[j]
	tempTest = re.sub("[^a-zA-Z]"," ", tempTest)
	tempTest = tempTest.lower()
	wordsTest = tempTest.split()
	excStopTest = [wTest for wTest in wordsTest if not wTest in stopWords]
	excStemTest = [engStemmer.stem(wTest1) for wTest1 in excStopTest]
	tempTest = " ".join(excStemTest)
	dataExcTest.append(tempTest)


X = vectorization.fit_transform(dataExc)
tfidfX = tfidfVec.fit_transform(X)
print(tfidfX.toarray().shape)
