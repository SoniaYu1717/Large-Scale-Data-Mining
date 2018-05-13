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
category = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
trainDatasets = fetch_20newsgroups(subset = 'train', categories = category, shuffle = True, random_state = 42, remove = ('headers','footers','quotes') )


# Load the test data
testCate = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
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


# Process data with TFxICF method
icfData = list()
for j in range(20):
	data_cl = ""
	for i in range(len(data[j])):
		data_cl = data_cl + " " + data[j][i]
	icfData.append(data_cl)

icfExcData = list()
for j in range(len(icfData)):
	temp_icf = icfData[j]
	temp_icf = re.sub("[^a-zA-Z]"," ",temp_icf)
	temp_icf = temp_icf.lower()
	words_icf = temp_icf.split()
	excstop_icf = [wicf for wicf in words_icf if not wicf in stopWords]
	excstem_icf = [engStemmer.stem(w1icf) for w1icf in excstop_icf]
	temp_icf = " ".join(excstem_icf)
	icfExcData.append(temp_icf)


icf_X = vectorization.fit_transform(icfExcData[:])
icf_X_train = tfidfVec.fit_transform(icf_X)

ibm_li = icf_X_train.toarray()[3]
mac_li = icf_X_train.toarray()[4]
forsale_li = icf_X_train.toarray()[6]
christian_li = icf_X_train.toarray()[15]

ibm_sort = sorted(ibm_li)
mac_sort = sorted(mac_li)
forsale_sort = sorted(forsale_li)
christian_sort = sorted(christian_li)

ibm_sort = ibm_sort[-10:]
mac_sort = mac_sort[-10:]
forsale_sort = forsale_sort[-10:]
christian_sort = christian_sort[-10:]

ibm_index = list()
mac_index = list()
forsale_index = list()
christian_index = list()

for i in range(len(ibm_li)):
    if ibm_li[i] in ibm_sort:
        ibm_index.append(i)
    if mac_li[i] in mac_sort:
        mac_index.append(i)
    if forsale_li[i] in forsale_sort:
        forsale_index.append(i)
    if christian_li[i] in christian_sort:
        christian_index.append(i)

ibm_feat = list()
mac_feat = list()
forsale_feat = list()
christian_feat = list()

for j in ibm_index:
	ibm_feat.append(vectorization.get_feature_names()[j])
for j in mac_index:
	mac_feat.append(vectorization.get_feature_names()[j])
for j in forsale_index:
	forsale_feat.append(vectorization.get_feature_names()[j])
for j in christian_index:
	christian_feat.append(vectorization.get_feature_names()[j])

print(" ")
print("10 most significant terms in ibm category are:")
print(ibm_feat)
print(" ")

print("10 most significant terms in mac category are:")
print(mac_feat)
print(" ")

print("10 most significant terms in forsale category are:")
print(forsale_feat)
print(" ")

print("10 most significant terms in christian category are:")
print(christian_feat)
print(" ")
