from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import numpy as np


# Load the training data from the datasets
category = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
trainDatasets = fetch_20newsgroups(subset = 'train', categories = category, shuffle = True, random_state = 42, remove = ('headers','footers','quotes') )


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


# Plot the histogram figure
plt.figure
pltIndex = range(20)
width = 1
color = ['r', 'b', 'b', 'b', 'b', 'r', 'r', 'c', 'c', 'c', 'c', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
picCate = plt.bar(pltIndex, length, width, color = color)
labelIndex = np.arange(0, 20, 1).tolist()
plt.xticks(labelIndex, ('athe', 'grap', 'misc', 'ibm', 'mac', 'win', 'fors', 'autos', 'motor', 'base', 'hock', 'crypt', 'elec', 'med', 'space', 'chris', 'guns', 'mid', 'p-misc', 'r-misc'))
plt.ylim([300, 650])
plt.ylabel('Number of documents in each topic')
plt.legend((picCate[1], picCate[7]), ('Computer Technology', 'Recreational Activity'), loc = 'upper right')
recreaLen = length[7] + length[8] + length[9] + length[10]
compLen = length[1] + length[2] + length[3] + length[4]
plt.title('Number of documents VS Topics')
plt.grid(True)
plt.show()

# Number of Docs in the 2 categories
print("The number of docs in Recreation categotry is: " + str(recreaLen))
print("The number of docs in Computer categotry is: " + str(compLen))
