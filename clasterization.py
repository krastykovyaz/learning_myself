import pandas as pd
import numpy as np
from itertools import chain
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pprint
from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.models import word2vec

handle = pd.read_excel('/Users/OUT-Kovyazin-AA/Desktop/work_space/credit/claster_cred.xlsx')
replics = ''
f = open("/Users/OUT-Kovyazin-AA/Desktop/work_space/credit/replics.txt", "w")
for line in handle['INPUT_TEXT']:
    replics += line
    replics += '\n'
f.write(replics)
f.close()

with open('/Users/OUT-Kovyazin-AA/Desktop/work_space/credit/replics.txt', 'r') as file_obj:
    sentences = [line.strip().lower() for line in file_obj]

regexp = re.compile('[^а-я]')
# print(sentences[0])
# print(re.split(regexp ,sentences[0]))
sentence = [[word for word in re.split(regexp, line) if word and len(word) > 3] for line in sentences]
# print(sentence)
dictionary = corpora.Dictionary(sentence)

# print(dictionary)
# pprint.pprint(dictionary.token2id)

# new_doc = "Хочу офомить кредит"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# # print(new_vec)
# bow_corpus = [dictionary.doc2bow(text) for text in sentence]
# # pprint.pprint(bow_corpus)
# tfidf = models.TfidfModel(bow_corpus)
#
# # transform the "system minors" string
# words = "какой ежемесячный платеж".lower().split()
# # print(tfidf[dictionary.doc2bow(words)])
#
# index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))
#
# query_document = 'Оставить заявку на ипотеку'.split()
# query_bow = dictionary.doc2bow(query_document)
# sim = tfidf[query_bow]
#
# sims = index[sim]

model = word2vec.Word2Vec(sentence, size=300, window=10, workers=2)

# print(model)
my_dict = dict({})
for idx, key in enumerate(model.wv.vocab):
    my_dict[key] = model.wv[key]
# print(my_dict)
matrix = np.zeros((len(my_dict.values()), len(list(my_dict.values())[0])))

# print(matrix.shape)
# for i in range(len(list(my_dict.values()))):
#     for j in range(len(list(my_dict.values())[i].tolist())):
#         matrix[i][j] = (list(my_dict.values())[i].tolist()[j])
# print(matrix.shape)
# km = KMeans(n_clusters=50, init='random', tol=1e-04, random_state=0)
# X = matrix
# y_km = km.fit_predict(X)
# plt.scatter(X[y_km==0,0], X[y_km==0,1], s=100, c='green', marker='s', label = 'kla1')
# plt.scatter(X[y_km==1,0], X[y_km==1,1], s=50, c='orange', marker='o', label = 'kla2')
# plt.scatter(X[y_km==2,0], X[y_km==2,1], s=50, c='red', marker='v', label = 'kla3')
# plt.scatter(X[y_km==2,1], X[y_km==2,2], s=50, c='blue', marker='^', label = 'kla4')
# plt.scatter(X[y_km==2,2], X[y_km==2,3], s=50, c='black', marker = 's', label = 'kla5')
# plt.scatter(X[y_km==2,3], X[y_km==2,4], s=50, c='yellow', marker='*', label = 'kla6')
# plt.scatter(X[y_km==2,4], X[y_km==2,5], s=50, c='lightgreen', marker='D', label = 'kla7')
# plt.scatter(X[y_km==2,5], X[y_km==2,6], s=50, c='lightblue', marker=',', label = 'kla8')
# plt.scatter(X[y_km==2,6], X[y_km==2,7], s=50, c='grey', marker='.', label = 'kla9')
# plt.legend()
# plt.grid()
# plt.show()




