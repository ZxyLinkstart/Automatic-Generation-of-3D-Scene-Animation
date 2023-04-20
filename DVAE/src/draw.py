from cProfile import label
import imp
from os import path
from pyexpat import features
import joblib
import numpy as np 
import sys
# from py import process
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from sympy import true
import torch
import os
from sklearn import preprocessing
from scipy import io

font = {'family' :'Times New Roman',
        'weight' : 'bold',
      }

plt.rc('font', **font)


path = "/data1/zxy/ACTOR/video/videos_result/ours"
featurelist = os.listdir(path)
featurelist.sort()
label = []
features = []

for item  in featurelist:
  label.append( item[item.index("_")+1:-4])
  features_path = os.path.join(path,item)
  data = joblib.load(features_path)[20:60]
  # data = np.append(data, data, axis=0)
  # data = np.append(data, data*1.05, axis=0)
  # data = np.append(data*1.02, data*1.01, axis=0)
  # data = preprocessing.scale(data,axis=0,with_mean=True,with_std=True)
  # mix_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
  # data = mix_max_normalizer.fit_transform(data) - np.mean(data)
  num = len(data)
  print(num)


Tsne = TSNE(n_components=2, learning_rate=5,random_state=6)
# tsne = TSNE(n_components=2, learning_rate=100).fit_transform(feature)
tsne = Tsne.fit_transform(features)

a = []
markerlist = ['o','v','p','o','*','<','>','h','d','^','p','s']
for i in range(len(label)):
  # plt.text(tsne[20*i:20*(i+1), 0], tsne[20*i:20*(i+1), 1],str('o'))
  a.append(plt.scatter(tsne[num*i:num*(i+1), 0], tsne[num*i:num*(i+1), 1],edgecolors='black',linewidths=0.1,c=plt.cm.Set3(i),marker=markerlist[i]))

# for i in range(len(label)):
#   # plt.text(tsne[20*i:20*(i+1), 0], tsne[20*i:20*(i+1), 1],str('o'))
#   a.append(plt.scatter(tsne[num*i:num*i+1, 0], tsne[num*i:num*i+1, 1],edgecolors='black',linewidths=0.5,c='black',marker=markerlist[i] ))


# hoi1 = plt.scatter(tsne[:20, 0], tsne[:20, 1],edgecolors='black',linewidths=0.2,c=plt.cm.Set1(1),marker='o')
# hoi2 =plt.scatter(tsne[20:, 0], tsne[20:, 1],edgecolors='black',linewidths=0.2,marker='^',c=plt.cm.Set1(2))


# plt.legend((hoi1,hoi2),('box','drink'),loc = 'best')
plt.legend(a,label,bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
# plt.savefig('/data1/zxy/ACTOR/ours.pdf',bbox_inches='tight',pad_inches=0.0)
plt.savefig('/data1/zxy/ACTOR/actor_test_show.pdf',bbox_inches='tight',pad_inches=0.0)
plt.show()


#  hotmap
# z = torch.randn(400)*0.01
# for i in range(0,num):
#   features.append(data[i]*0.99 +  z.numpy() )
# io.savemat("/data1/zxy/ACTOR/feature_ttt4_all.mat",{'A':features})