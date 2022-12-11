# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:21:59 2022

@author: ashwi
"""
import pandas as pd 
df = pd.read_csv("crime_data.csv")

df.shape
df.dtypes
df.head()

#========================================================
# Box plot
df.boxplot("Murder",vert=False)
df.boxplot("Assault",vert=False)
df.boxplot("UrbanPop",vert=False)
# there are no outliers from the above variables

df.boxplot("Rape",vert=False)
import numpy as np
Q1 = np.percentile(df['Rape'],25)
Q3 = np.percentile(df['Rape'],75)
IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["Rape"] < LW) | (df["Rape"]> UW)]
len(df[(df["Rape"] < LW) | (df["Rape"]> UW)])
# therefore 2 outliers
df.drop([1,27],axis=0,inplace=True)
df.shape
#=========================================================
X = df.iloc[:,1:]

#===========================================================
# Standardization of data 
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X = SS.fit_transform(X)
X = pd.DataFrame(X)

#==========================================================
#################   Hierarchical clustering  ###################
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete")
Y = cluster.fit_predict(X)
Y = pd.DataFrame(Y,columns=["cluster"])
Y.value_counts()
new_data = pd.concat([df,Y],axis=1)

# now we can apply this new_data to any classifier techniques for making model

#================================================================
#######################   KMeans  ############################
from sklearn.cluster import KMeans
KM = KMeans()

inertia = []
for i in range(1,11):
    KM = KMeans(n_clusters=i,random_state=0)
    KM.fit(X)
    inertia.append(KM.inertia_)

print(inertia)

# Elbow Method

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.plot(range(1,11),inertia)
plt.title("Elbow Method")
plt.xlabel("No clusters")
plt.ylabel("inertia")
plt.show()


# scree plot
import seaborn as sns
d1 = {"kvalue":range(1,11),"inertiavalues":inertia}
d2 = pd.DataFrame(d1)
sns.barplot(x="kvalue",y="inertiavalues",data= d2,)

# therefore by seing the Elbow mehtod and screen plot i have decided that 4 clusters the best for this data set

KM = KMeans(n_clusters=4, n_init=30)
Y = KM.fit_predict(X)
Y
Y = pd.DataFrame(Y,columns=["cluster1"])
Y
new_data = pd.concat([df,Y],axis=1)
new_data 

# now we can apply this new_data to any classifier techniques for making model
#=============================================================
############################ DBSCAN ################################

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1,min_samples=3)
dbscan.fit_predict(X)
Y = dbscan.labels_
Y =pd.DataFrame(Y,columns=["cluster2"])
Y.value_counts()
clustering = pd.concat([df,Y],axis=1)
clustering 

noise_data = clustering[clustering["cluster2"]==-1]
noise_data 

final_data = clustering[clustering["cluster2"]!=-1]
final_data 

# outliers are removed from the final_data and we can use this final_data for other clustering techniques for better resluts 

# therefore K-Means are prividing better results










