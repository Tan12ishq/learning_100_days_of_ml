import idx2numpy
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load images
x= idx2numpy.convert_from_file("train-images-idx3-ubyte")
y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
x = x.reshape(60000, 28 * 28)
x=pd.DataFrame(x)
y=pd.DataFrame(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#high pc
pca=PCA(n_components=500)

x_train_trf=pca.fit_transform(x_train)
x_test_trf=pca.transform(x_test)


knn1=KNeighborsClassifier()
knn1.fit(x_train_trf,y_train)

y_pred_trf=knn1.predict(x_test_trf)




#finding optimum number of pc
sum=np.cumsum(pca.explained_variance_ratio_*100)
mx=0
for i in range (len(sum)):
    if(sum[i]>=90):
        mx=i
        break
    

pca2=PCA(n_components=mx)

x_train_op=pca2.fit_transform(x_train)
x_test_op=pca2.transform(x_test)


knn2=KNeighborsClassifier()
knn2.fit(x_train_op,y_train)

y_pred_op=knn2.predict(x_test_op)



#without pca

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
        


accuracy_score(y_pred=y_pred_trf,y_true=y_test)
accuracy_score(y_pred=y_pred_op,y_true=y_test)
accuracy_score(y_pred=y_pred,y_true=y_test)
