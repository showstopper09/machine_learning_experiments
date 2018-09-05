import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

imagtt=pd.read_csv("CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt",sep="\s+",header=None,error_bad_lines=False,warn_bad_lines=False,usecols=[0,1,2],names=["imgid","attid","present"])

imagtt2=imagtt.pivot(index="imgid",columns="attid",values="present")
imglabels=pd.read_csv("CUB_200_2011/CUB_200_2011/image_class_labels.txt",sep=" ",header=None,names=["imgid","label"])
imglabels=imglabels.set_index("imgid")
df=imagtt2.join(imglabels)

x_train, x_test, y_train, y_test = train_test_split(df, df['label'], test_size = 0.20, random_state = 5)
x_train.drop(['label'],inplace=True,axis=1)
x_test.drop(['label'],inplace=True,axis=1)

#how many features each tree has and how many trees this forest will have
clf=RandomForestClassifier(max_features=50,random_state=0,n_estimators=100)
clf.fit(x_train,y_train)

print(clf.predict(x_train.head()))

clf.score(x_test,y_test)

pred_labels=clf.predict(x_test)
cm=confusion_matrix(y_test,pred_labels)


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
	
	
	
	
birds=pd.read_csv("CUB_200_2011/CUB_200_2011/classes.txt",sep="\s+",usecols=[1],names=["birdname"])
birds=birds["birdname"]

np.set_printoptions(precision=2)
plt.figure(figsize=(60,60),dpi=300)
plot_confusion_matrix(cm,classes=birds,normalize=True)
plt.show()

from sklearn import tree
clftree=tree.DecisionTreeClassifier()
clftree.fit(x_train,y_train)
clftree.score(x_test,y_test)

from sklearn.model_selection import cross_val_score

scores=cross_val_score(clf,x_train,y_train,cv=5)
print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))

scorestree=cross_val_score(clftree,x_train,y_train,cv=5)
print("Accuracy:%0.2f (+/-%0.2f)" %(scorestree.mean(),scorestree.std()*2))

max_features_opts=range(5,50,5)
n_estimators_opts=range(10,200,20)
rf_params=np.empty((len(max_features_opts)*len(n_estimators_opts),4),float)
i=0
for max_features in max_features_opts:
	for n_estimators in n_estimators_opts:
		clf=RandomForestClassifier(max_features=max_features,random_state=0,n_estimators=n_estimators)
		scores=cross_val_score(clf,x_train,y_train,cv=5)
		rf_params[i,0]=max_features
		rf_params[i,1]=n_estimators
		rf_params[i,2]=scores.mean()
		rf_params[i,3]=scores.std()*2
		i+=1
		print("Maxfeatures:%0.2f, n_estimators:%0.2f Accuracy:%0.2f (+/-%0.2f)" %(max_features,n_estimators,scores.mean(),scores.std()*2))

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
fig=plt.figure()
fig.clf()
ax=fig.gca(projection='3d')
x=rf_params[:,0]
y=rf_params[:,1]
z=rf_params[:,2]
ax.scatter(x,y,z)
ax.set_zlim(0.2,0.5)
ax.set_xlabel("Maxfeatures")
ax.set_ylabel("Number of estimators")
ax.set_zlabel("Accuracy")
plt.show()