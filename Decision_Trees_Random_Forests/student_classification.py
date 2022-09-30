import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
import graphviz
from sklearn.model_selection import cross_val_score
d=pd.read_csv("student/student-por.csv",sep=";")
len(d)

d['pass']=d.apply(lambda row:1 if(row['G1']+row['G2']+row['G3'])>=35 else 0,axis=1)
d.drop(['G2','G1','G3'],axis=1,inplace=True)
d=pd.get_dummies(d,columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
 'nursery', 'higher', 'internet', 'romantic'])
 
 
 
x_train, x_test, y_train, y_test = train_test_split(d, d['pass'], test_size = 0.20, random_state = 5)
x_train.drop(['pass'],inplace=True,axis=1)
x_test.drop(['pass'],inplace=True,axis=1)
t=tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
t=t.fit(x_train,y_train)


dot_data=tree.export_graphviz(t,out_file=None,label="all",impurity=False,proportion=True,feature_names=list(x_train),class_names=["pass","fail"],filled=True,rounded=True)
graph=graphviz.Source(dot_data)
graph

t.score(x_test,y_test)
scores=cross_val_score(t,x_train,y_train,cv=5)
print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))
i=0
depth=np.empty((19,3),float)
for mxd in range(1,20):
	t=tree.DecisionTreeClassifier(criterion='entropy',max_depth=mxd)
	scores=cross_val_score(t,x_train,y_train,cv=5)
	depth[i,0]=mxd
	depth[i,1]=scores.mean()
	depth[i,2]=scores.std()*2
	print("MaxDepth:%0.2f, Accuracy:%0.2f (+/-%0.2f)" %(mxd,scores.mean(),scores.std()*2))
	i+=1

#plotting error bars
fig,ax=plt.subplots()
ax.errorbar(depth[:,0],depth[:,1],yerr=depth[:,2])
plt.show()
