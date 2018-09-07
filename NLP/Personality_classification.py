import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import cross_val_score 
path="C:\\Users\\rahul.thereddevils\\Downloads\\machine_learning_experiments"
df=pd.read_csv("mbti_clean_train.csv",encoding = "ISO-8859-1")

df['ie'] = df.type
df['ns'] = df.type
df['ft'] = df.type
df['pj'] = df.type

df['ie']= df['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
df['ns'] = df['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
df['ft'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
df['pj'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['posts'])
xIETrain, xIETest, yIETrain, yIETest = train_test_split(X, df['ie'])
xNSTrain, xNSTest, yNSTrain, yNSTest = train_test_split(X, df['ns'])
xFTTrain, xFTTest, yFTTrain, yFTTest = train_test_split(X, df['ft'])
xPJTrain, xPJTest, yPJTrain, yPJTest = train_test_split(X, df['pj'])
xTrain, xTest, yTrain, yTest = train_test_split(X, df['type'])




# clf=RandomForestClassifier(n_estimators=80)
# clf.fit(x_train,y_train)
# clf.score(xIETest,yIETest)
# preds=clf.predict(x_test)
# cm=confusion_matrix(preds,y_test)
# scores=cross_val_score(clf,x_train,y_train,cv=5)
# print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))

# scores=cross_val_score(pipeline,x_train,y_train,cv=5)
# print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))

from sklearn.feature_extraction.text import TfidfTransformer
pipeline2=make_pipeline(TfidfTransformer(norm=None),RandomForestClassifier())
pipeline2.fit(xIETrain,yIETrain)
pipeline2.score(xIETest,yIETest)

scores=cross_val_score(pipeline2,xIETrain,yIETrain,cv=5)
print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))
xIETrain, xIETest, yIETrain, yIETest = train_test_split(df['posts'], df['ie'])
pipeline2=make_pipeline(CountVectorizer(),TfidfTransformer(norm=None),RandomForestClassifier())

parameters={'countvectorizer__max_features':(None,1000,2000),'countvectorizer__ngram_range':((1,1),(1,2)),'countvectorizer__stop_words':('english',None),'tfidftransformer__use_idf':(True,False),'randomforestclassifier__n_estimators':(20,50,100)}

from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(pipeline2,parameters,n_jobs=1,verbose=1)
grid_search.fit(xIETrain,yIETrain)

print("Best score:%0.3f" % grid_search.best_score_)
print("Best params set:")
best_params=grid_search.best_estimator_.get_params()
for param_name in (parameters.keys()):
	print("%s: %r" %(param_name,best_params[param_name]))