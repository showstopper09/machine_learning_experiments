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
d=pd.read_csv("YouTube-Spam-Collection-v1/Youtube01-Psy.csv")
vectorizer=CountVectorizer()
dvec=vectorizer.fit_transform(d["CONTENT"])
analyze=vectorizer.build_analyzer()
print(d["CONTENT"][349])
analyze(d["CONTENT"][349])

x_train, x_test, y_train, y_test = train_test_split(d['CONTENT'], d['CLASS'], test_size = 0.20, random_state = 5)
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.transform(x_test)

clf=RandomForestClassifier(n_estimators=80)
clf.fit(x_train,y_train)

clf.score(x_test,y_test)
preds=clf.predict(x_test)
cm=confusion_matrix(preds,y_test)


scores=cross_val_score(clf,x_train,y_train,cv=5)
print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))

d=pd.concat([pd.read_csv("YouTube-Spam-Collection-v1/Youtube01-Psy.csv"),pd.read_csv("YouTube-Spam-Collection-v1/Youtube02-KatyPerry.csv"),pd.read_csv("YouTube-Spam-Collection-v1/Youtube03-LMFAO.csv"),pd.read_csv("YouTube-Spam-Collection-v1/Youtube04-Eminem.csv"),pd.read_csv("YouTube-Spam-Collection-v1/Youtube05-Shakira.csv")])
pipeline=Pipeline([('bag of words',CountVectorizer()),('random forest',RandomForestClassifier())])
#make_pipeline=(CountVectorizer(),RandomForestClassifier())
x_train, x_test, y_train, y_test = train_test_split(d['CONTENT'], d['CLASS'], test_size = 0.20, random_state = 5)
pipeline.fit(x_train,y_train)
pipeline.score(x_test,y_test)

scores=cross_val_score(pipeline,x_train,y_train,cv=5)
print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))

from sklearn.feature_extraction.text import TfidfTransformer
pipeline2=make_pipeline(CountVectorizer(),TfidfTransformer(norm=None),RandomForestClassifier())
pipeline2.fit(x_train,y_train)
pipeline2.score(x_test,y_test)

scores=cross_val_score(pipeline2,x_train,y_train,cv=5)
print("Accuracy:%0.2f (+/-%0.2f)" %(scores.mean(),scores.std()*2))

parameters={'countvectorizer__max_features':(None,1000,2000),'countvectorizer__ngram_range':((1,1),(1,2)),'countvectorizer__stop_words':('english',None),'tfidftransformer__use_idf':(True,False),'randomforestclassifier__n_estimators':(20,50,100)}

from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(pipeline2,parameters,n_jobs=1,verbose=1)
grid_search.fit(x_train,y_train)

print("Best score:%0.3f" % grid_search.best_score_)
print("Best params set:")
best_params=grid_search.best_estimator_.get_params()
for param_name in (parameters.keys()):
	print("%s: %r" %(param_name,best_params[param_name]))