import nltk
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import pandas as pd
import numpy as np

with open ("C:/Users/rahul.thereddevils/Downloads/machine_learning_experiments/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt","r") as f:
	review_lines=f.readlines()
	
type(review_lines)
len(review_lines)

for one_line in review_lines[0:10]:
	print(one_line)

reviews=[]
review_labels=[]

for line in review_lines:
	line_list=line.split("\t")
	rev=line_list[0]
	reviews.append(rev.lower())
	review_labels.append(int(line_list[1]))
	
review_string=" ".join(reviews)
tknzr=TweetTokenizer()
review_tokens=tknzr.tokenize(review_string)

punctuation=re.compile(r'[-.?!,":;()|0-9]')
review_tokens2=[]
for token in review_tokens:
	word=punctuation.sub("",token)
	if len(word)>0:
		review_tokens2.append(word)

		
review_tokens3=[]
stp_wrds=set(stopwords.words('english'))
for token in review_tokens2:
	token=token.lower()
	if token not in stp_wrds:
		review_tokens3.append(token)
		
fdist=FreqDist()
for word in review_tokens3:
	fdist[word]+=1
len(fdist)
fdist20=fdist.most_common(20)
fdist20

pos_list=[]
for token in review_tokens3:
	pos_list.append(nltk.pos_tag([token]))
for ps in pos_list:
	pos_set.add(ps[0][1])
	
pos_jj=[]
for each_POS in pos_list:
	if each_POS[0][1] in ["JJ","JJR","JJS"]:
		pos_jj.append(each_POS[0][0])
fqdist_jj=FreqDist()
for word in pos_jj:
	fqdist_jj[word]+=1


	
word_lem=WordNetLemmatizer()
lem_adj=[]
lem_verb=[]
lem_adv=[]
for word in review_tokens3:
	word_pos=nltk.pos_tag([word])
	if word_pos[0][1] in ["JJ","JJR","JJS"]:
		lem_adj.append((word_pos[0][0],word_lem.lemmatize(word,wordnet.ADJ)))
	if word_pos[0][1] in ["RB","RBR","RBS"]:
		lem_adv.append((word_pos[0][0],word_lem.lemmatize(word,wordnet.ADV)))
	if word_pos[0][1] in ["VB","VBD","VBG","VBN","VBZ"]:
		lem_verb.append((word_pos[0][0],word_lem.lemmatize(word,wordnet.VERB)))
	
positive_reviews=[]
negative_reviews=[]
for line in review_lines:
	line_list=line.split("\t")
	rev=line_list[0]
	if 1==(int(line_list[1])):
		positive_reviews.append(rev.lower())
	else:
		negative_reviews.append(rev.lower())
		
positive_review_string=" ".join(positive_reviews)
positive_review_tokens=nltk.word_tokenize(positive_review_string)

negative_review_string=" ".join(negative_reviews)
negative_review_tokens=nltk.word_tokenize(negative_review_string)

pos_rev_trigrams=list(nltk.trigrams(positive_review_tokens))
neg_rev_trigrams=list(nltk.trigrams(negative_review_tokens))



tf_vect=TfidfVectorizer(min_df=2,lowercase=True,stop_words="english")
X_tfidf=tf_vect.fit_transform(reviews)

X_tfidf_names=tf_vect.get_feature_names()
X_tf=pd.DataFrame(X_tfidf.toarray(),columns=X_tfidf_names)

y=pd.Series(review_labels)


docs_train, docs_test, y_train, y_test = train_test_split(X_tf, y, test_size = 0.20, random_state = 5)

clf_TF=MultinomialNB()
clf_TF.fit(docs_train,y_train)
y_TF_pred=clf_TF.predict(docs_test)
print(metrics.accuracy_score(y_test,y_TF_pred))

score_tf=confusion_matrix(y_test,y_TF_pred)


def findSent(sentence):
	sent_list=[sentence]
	sent_vect=tf_vect.transform(sent_list)
	send_pred=clf_TF.predict(sent_vect)
	print(send_pred[0])