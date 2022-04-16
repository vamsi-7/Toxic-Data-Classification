
# Toxic-Data-Classification

People tend to leave online conversations due to people posting toxic or disrespectful comments. 
You need to make a machine learning model to recognize if a comment is normal or toxic.
If we can recognize such harmful contributions, we will have a healthier, more open internet.

## What are toxic comments?
Negative online behaviours, like comments that are rude, disrespectful or otherwise likely to make someone leave a discussion.

## The main steps involved in Machine Learning (ML) model are:
```
 Importing necessary packages
 Importing dataset path
 Data Pre-processing
 Data Split
 Building the model
 Training the model
 Test the model
 Apply different Machine Learning algorithms
 Compute results of the accuracy

```

## Importing necessary packages:
```The packages used for toxic comment classification are given below.
-Numpy: a python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.

-Pandas: a software library written for the Python programming language for data manipulation and analysis.

-Sklearn: a Python module integrating classical machine learning algorithms in the tightly-knit world of scientific Python packages

-Nltk: Natural Language Toolkit, a leading platform for building Python programs to work with human language data.

-WordCloud: an image made of words that together resemble a cloudy shape.

-Matplotlib: a plotting library for the Python programming language and its numerical mathematics extension NumPy.
```

## Importing Data-set Path:
```
The next step is importing the dataset. 

The dataset will be stored in the respective directory where the jupyter notebook file is stored.

The path of the dataset directory will be given for importing the dataset for the further steps.
```

## Data Pre-processing:
```
In this particular step, the text inside the dataset will be pre-processed. 

The text pre-processing techniques followed before processing the text data are: - 

- Tokenization of the text: Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual words or terms. Each of these smaller units are called tokens. The tokens could be words, numbers or punctuation marks.

- Removing Stop-words: Frequently occurring common words like articles, prepositions etc. are called stop-words. So, stop-words are removed for each comment. 

- Normalization of the text: Text normalization is the process of transforming text into a single canonical form that it might not have had before. Normalizing text before storing or processing it allows for separation of concerns, since input is guaranteed to be consistent before operations are performed on it.
```

## Data Split: 

```
Before building the models, the data should be split among the training part and the testing part. 80% of the data will be given for the training phase and 20% of the data will be given for the testing phase.
```

## Building the Models for training and testing:
```
1. Bag-of-Words using Word Count Vectorizer:
Bag-of-Words is a feature engineering technique in which a bag is maintained which contains all the different words present in the corpus. 
This bag is known as Vocabulary or Vocab. For each and every word present in the Vocabulary, counts of these words become the features for all the comments present in the corpus. 
It is simple to generate but far from perfect. If we count all words equally, then some words end up being emphasized more than we need.

2. Tf-Idf Transformer:
Tf-Idf is a simple twist on the bag-of-words approach. 
It stands for term frequency–inverse document frequency. 
Instead of looking at the raw counts of each word in each document in a dataset, tf-idf looks at a normalized count where each word count is divided by the number of documents this word appears in.

```

## Apply different Machine learning algorithms on the models built:
```
1. Logistic Regression Model:
Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes.
In simple words, the dependent variable is binary in nature having data coded as either 1 (stands for success/yes) or 0 (stands for failure/no).
Mathematically, a logistic regression model predicts P(Y=1) as a function of X. It is one of the simplest ML algorithms that can be used for various classification problems such as spam detection, Diabetes prediction, cancer detection etc.
```
```
2. Stochastic Gradient Descent (SGD): 

Stochastic Gradient Descent is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. 
Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning. SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. 
Given that the data is sparse, the classifiers in this module easily scale to problems with more than 10^5 training examples and more than 10^5 features. 
```
```
3. Multinomial Naive Bayes: Naive Bayes is based on Bayes’ theorem, where the adjective Naïve says that features in the dataset are mutually independent. 
Occurrence of one feature does not affect the probability of occurrence of the other feature. 
For small sample sizes, Naïve Bayes can outperform the most powerful alternatives. Being relatively robust, easy to implement, fast, and accurate, it is used in many different fields.
```

## Requirements

```
•	Language: Python.
•	Platform: Jupyter Notebook.
•	Packages and tootlkits: Scikit-Learn Machine Learning Toolbox and Natural Language Processing Toolkit.
•	Models used for Training: Bag of words model and Term Frequency-Inverse Document Frequency model.
•	Machine Learning Algorithms: Logistic Regression, Stochastic Gradient Descent, Multinomial Naïve Bayes.
•	Processor: Intel(R) Core(TM) i5-8250U processor, CPU @ 1.60 GHz 1.80 GHz and 8 GB RAM.

```
## Building a Bag-of-words Model:
```python
from sklearn.feature_extraction.text import CountVectorizer

#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))

#transformed train comments
cv_train_comments=cv.fit_transform(norm_train_comments)

#transformed test comments
cv_test_comments=cv.transform(norm_test_comments)

print('BOW_cv_train:',cv_train_comments.shape)
print('BOW_cv_test:',cv_test_comments.shape)

#vocab=cv.get_feature_names()-toget feature names

```
## Building a Term Frequency-Inverse Document Frequency model:
```python
#Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))

#transformed train comments
tv_train_comments=tv.fit_transform(norm_train_comments)

#transformed test comments
tv_test_comments=tv.transform(norm_test_comments)

print('Tfidf_train:',tv_train_comments.shape)
print('Tfidf_test:',tv_test_comments.shape)
```
# Mode Building and Results

## 1.	Logistic Regression Model:

```python
# Modelling the data using Logistic Regression MOdel
#training the model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)

#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_comments,train_toxic)
print(lr_bow)

#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_comments,train_toxic)
print(lr_tfidf)
```

## 2.	Stochastic Gradient Descent:
```python
from sklearn.linear_model import SGDClassifier

#training the linear svm
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)

#fitting the svm for bag of words
svm_bow=svm.fit(cv_train_comments,train_toxic)
print(svm_bow)

#fitting the svm for tfidf features
svm_tfidf=svm.fit(tv_train_comments,train_toxic)
print(svm_tfidf)
```

## 3.	Multinomial Naïve Bayes:
```python
from sklearn.naive_bayes import MultinomialNB

#training the model
mnb=MultinomialNB()

#fitting the svm for bag of words 
mnb_bow=mnb.fit(cv_train_comments,train_toxic)
print(mnb_bow)

#fitting the svm for tfidf features
mnb_tfidf=mnb.fit(tv_train_comments,train_toxic)
print(mnb_tfidf)
```
# Accuracy Prediction

## 1.	Logistic Regression Model:
```python

#Accuracy score for bag of words
from sklearn.metrics import accuracy_score

lr_bow_score=accuracy_score(test_toxic,lr_bow_predict)*100
print("lr_bow_score :",lr_bow_score)

#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_toxic,lr_tfidf_predict)*100
print("lr_tfidf_score :",lr_tfidf_score)

```

## 2.	Stochastic Gradient Descent:
```python
#Accuracy score for bag of words
svm_bow_score=accuracy_score(test_toxic,svm_bow_predict)*100
print("svm_bow_score :",svm_bow_score)

#Accuracy score for tfidf features
svm_tfidf_score=accuracy_score(test_toxic,svm_tfidf_predict)*100
print("svm_tfidf_score :",svm_tfidf_score)
```

## 3.	Multinomial Naïve Bayes:
```python
#Accuracy score for bag of words
mnb_bow_score=accuracy_score(test_toxic,mnb_bow_predict)*100
print("mnb_bow_score :",mnb_bow_score)

#Accuracy score for tfidf features
mnb_tfidf_score=accuracy_score(test_toxic,mnb_tfidf_predict)*100
print("mnb_tfidf_score :",mnb_tfidf_score)
```

# Conclusion

```
From the above results, I had three machine learning algorithms for comparative study i.e., Logistic Regression, Stochastic Gradient Descent, Multinomial Naïve Bayes. 

Among them, logistic Regression has less loss rate and more accuracy i.e., 90.71% for Bag-of-words model of Logistic Regression and 90.51% for Term Frequency-Inverse Document Frequency model of Logistic Regression. 

So, by this it concludes that the Logistic Regression Model will be best suit for the Toxic Comment Classification project implementation.

```
