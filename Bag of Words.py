import pandas as pd
import nltk
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

def preprocess_text(text):
  text=re.sub(r'<.*?>','',text)
  text=re.sub(r'[^a-zA-Z\s]','',text)

  tokens=word_tokenize(text)
  tokens=[token.lower() for token in tokens]
  stop_words=set(stopwords.words('english'))
  #stemmer=PorterStemmer()
  #tokens=[stemmer.stem(token) for token in tokens]
  tokens=[token for token in tokens if token not in stop_words]
  lemmatizer=WordNetLemmatizer()
  tokens=[lemmatizer.lemmatize(token,pos='v') for token in tokens]
  tokens=[lemmatizer.lemmatize(token,pos='n') for token in tokens]

  preprocessed_text=' '.join(tokens)

  return preprocessed_text

data=pd.read_csv("D:\\Practice\\NLP using Bag Of Words\\IMDB Dataset.csv")
print(data.head())
data['Cleaned_Review']=data['review'].apply(preprocess_text)
print(data.head())

vectorizer=CountVectorizer()
bag_of_words=vectorizer.fit_transform(data['Cleaned_Review'])

print(bag_of_words.shape)

word_frequencies=bag_of_words.sum(axis=0)
feature_names=vectorizer.get_feature_names_out()

word_freq_dict=dict(zip(feature_names,word_frequencies.A1))

top_words=sorted(word_freq_dict.items(),key=lambda x:x[1],reverse=True)[:20]
print(top_words)

word,freqs=zip(*top_words)
plt.figure(figsize=(10,6))
plt.bar(word,freqs)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Most Frequent Words')
plt.show()



# Initialize TruncatedSVD
svd = TruncatedSVD(n_components=500)

# Fit TruncatedSVD to your sparse matrix
svd.fit(bag_of_words)

plt.figure(figsize=(10,6))
plt.plot(range(1,500+1),svd.explained_variance_ratio_.cumsum(),marker='o',linestyle='-')
plt.xlabel('Number of components')
plt.ylabel("Cumulative explained variance ratio")
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

transformed_data=svd.transform(bag_of_words)
print(transformed_data.shape)


xtrain,xtest,ytrain,ytest=train_test_split(transformed_data,data['sentiment'],test_size=0.2,random_state=42,stratify=data['sentiment'])

encoder=LabelEncoder()
ytrain_encoded=encoder.fit_transform(ytrain)
ytest_encoded=encoder.fit_transform(ytest)

print("Encoded Labels:", ytrain_encoded)
print("Mapping Learned by Encoder:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
model=LogisticRegression()
model.fit(xtrain,ytrain_encoded)

y_train_pred=model.predict(xtrain)
accuracy_train=accuracy_score(y_train_pred,ytrain_encoded)
print("Training accuracy of the model is",accuracy_train)
y_test_pred=model.predict(xtest)
accuracy_test=accuracy_score(y_test_pred,ytest_encoded)
print("Testing accuracy of the model is",accuracy_test)
user_input="""Petter Mattei's "Love in the Time of Money" is a visually stunning film to watch. Mr. Mattei offers us a vivid portrait about human relations. This is a movie that seems to be telling us what money, power and success do to people in the different situations we encounter. <br /><br />This being a variation on the Arthur Schnitzler's play about the same theme, the director transfers the action to the present time New York where all these different characters meet and connect. Each one is connected in one way, or another to the next person, but no one seems to know the previous point of contact. Stylishly, the film has a sophisticated luxurious look. We are taken to see how these people live and the world they live in their own habitat.<br /><br />The only thing one gets out of all these souls in the picture is the different stages of loneliness each one inhabits. A big city is not exactly the best place in which human relations find sincere fulfillment, as one discerns is the case with most of the people we encounter.<br /><br />The acting is good under Mr. Mattei's direction. Steve Buscemi, Rosario Dawson, Carol Kane, Michael Imperioli, Adrian Grenier, and the rest of the talented cast, make these characters come alive.<br /><br />We wish Mr. Mattei good luck and await anxiously for his next work.
"""
cleaned_input=preprocess_text(user_input)
vectorized_input=vectorizer.transform([cleaned_input])
print(vectorized_input)
transformed_input=svd.transform(vectorized_input)
predicted_sentiment=model.predict(transformed_input)
if predicted_sentiment==1:
  print("The review is postive")
else:
  print("The review is negative")