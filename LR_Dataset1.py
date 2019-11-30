import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import nltk
nltk.download('stopwords')
nltk.download('punkt')

ps=PorterStemmer()
vectorizer = TfidfVectorizer()
stopwords=set(stopwords.words("english"))

data=pd.read_csv('/Users/anjithakarattuthodi/Downloads/Dataset1.csv')
data.head()


data['Review comments']= data['Review comments'].str.replace("[^a-zA-Z#]", " ")
data['Review comments']= data['Review comments'].apply(lambda x: x.lower())
data['Review comments'] = data['Review comments'].apply(lambda x: word_tokenize(x)) #word tokenize
data['Review comments']= data['Review comments'].apply(lambda x: [i for i in x if i not in stopwords])
data['Review comments'] = data['Review comments'].apply(lambda x: [ps.stem(i) for i in x]) # stemming
data['Review comments']= data['Review comments'].apply(lambda x: [i for i in x if len(i)>3]) #remove short words
data['Review comments'] = data['Review comments'].apply(lambda x:' '.join([w for w in x ])) #stich them back


def plot_word_cloud(column):
    positive_string = ' '.join(column)
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_string)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

plot_word_cloud(data[data['Rating'] == 1]['Review comments'])
plot_word_cloud(data[data['Rating'] == 2]['Review comments'])


tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(data['Review comments'])

X_train, X_test, y_train, y_test = train_test_split(tfidf, data['Rating'], test_size=0.30)


lreg = LogisticRegression()
lreg.fit(X_train, y_train)


y_pred_test = lreg.predict(X_test)
print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test))
