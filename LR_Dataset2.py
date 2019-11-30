import pandas as pd
with open('C:\\Users\\amdut\\Desktop\\Beauty_5.json', 'rU') as f:
    data = f.readlines()
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ",".join(data) + "]"

# now, load it into pandas
data = pd.read_json(data_json_str)
data.head()

data["review"] = data["reviewText"] + data["summary"]
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import nltk
nltk.download('stopwords')


nltk.download('punkt')

ps=PorterStemmer()
vectorizer = TfidfVectorizer()
stopwords=set(stopwords.words("english"))



data["review"] = data["reviewText"] + data["summary"]
data['review']=data['review'].str.replace("[^a-zA-Z#]", " ")
data['review']= data['review'].apply(lambda x: x.lower())
data['review'] = data['review'].apply(lambda x: word_tokenize(x)) #word tokenize
data['review']= data['review'].apply(lambda x: [i for i in x if i not in stopwords])
data['review'] = data['review'].apply(lambda x: [ps.stem(i) for i in x]) # stemming
data['review']= data['review'].apply(lambda x: [i for i in x if len(i)>3]) #remove short words
data['review'] = data['review'].apply(lambda x:' '.join([w for w in x ])) #stich them back



tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(data['review'])

X_train, X_test, y_train, y_test = train_test_split(tfidf, data['overall'], test_size=0.30)
X_train, X_test, y_train, y_test = train_test_split(tfidf, data['overall'], test_size=0.30)


lreg = LogisticRegression(multi_class='multinomial',solver='lbfgs')
lreg.fit(X_train, y_train)

y_pred_test = lreg.predict(X_test)
print('Accuracy of Logistics Regression  for multilabel sentiment: %.2f' %accuracy_score(y_test, y_pred_test))