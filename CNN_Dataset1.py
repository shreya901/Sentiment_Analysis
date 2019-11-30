import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot

import nltk
nltk.download('stopwords')
nltk.download('punkt')


data=pd.read_csv('/Users/anjithakarattuthodi/Downloads/Dataset1.csv')
data.head()

data['Review comments']=data['Review comments'].str.replace("[^a-zA-Z#]", " ")
data['Review comments']= data['Review comments'].apply(lambda x: x.lower())

data['Review comments'] = data['Review comments'].apply(lambda x: one_hot(x, 10000, split=' ')) #stich them back
data['Rating'] = data['Rating'].apply(lambda x: x - 1)

X_train, X_test, y_train, y_test = train_test_split(data['Review comments'], data['Rating'], test_size=0.20)
max_words = 300
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add(Embedding(10000,  32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)
y_pred_test_cnn = model.predict(X_test)
y_pred_test_cnn = y_pred_test_cnn >0.5
print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test_cnn ))




