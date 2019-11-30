import pandas as pd
with open('/user/user01/Project/Beauty_5.json', 'rU') as f:
    data = f.readlines()
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ",".join(data) + "]"

# now, load it into pandas
data = pd.read_json(data_json_str)


data["review"] = data["reviewText"] + data["summary"]
import pickle


from kafka import KafkaProducer
from kafka import KafkaConsumer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
ps=PorterStemmer()

stopword=set(stopwords.words("english"))
trained_model=pickle.load(open('finalized_model.sav','rb'))

stopwords=set(stopwords.words("english"))
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
from json import dumps, loads
from xlrd import open_workbook
import copy
import xlwt
from xlwt import Workbook

wb=Workbook()
sheet1=wb.add_sheet('Sheet1',cell_overwrite_ok=True)
sheet1.write(0,0,"Product_ID")
sheet1.write(0,1,"Rating")

class mykafka():
    def __init__(self,data):
        self.consumer=KafkaConsumer('amazon1',bootstrap_servers='localhost:9092',auto_offset_reset='earliest',
                                    consumer_timeout_ms=10000,value_deserializer=lambda x:x.decode('utf-8'),api_version=(0,10))
        self.producer=KafkaProducer(bootstrap_servers='localhost:9092',api_version=(0,10),
                                    value_serializer=lambda x:dumps(x).encode('utf-8'))
       
        self.topic='amazon1'
        self.model=trained_model
        X_train, X_test, y_train, y_test =train_test_split(data,data['overall'],test_size=0.3)
        self.data=X_test
        
    def preprocess(self,data):
        
        data["review"]=data["review"].str.replace("[^a-zA-Z#]", " ")
        data["review"]= data["review"].apply(lambda x: x.lower())
        data["review"]= data["review"].apply(lambda x: word_tokenize(x)) #word tokenize
        data["review"]= data["review"].apply(lambda x: [i for i in x if i not in stopwords])
        data["review"]= data["review"].apply(lambda x: [ps.stem(i) for i in x]) # stemming
        data["review"]= data["review"].apply(lambda x: [i for i in x if len(i)>3]) #remove short words
        data["review"]= data["review"].apply(lambda x:' '.join([w for w in x ])) #stich them back
        return data
    
    def Consumer_read(self):
        
        i=1
        
        for msg in self.consumer:
            
            i+=1
            
            temp = (msg.value.split(':'))
            k=''.join(e for e in temp[0] if e.isalnum())
            v=''.join(i for i in temp[1] if i.isdigit())
            sheet1.write(i,0,k)
            sheet1.write(i,1,v)
            print(msg.value)
        wb.save('analysis.xls')
            
            
            
            
           
        
       
            
        self.consumer.close()
        
    def Producer_publish(self):
        clean_data=self.preprocess(self.data)
        myvec=tfidf_vectorizer.fit_transform(clean_data["review"])
        
        
        
        ypred=self.model.predict(myvec)
        
        for i in range(len(ypred)):
            rec={self.data['asin'].iloc[i]:str(ypred[i])}
            self.producer.send('amazon1',value=rec)
            self.producer.flush()
            
        self.producer.close()
K=mykafka(data)
K.Producer_publish()
K.Consumer_read()