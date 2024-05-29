import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps=PorterStemmer()

#preprocessing
#vectorise
#predict
#display

def transform_text(text):
    text=text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text :
        if i.isalnum():
            y.append(i)

    x=[]


    for i in y:
        if i not in  stopwords.words('english') and i not in  string.punctuation:
            x.append(i)
    z=[]
    for i in x:
        z.append( ps.stem(i))

    

    return ' '.join(z)

tf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email Classifier')
sms=st.text_input('Write your massage')
if  st.button('Predict'):
    tranformed_sms=transform_text(sms)

    final_input=tf.transform(tranformed_sms)

    result=model.predict(final_input)[0]

    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')






    
