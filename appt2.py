# https://docs.streamlit.io/library/api-reference/text/st.header


import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import tensorflow
from tensorflow.keras import models 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')



st.title('Emotion Detection Application')

#importing all the pkl, h5:
with open(r'C:\Users\SDubey1\OneDrive - Oceaneering\INSAID Notebooks\proj\NLP\Streamlit\wordvec.pkl', 'rb') as file:    
    words_mine = pickle.load(file)

#Loading the model, model1.h5
Loaded_model1 = models.load_model(r'C:\Users\SDubey1\OneDrive - Oceaneering\INSAID Notebooks\proj\NLP\Streamlit\streamlit_model.h5')



def prediction_steps(input_statement):

    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()
    user_corp = []

    review = re.sub('[^a-zA-Z0-9]', ' ', input_statement)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')] 

    review = ' '.join(review)
    user_corp.append(review)

    test_onehot_repr = [one_hot(words_mine,10000) for words_mine in user_corp]

    ### Embedding
    test_embedded_docs = pad_sequences(test_onehot_repr, padding = 'pre', maxlen = 25)
    ### Preparing X :
    X_test_final = np.array(test_embedded_docs)
    y_test_prediction = list(np.argmax(Loaded_model1.predict(X_test_final),axis=1)) 

    emotions = {3:'sadness', 0:'joy', 2:'love', 1:'anger', 5:'surprise', 4:'fear'}
    emoji = emotions[y_test_prediction[0]] 

    return emoji

def main():    

    input_statement = st.text_input("Enter the sentence", placeholder = "Type here" )
    emotion_detected = ""
    if st.button("Predict"):
        emotion_detected = prediction_steps(input_statement)
    st.success('The output is {}'.format(emotion_detected))

if __name__=='__main__':
    main()

