# Emotion Detection from Statements

## Table of Contents

[1. Problem Statement](#section1)<br>
[2. Approach towards problem](#section2)<br>
[3. Libraries used](#section3)<br>
[4. Pre-Processing of Data](#section4)<br>
[5. Converting words to numbers & Training Machine Learning Model](#section5)<br>
[6. Deep Learning Model using LSTM RNN](#section6)<br>
[7. Hypertuning model using Keras Tuner](#section7)<br>
[8. Inference](#section8)<br>
[9. Future Works](#section9)<br>
[10. Glossary of Terms](#section10)<br>

<a id=section1></a>
### 1. Problem Statement

The aim of this project is to predict basic human emotions like Love, Joy, Surprise, Fear, Anger and Sadness from a given sentence.
![](https://github.com/sanjayd89/Emotion_Detection/blob/main/images/Emotions.jpg?raw=true)

**Scenario:**
- This dataset contains textual information about human emotions.
- It is a collection of approximately 16,000 different instances sentences conveying a specific human emotions to be used by a Bot for emotion detection.
- The objective here would be to come up with an accurate AI system that can tell emotion after analysing a sentence.

<a id=section2></a>
### 2. Approach towards problem
- The problem statement consists of Unstructred data which cannot be directly fed to any Machine Learning or Deep Learning Algorithm. The primary step will be to convert this **Textual Data** into **Numbers** which is understood by algorithms.
- Methods like **Bag of Words (BoW), Term Frequency - Inverse Document Frequency (TF-IDF), Word2Vec **and** Embeddings** shall be used for converting this text to numbers.
- Both Machine Learning and Deep Learning Models shall be developed for model creation and better of them shall be used to check on a test dataset.

<a id=section3></a>
### 3. Libraries used

In addition to frequently used python libraries like pandas, numpy, matplotlib, the project involves use of following libraries:

|  Scikit Learn | NLTK  | Gensim  | TensorFlow | Keras |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| ![](https://th.bing.com/th/id/OIP.FiRI9nCcHGQSK_GiR-KTQwAAAA?w=137&h=58&c=7&r=0&o=5&dpr=1.25&pid=1.7) | ![](https://th.bing.com/th/id/OIP.dGjHGMXlG7g2TxGcCZ36IwAAAA?w=130&h=150&c=7&r=0&o=5&dpr=1.25&pid=1.7) | ![](https://th.bing.com/th/id/OIP.xAtE3lIaPvLGS5CwvYZmmAAAAA?w=230&h=92&c=7&r=0&o=5&dpr=1.25&pid=1.7) | ![](https://th.bing.com/th/id/OIP.YVokXufleWcPqX4RHC5TAAAAAA?w=151&h=136&c=7&r=0&o=5&dpr=1.25&pid=1.7) | ![](https://th.bing.com/th/id/OIP.NqXP1OWO4WRZnVl0cKlZNQAAAA?w=140&h=150&c=7&r=0&o=5&dpr=1.25&pid=1.7) |

<a id=section4></a>
### 4. Pre-Processing of Data

- The data received had missing punctuations which could lead to poor model training and so the first step after checking for null values was to add appropriate punctuations to the missing words.
- After that following techniques were used to pre-process the data:
	- Removing any extra **spaces**. 
	- Convert all text to **lower case**. 
	- Removal of **Stopwords**. 
	- Using **Porter Stemmer** to **stem** all the words to base word.
	- Using **WordNetLemmatizer** to link words with similar meanings to one word.

Refer [Glossary of Terms](#section11) to understand more about these terms.


<a id=section5></a>
### 5. Converting words to numbers and training ML Models
- Using Scikit learn library, Bag of Words and TF-IDF techniques were used to convert text into numbers so as to train a Machine Learning model. Following are the results:

|   Bag of Words (BoW)   |   Term Frequency - Inverse Document Frequency (TF-IDF)   |
| :------------: | :------------: |
| **Random Forest** Model Results: *accuracy_train*:  0.998, *accuracy_test*: 0.833| **Random Forest** Model Results: *accuracy_train*:  0.998, *accuracy_test*: 0.842  |
| **Naive Bayes** Model Results: *accuracy_train*:  0.895, *accuracy_test*: 0.787  | **Naive Bayes** Model Results: *accuracy_train*:  0.809, *accuracy_test*: 0.728  |

- As can be seen from the result, Random Forest model is overfitting and Naive Bayes has low accuracy.
- This led to using **Word2Vec** technique which is based on neural networks. For a baseline performance, a Random Forest model was trained on the vectors obtained from Word2Vec and the results were:
	- accuracy_train:  0.9980544747081712
	- accuracy_test:  0.37427063073075856
- An Deep Learning model shall be trained to reduce this overfitting.

<a id=section6></a>
### 6. Deep Learning Model using LSTM RNN
- A simple base model of LSTM RNN produced following result which was better than the Random Forest model:
	- accuracy_train:  0.9980544747081712
	- accuracy_test:  0.8052236732425674

- To improve model performance, various techniques like **Dropout, BatchNormalization, 2-Layer LSTM** were tried which gave promising results.
- This led to using **Keras Tuner** for Hyperparameter tuning of LSTM Model.

<a id=section7></a>
### 7. Hypertuning model using Keras Tuner
- Two Tuner classes viz. **RandomSearch & BayesianOptimization** were used to find the better performing model.
- Following were the results from the same:

|   RandomSearch   |   BayesianOptimization   |
| :------------: | :------------: |
| **Model Results:** *accuracy_train*:  0.989, *accuracy_test*: 0.884| **Model Results:** *accuracy_train*:  0.992, *accuracy_test*: 0.866  |

<a id=section8></a>
### 8. Inference
- The RandomSearch model was loaded and used to predict on the test dataset.
- The results from the prediction looked fairly good. Following are some of the examples:

| Input  | Sentiment predicted  |
| :------------ | :------------ |
| i could almost feel it as the flames singed and tortured her frail delicate body leaving nothing behind but a foul smelling concoction of wood and burnt flesh  | sadness  |
|  i realise i am sounding surprisingly like every other person on this site i wish i liked mud wrestling or something a bit more outrageous i feel rather dull and dare i say average |  sadness |
|  i have all of that obviously because of what i do on youtube and my blog and while i have a ton i like that i can feel ok about it because i have it managed in a nice and organized way | joy  |
|  i cannot speak for others but all i know is i feel i am the most successful prettiest version of myself when i walk out of my starbucks with my red cup holiday cup in hand |  joy |
|  i am pretty happy but a little on the nauseated side to feel thrilled | joy |



<a id=section9></a>
### 9. Future Works
- The future work include deploying the model on a cloud platform which can then be used by people all over the world

<a id=section10></a>
### 10. Glossary of Terms
- **Stopwords:**. These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. Examples of a few stop words in English are "the", "a", "an", "has", "have", etc.
- **Lower case:** The key concept of converting to lower case is to reduce the number of words, in particular if they are same.
- **Stemming:** Stemming is the process of producing morphological variants of a root/base word. Stemming programs are commonly referred to as stemming algorithms or stemmers. A stemming algorithm reduces the words "chocolates”, “chocolatey”, “choco” to the root word, “chocolate” and “retrieval”, “retrieved”, “retrieves” reduce to the stem “retrieve".
- **Lemmatization:** Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item. Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meanings to one word. 
- **Bag of Words:** Bag of words is a Natural Language Processing technique of text modelling. In technical terms, we can say that it is a method of feature extraction with text data. This approach is a simple and flexible way of extracting features from documents. A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order. It is called a “bag” of words because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document. Refer this [link](https://www.mygreatlearning.com/blog/bag-of-words/#:~:text=Bag%20of%20words%20is%20a%20Natural%20Language%20Processing,describes%20the%20occurrence%20of%20words%20within%20a%20document. "link") for more details.
- **TF-IDF:** TF-IDF stands for Term Frequency Inverse Document Frequency of records. It can be defined as the calculation of how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the corpus (data-set). Refer this [link](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/) for more details.
- **Word2Vec:** Word2vec is one of the most popular technique to learn word embeddings using a two-layer neural network. Its input is a text corpus and its output is a set of vectors. Word embedding via word2vec can make natural language computer-readable, then further implementation of mathematical operations on words can be used to detect their similarities. A well-trained set of word vectors will place similar words close to each other in that space. Refer this [link](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html) for more details.
- **LSTM RNN:** Long-Short Term Memory networks or LSTMs are a variant of RNN that solve the Long term memory problem of the former. Refer this [link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for more details.
