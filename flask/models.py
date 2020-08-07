#from app import db
from sqlalchemy.dialects.postgresql import JSON

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import recall_score
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pandas as pd
import ast
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
  
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("punkt")

'''
class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String())
    result_all = db.Column(JSON)
    result_no_stop_words = db.Column(JSON)

    def __init__(self, url, result_all, result_no_stop_words):
        self.url = url
        self.result_all = result_all
        self.result_no_stop_words = result_no_stop_words

    def __repr__(self):
        return '<id {}>'.format(self.id)

'''
def seperate_headline_to_list(train_df,Train_Y_df):
  train_list_sentence=[]
  train_label_sentence=[]
  for i in range(train_df.values.shape[0]):
    for j in range(train_df.values.shape[1]):
      train_string=" ".join(ast.literal_eval(train_df.values[i][j]))
      train_list_sentence.append(train_string)
      train_label_sentence.append(Train_Y_df.values[i][0])

  '''
  test_list_sentence=[]
  test_label_sentence=[]
  for i in range(test_df.values.shape[0]):
    for j in range(test_df.values.shape[1]):
      test_string=" ".join(ast.literal_eval(test_df.values[i][j]))
      test_list_sentence.append(test_string)
      test_label_sentence.append(Test_Y_df.values[i][0])
  '''
  return train_list_sentence,train_label_sentence

def vectorize_data(train_list,word_array):

  cv=CountVectorizer()
  # this steps generates word counts for the words in your docs
  word_count_vector=cv.fit_transform(train_list)

  #compute idf values of each word based on the word_count_vector
  tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
  tfidf_transformer.fit(word_count_vector)

  #df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
  #visualize idf weights of each word in the training corpus
  #df_idf.sort_values(by="idf_weights")
  # count matrix
  count_vector=cv.transform(train_list)

  # tf-idf scores
  tf_idf_vector=tfidf_transformer.transform(count_vector)
  #feature_names = cv.get_feature_names()
  #get tfidf vector for first document
  #first_document_vector=tf_idf_vector[0]

  #print the scores
  #df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
  #visualize idf vector of the 25 headlines of the first date
  #df.sort_values(by="tfidf", ascending=False)
  #convert test data to vectors
  #count_vector_test=cv.transform(test_list)
  #test_idf_vector=tfidf_transformer.transform(count_vector_test)
  
  temp_count_vector=cv.transform([word_array])
  analysis_idf_vector=tfidf_transformer.transform(temp_count_vector)
  print("idf vector for this headline:\n",analysis_idf_vector)
  return tf_idf_vector,analysis_idf_vector

def load_data():
  url = 'https://raw.githubusercontent.com/kevinflynn11/ML-Final-Project/master/combined%20data%20raw.csv'
  stock_price_url = "https://raw.githubusercontent.com/kevinflynn11/ML-Final-Project/master/stock%20price%20data.csv"

  price_df = pd.read_csv(stock_price_url, index_col =0)
  price_df.index = pd.to_datetime(price_df.index)

  df = pd.read_csv(url, index_col =0)
  df.dropna(inplace=True)
  df=df[1500:]
  Corpus = df.drop("Label", axis="columns")
  Corpus.dropna(inplace=True)

  # Changing str to lower case
  for column in Corpus.columns:
    temp = pd.DataFrame(Corpus[column].str.lower())
    temp.columns =  [column + "_lower"]
    Corpus = pd.concat([Corpus, temp], axis=1).drop(column, axis="columns")

  # Tokenize each word
  for column in Corpus.columns:
    temp = [word_tokenize(entry) for entry in Corpus[column]]
    temp = pd.Series(list(temp)).rename(column + "_token").to_frame()
    temp = temp.set_index(Corpus[column].index)

    Corpus = Corpus.join(temp).drop(column, axis = "columns")

  # Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
  # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
  tag_map = defaultdict(lambda : wn.NOUN)
  tag_map['J'] = wn.ADJ
  tag_map['V'] = wn.VERB
  tag_map['R'] = wn.ADV

  df_list =[]

  for column in Corpus.columns:

    list_temp = []
    for index,entry in enumerate(Corpus[column]):
        
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        
        list_temp.append(str(Final_words))
    #testdf = pd.DataFrame(list_temp).set_index(Corpus.index).rename(column + "_lemmatized")
    testdf = pd.DataFrame(list_temp).set_index(Corpus.index)
    #print(testdf.head())
    testdf = testdf[0].rename(column + "_lemma")
    df_list.append(testdf)

  new_Corpus = pd.concat(df_list, axis=1)
  new_Corpus.columns = Corpus.columns
  new_Corpus = new_Corpus.set_index(Corpus.index)
  new_Corpus.index = pd.to_datetime(new_Corpus.index)

  #break up training vs. testing
  cutoff_dt = pd.to_datetime("2015-01-01")
  train_df = new_Corpus.loc[cutoff_dt : new_Corpus.index[-1]]
  #test_df = new_Corpus.loc[cutoff_dt : new_Corpus.index[-1]]

  result = df["Label"]
  result.index = pd.to_datetime(result.index)
  #splitting label
  train_results = result.loc[cutoff_dt : new_Corpus.index[-1]]
  #test_results = result.loc[cutoff_dt : result.index[-1]]
  
  Encoder = LabelEncoder()
  #Test_Y_df = pd.DataFrame(Encoder.fit_transform(test_results))
  #Test_Y_df = Test_Y_df.set_index(test_results.index)

  Train_Y_df = pd.DataFrame(Encoder.fit_transform(train_results))
  Train_Y_df = Train_Y_df.set_index(train_results.index)
  return train_df, Train_Y_df 

def process_string(string_to_analyze):
    temp = string_to_analyze.lower()
    temp=word_tokenize(temp)
    word_Lemmatized = WordNetLemmatizer()
    word_array=[]
    for word in temp:
        if word not in stopwords.words('english') and word.isalpha():
            word_new=word_Lemmatized.lemmatize(word)
            word_array.append(word_new)
    word_string=" ".join(word_array)
    print("word string for this headline:\n",word_string)
    return word_string

def analyze_headline(string_to_analyze):
    train_df, Train_Y_df= load_data()
    word_array=process_string(string_to_analyze)
    train_list_sentence,train_label_sentence=seperate_headline_to_list(train_df,Train_Y_df)
    train_vector_headline,vector_headline = vectorize_data(train_list_sentence,word_array)
    
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_vector_headline,train_label_sentence)
    # predict the labels on validation dataset

    predictions_SVM = SVM.predict(vector_headline)>0.5
    # Use accuracy_score function to get the accuracy
    return predictions_SVM