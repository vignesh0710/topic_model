import sys, os;
import re
import numpy as np
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import operator
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk
nltk.download('stopwords')
from nltk.tokenize.treebank import TreebankWordTokenizer
t = TreebankWordTokenizer()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
nltk.download('wordnet')
from collections import  Counter
from langdetect import detect

def filter_data(raw_data):
    
    english_index = []
    for index,rows in raw_data.iterrows():
        if detect(rows["text"]) == "en":
            english_index.append(index)
    
    raw_data = raw_data.loc[english_index]
    return raw_data


def clean_data(raw_titles):
        
    cleaned_titles = []
    for each_title in raw_titles:
 
        l = [lem.lemmatize(each.lower().strip()) for each in each_title.split(" ") if  each.lower().strip() not in stop_words and each.isdigit()!= True and len(lem.lemmatize(each.lower().strip())) > 3 and each.isalpha()== True]    
        l = list(map(lambda x: re.sub('[^A-Za-z]+', '', x)  ,l ))
        cleaned_titles.append(l)
        
    return cleaned_titles

def get_domain_stop_words(text, count_limit, eng_stop_words):

    word_count= {}
    for each_sent in text:
        for each_word in each_sent:
            if each_word in word_count:
                word_count[each_word] += 1
            else:
                word_count[each_word] = 1
    ds_stop_words_max =  list(sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)[:count_limit])
    ds_stop_words_max = [each[0].strip() for each in ds_stop_words_max] 
    ds_stop_words_min =  list(sorted(word_count.items(), key=operator.itemgetter(1), reverse=False)[:count_limit])
    ds_stop_words_min = [each[0].strip() for each in ds_stop_words_min]
    eng_stop_words.extend(ds_stop_words_max)
    eng_stop_words.extend(ds_stop_words_min)
    
    return eng_stop_words
   
def remove_domain_specific_stop_words(text,eng_stop_words):
    
    cleaned =[]
    for each_sent in text:
        l = []
        for each_word in each_sent:
            if each_word not in eng_stop_words:
                l.append(each_word)
        cleaned.append(l)
            

    return cleaned
                    
def create_corpus(cleaned):
    
    id2word = corpora.Dictionary(cleaned)
    corpus = [id2word.doc2bow(text) for text in cleaned]

    return id2word,corpus  
    
def find_optimal_topic(data_corpus, corp_dictionary,start,end,cleaned):
    
    coherence_dict = {}
    for num_topics in range(start,end):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=data_corpus,
                                           id2word=corp_dictionary,
                                           num_topics=num_topics, 
                                           random_state = 100,
                                           update_every=1,
                                           chunksize=350,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=cleaned, dictionary=corp_dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_dict[num_topics] = coherence_lda

    return max(coherence_dict.items(), key=operator.itemgetter(1))[0]
    
def model(data_corpus, corp_dictionary,topics):
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus= data_corpus,
                                           id2word = corp_dictionary,
                                           num_topics = topics, 
                                           random_state = 100,
                                           update_every=1,
                                           chunksize=300,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
                                           
    print("The word distribution for each topic",lda_model.print_topics())
    
    return lda_model
    
def write_topics(filtered_data, dictionary, model,cleaned):
    
    topics = []
    for i in range(len(cleaned)):
        processed = dictionary.doc2bow(cleaned[i])
        doc_topic = dict(model.get_document_topics(processed))
        topic_order = sorted(doc_topic.items(), key=operator.itemgetter(1))
        #print (topic_order)
        topic_order = ["topic"+str(each[0]) for each in topic_order[:5]]
        topics.append(topic_order)
        

    filtered_data["topics"] = topics
    return filtered_data
     
def write_op_file(filtered_data,op_file_path):

    filtered_data.to_json(op_file_path, orient="records")


def main(data,op_file_path):

	
    """
    main function used to call all the required functions above
    """
    filtered_data = filter_data(data)
    cleaned_data = clean_data(filtered_data["text"].values)
    get_domain_stop_words(cleaned_data,100,stop_words)
    cleaned_text = remove_domain_specific_stop_words(cleaned_data,stop_words)
    op_create_corpus = create_corpus(cleaned_text)
    id2word = op_create_corpus[0]
    corpus = op_create_corpus[1]
    optimal_topic = find_optimal_topic(corpus, id2word,5,11,cleaned_text)
    topic_model = model(corpus, id2word,optimal_topic)
    op_df = write_topics(filtered_data, id2word, topic_model,cleaned_text)
    write_op_file(op_df,op_file_path)
    

if __name__ == "__main__":
    
    inp_file_path = sys.argv[1]
    op_file_path = sys.argv[2]
    train_df = pd.read_json(inp_file_path, lines=True)
    stop_words = list(stopwords.words('english'))
    stop_words.extend(["div", "tr","font","frameset","hr","href","td","tr","tdim","oko","sure","td","img","tdimg","later","used","tab","alttd","mondial","descargar","gratis","para","descarga","espaol","lagu"])
    main(train_df,op_file_path)