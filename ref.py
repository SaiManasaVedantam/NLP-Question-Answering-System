import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
import elasticsearch
from elasticsearch import Elasticsearch
import pandas as pd
import spacy as sp
nlp = sp.load("en_core_web_lg")
import requests
import os
import glob

#es=Elasticsearch('http://localhost:9200/')
# Setup Elastic Search
elastic = Elasticsearch([{'host': 'localhost', 'port': 9200, 'use_ssl' : False, 'ssl_verify' : False}], timeout=30, max_retries=10)
    
# Obtain requests from the page
#req = requests.get("http://localhost:9200", verify=False)

os.chdir("C:/CS6320/NLP-Question-Answering-System/articles")
files = glob.glob("*.*")


########Task 1############
def read_text_file(file_path):
    with open(file_path, 'r',encoding = 'utf8') as f:
        return f.read()

df = pd.DataFrame(columns = ("name","content" ))

dict1={}
this_loc = 0
for file in files:
    content = read_text_file(file)
    sent_text = nltk.sent_tokenize(content) 
    for i in sent_text:
        df.loc[this_loc] = file,i
        dict1[file]=i
        this_loc += 1


########Task 2############
col_names=df.columns
for row_number in range(df.shape[0]):
    body=dict([(name, str(df.iloc[row_number][name])) for name in col_names])
    elastic.index(index='nlp_content',doc_type='books',document=body)


########Task 3############
def nlpPipeline(art):
    tokens = nltk.word_tokenize(art)
    lemmatizer=WordNetLemmatizer()
    for i in range(len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i])
    return (nltk.pos_tag(tokens)) 



input=" What is ukiyo-e an example of?"
question_type=""
look_for=""
if " What " in input:
    question_type="what"
    look_for="ORGANIZATION"
    
if " who " or " whom " in input:
    question_type="who"
    look_for="PERSON"
    
if " when " in input:
    question_type="when"
    look_for="DATE"

question=nlpPipeline(input)

search_results=elastic.search(index='nlp_content',doc_type='books',
                        body={"_source":"content","from" : 0, "size" : 100,
                             'query':{
                                 'match':{"content":input }, 
                             }
                             })

my_set=set()

my_list=[]
for i in range(0,100):
    my_set.add(search_results['hits']['hits'][i]['_source']['content'])
    my_list.append(search_results['hits']['hits'][i]['_source']['content'])

Answer_sent=""
max_score=0
flag=False


for i in my_set:
    flag= False
    entity_doc = nlp(i)
    
        
    for entity in entity_doc.ents:
        if entity.label_ is look_for:
            flag=True
            
    if flag is True:
        input=nlp(input)
        i=nlp(i)
        score=i.similarity(input)
        if(score>max_score):
            Answer_sent=i
            max_score=score
    
print(Answer_sent)
print(max_score)


    






