#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We perform Task1 and Task2 for the query given. Then, we perform search, rank them and obtain the output.

# Turn off unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import all the required packages
import json
import nltk
import urllib
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.wsd import lesk
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser


# Global initializations
lemmatizer = WordNetLemmatizer()
dependencyParser = CoreNLPDependencyParser(url='http://localhost:9000')


# Build the Question Pipeline for our Question types
def QuestionPipeline(question):
    corpus_dict = {}
    count = 0
    tokens = nltk.word_tokenize(question)
    
    # Identify current question type & lemmas
    question_types = ["what", "when", "who", "whom", "What", "When", "Who", "Whom"]
    Qtype = [i for i in question_types if i in tokens]
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Perform POS tagging to extract POS features
    POStags = nltk.pos_tag(tokens)
    
    # Perform Dependency parsing
    parse, = dependencyParser.raw_parse(question)
    depParseResult = list(parse.triples())
    
    # Extract best sense of the word using LESK
    best_sense = [lesk(question, word) for word in tokens]
    
    # Obtain heads
    headList = GenerateHeads(question, tokens)
    
    # Obtain WordNet features
    hypernyms, hyponyms, meronyms, holonyms, synonyms = WordNetFeatures(question, tokens)

    # Update the corpus dictionary
    count = count + 1
    corpus_dict[count] = {}
    
    corpus_dict[count]["sentence"] = {}
    corpus_dict[count]["sentence"] = question
    
    corpus_dict[count]["type_of_question"] = {}
    corpus_dict[count]["type_of_question"] = Qtype
    
    corpus_dict[count]["tokenized_text"] = {}
    corpus_dict[count]["tokenized_text"] = tokens
    
    corpus_dict[count]["lemma"] = {}
    corpus_dict[count]["lemma"] = lemmas
    
    corpus_dict[count]["tagged"] = {}
    corpus_dict[count]["tagged"] = POStags
    
    corpus_dict[count]["dependency_parse"] = {}
    corpus_dict[count]["dependency_parse"] = depParseResult
    
    corpus_dict[count]["synonyms"] = {}
    corpus_dict[count]["synonyms"] = synonyms
            
    corpus_dict[count]["hypernyms"] = {}
    corpus_dict[count]["hypernyms"] = hypernyms
            
    corpus_dict[count]["hyponyms"] = {}
    corpus_dict[count]["hyponyms"] = hyponyms
            
    corpus_dict[count]["meronyms"] = {}
    corpus_dict[count]["meronyms"] = meronyms
            
    corpus_dict[count]["holonyms"] = {}
    corpus_dict[count]["holonyms"] = holonyms
            
    corpus_dict[count]["head_word"] = {}
    corpus_dict[count]["head_word"] = headList[0]
    
    return corpus_dict
    
    
# Obtains sentence heads
def GenerateHeads(sentence, word_tokens):
    # Set up dependency parser
    dependencyParser = CoreNLPDependencyParser(url='http://localhost:9000')
    headList = []
    
    # Split the sentence
    stripedSen = sentence.strip(" '\"")
    if stripedSen != "":
        # Perform dependency parse
        depParse = dependencyParser.raw_parse(stripedSen)
        parseTree = list(depParse)[0]
        headWord = ""
        headWord = [k["word"] for k in parseTree.nodes.values() if k["head"] == 0][0]
        
        # Appends head if it's not empty
        if headWord != "":
            headList.append([headWord])
            
        # Obtain head word based on two cases
        else:
            for i, pp in enumerate(tagged):
                if pp.startswith("VB"):
                    headList.append([word_tokens[i]])
                    break
            if headWord == "":
                for i, pp in enumerate(tagged):
                    if pp.startswith("NN"):
                        headList.append([word_tokens[i]])
                        break
                        
    # For empty sentence, we just append "" as head
    else:
        headList.append([""])
 
    return headList


# Obtains WordNet Features
def WordNetFeatures(sentence, word_tokens):
    # Creates dictionaries for important word senses
    hypernyms_list = []
    hyponyms_list = []
    meronyms_list = []
    holonyms_list = []
    synonyms_list = []
    
    # Populates the above dictionaries according to the word senses associated with them
    for token in word_tokens:
        # Extracts best sense for each word using LESK
        best_sense = lesk(sentence, token)
        
        if best_sense is not None:
            # Obtains Synonyms
            synonym = token
            if best_sense. lemmas()[0].name() != token:
                synonym = best_sense.lemmas()[0].name()
            synonyms_list.append(synonym)
            
            # Obtains Hypernyms
            if best_sense.hypernyms() != []:
                hypernyms_list.append(best_sense.hypernyms()[0].lemmas()[0].name())
        
            # Obtains Hyponyms
            if best_sense.hyponyms() != []:
                hyponyms_list.append(best_sense.hyponyms()[0].lemmas()[0].name())
            
            # Obtains Meronyms
            if best_sense.part_meronyms() != []:
                meronyms_list.append(best_sense.part_meronyms()[0].lemmas()[0].name())
                
            # Obtains Holonyms
            if best_sense.part_holonyms() != []:
                holonyms_list.append(best_sense.part_holonyms()[0].lemmas()[0].name())
          
        # When there's no best sense, the token itself is the Synonym
        else:
            synonyms_list.append(token)
            
    return hypernyms_list, hyponyms_list, meronyms_list, holonyms_list, synonyms_list
   
    
# Obtains features of the question
def quesFeatures(queFromPipeline):
    similarWords, depElements = [], []
    ques_type = queFromPipeline[1]['type_of_question']
    lemma = queFromPipeline[1]['lemma']
    depParse = queFromPipeline[1]['dependency_parse']
    dep_list = list(list(x) for x in depParse)

    for i in dep_list:
        if i[1] == 'nsubj':
            depElements.append(i[0])
        if i[1] == 'dobj':
            depElements.append(i[0])
            
    dep_list2 = list(list(x) for x in depElements)
    
    similarWords = queFromPipeline[1]['synonyms'] + 
                   queFromPipeline[1]['meronyms'] + 
                   queFromPipeline[1]['hyponyms'] + 
                   queFromPipeline[1]['hypernyms'] + 
                   queFromPipeline[1]['holonyms']
    
    return similarWords, ques_type, lemma, dep_list2


# Checks match
def query_match(theQuery, dep_list2):
    querybody = {
        "query": {
            "dis_max": {
                "queries": [
                    # { "match": { "lemma": {"query": spclQuery,"boost": 2}  }},
                    {"multi_match": {'query': theQuery, "fields": [
                        # "lemma^2.0", "synonyms^0.5", "meronyms^0.1", "holonyms^0.1", "hypernyms^0.1", "hyponyms^0.1"]}},
                        "lemma^2", "ner_tag", "synonyms", "meronyms^0.5", "holonyms^0.5", "hypernyms^0.5", "hyponyms^0.5"]}},]
                    }
                }
            }

    ans = es.search(index="articles", body=querybody)
    answers = ans['hits']['hits']
    depParses, sentences, scores, articles = [], [], [], []
   
    for i in range(len(answers)):
        sentence = ans['hits']['hits'][i]['_source']['sentence']
        sentences.append(sentence)
        
        score = ans['hits']['hits'][i]['_score']
        scores.append(score)
        
        depParse = ans['hits']['hits'][i]['_source']['dependency_parse']
        depParses.append(depParse)
        
        article = ans['hits']['hits'][i]['_source']['file_name']
        articles.append(article)
        
        # print("Sentence: '{}' DepParse: '{}' Score:'{}'".format(sent, depparse, score))
        # print("--------------------------------------------")
    return sentences, scores, depParses, articles


# In[ ]:




