#!/usr/bin/env python
# coding: utf-8

# ## TASK 1 - Feature Extraction by passing text through the NLP Pipeline

# In[1]:


#This file implements all the mandatory feature extraction specified in Task 1.
#    It only prints the features. We save these features along with few more in the actual training file.
#    Features extracted here are:
#    1. Text tokenization into sentences & words
#    2. Word Lemmatization
#    3. Part-of-Speech (POS) tagging
#    4. Dependency Parsing
#    5. WordNet features - Hypernymns, Hyponyms, Meronyms, Holonyms


# ### All necessary imports

# In[2]:


# Turn off unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import all the required packages
import json
import nltk
import string
import urllib
from pprint import pprint
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.wsd import lesk
from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPDependencyParser


# ### Set up globally used variables

# In[3]:


# Start common things globally
stop_words = stopwords.words('english') + list(string.punctuation)
dependencyParser = CoreNLPDependencyParser(url='http://localhost:9000')
wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


# ### Methods to perform Tokenization, Lemmatization, Part-of-Speech (POS) Tagging

# In[4]:


# Performs Word tokenization on sentences
def Tokenization(sentence):
    tokens = [i for i in nltk.word_tokenize(sentence.lower()) if i not in stop_words]
    return tokens


# Performs Word Lemmatization : Uses context
def Lemmatization(word_tokens):
    lemmas = []
    for token in word_tokens:
        lemmas.append(wordnet_lemmatizer.lemmatize(token))
    return lemmas


# Performs POS tagging
def POSTagging(sentence):
    word_tokens = [i for i in nltk.word_tokenize(sentence.lower()) if i not in stop_words]
    POStags = nltk.pos_tag(word_tokens)
    return POStags   


# ### Method to perform Dependency Parsing

# In[5]:


# Performs Dependency Parsing
def DependencyParsing(sentence):
    # Perform dependency parsing
    parse, = dependencyParser.raw_parse(sentence)
    
    # Dependency parsing to parse tree based patterns as features
    depParseResult = list(parse.triples())
    
    return depParseResult


# ### Method to extract WordNet features : Synonyms, Meronyms, Holonyms, Hypernyms, Hyponyms

# In[6]:


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
   


# ### Method that helps us to pass the articles through the NLP pipeline

# In[12]:


# NLP pipeline through which all the articles & question will pass
def NLP_Pipeline(sentence):
    word_tokens = Tokenization(sentence)
    print("\nWord Tokenization : Done")
    print(word_tokens)
    
    word_POStags = POSTagging(sentence)
    print("\nPOS Tagging : Done")
    print(word_POStags)
    
    word_lemmas = Lemmatization(word_tokens)
    print("\nWord Lemmatization : Done")
    print(word_lemmas)

    hypernyms, hyponyms, meronyms, holonyms, synonyms = WordNetFeatures(sentence, word_tokens)
    print("\nWordNet Feature Extraction : Done")
    
    print("\nSynonyms")
    print(synonyms)
    
    print("\nHolonyms")
    print(holonyms)
    
    print("\nMeronyms")
    print(meronyms)
    
    print("\nHyponyms")
    print(hyponyms)
    
    print("\nHypernyms")
    print(hypernyms)
            
    depParse = DependencyParsing(sentence)
    print("\nDependency Parsing : Done")
    pprint(depParse)


# ### Execute Task 1

# In[14]:


# Get contents from the sample file
urlPath = "https://raw.githubusercontent.com/SaiManasaVedantam/NLP-QA-System-Datasets/main/"
fileName = urlPath + "Sample-6.txt"
response = urllib.request.urlopen(fileName)
webContents = response.read()
stringTypeData = webContents.decode("utf-8")
content = stringTypeData

print("Started processing the sample file")

# Get tokenized sentences
sentences = []
sentences.extend(tokenizer.tokenize(content))

# Sentence count
#print("Total Sentences After splitting the document: ", len(sentences))
print("Extracting features for each sentence in the file...")
    
# Extracting words
for sen in sentences:
    print("\n------SENTENCE------")
    print(sen)
    NLP_Pipeline(sen)
        
print("\nCompleted processing the sample file")    
print("\nSample Task 1 Successfully Completed !!!")
    


# ### Extracting features from a sample question

# In[15]:


sample_que = "What are being analyzed here?"
print(sample_que)
NLP_Pipeline(sample_que)

