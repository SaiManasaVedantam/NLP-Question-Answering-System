#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Turn off unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import all the required packages
import json
import nltk
import string
import urllib
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.wsd import lesk
from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPDependencyParser

# Start common things globally
stop_words = stopwords.words('english') + list(string.punctuation)
dependencyParser = CoreNLPDependencyParser(url='http://localhost:9000')
namedEntityTagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')

# Performs Word tokenization on sentences
def Tokenization(sentence):
    tokens = [i for i in nltk.word_tokenize(sentence.lower()) if i not in stop_words]
    return tokens


# Performs Word Lemmatization
def Lemmatization(word_tokens):
    lemmas = []
    for token in word_tokens:
        lemmas.append(wordnet_lemmatizer.lemmatize(token))
    return lemmas


# Performs POS tagging
def POSTagging(word_tokens):
    POStags = nltk.pos_tag(word_tokens)
    return POStags   


# Obtains sentence heads
def getHeads(sentence, word_tokens):
    # Create a head list to add the heads
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
   
    
# Performs Dependency Parsing
def DependencyParsing(sentence):
    # Perform dependency parsing
    parse, = dependencyParser.raw_parse(sentence)
    
    # Dependency parsing to parse tree based patterns as features
    depParseResult = list(parse.triples())
    
    return depParseResult
    
    
# Obtains Named Entities
def NamedEntities(sentence, tokens):
    # Word tokenize again and use them if NEs are present
    namedTokens = nltk.word_tokenize(sentence)
    NEtags = None
    
    try:
        NEtags = namedEntityTagger.tag(namedTokens)
    except:
        NEtags = namedEntityTagger.tag(tokens)
        
    return NEtags

# NLP pipeline through which all the articles & question will pass
def NLP_Pipeline(sentence, count, corpus_dict, articleName = None):
    #print("\n------SENTENCE------")
    #print(sen)

    word_tokens = Tokenization(sentence)
    #print("\nWord Tokenization : Done")
    #print(word_tokens)

    word_NEtags = NamedEntities(sentence, word_tokens)
    #print("\nNamed Entity Tagging : Done")
    #print(word_NEtags)
    
    word_lemmas = Lemmatization(word_tokens)
    #print("Word Lemmatization : Done")
    #print(word_lemmas)

    word_POStags = POSTagging(word_tokens)
    #print("POS Tagging : Done")
    #print(word_POStags)

    hypernyms, hyponyms, meronyms, holonyms, synonyms = WordNetFeatures(sentence, word_tokens)
    #print("WordNet Feature Extraction : Done")
    #print(holonyms)
            
    depParse = DependencyParsing(sentence)
    #print("Dependency Parsing : Done")
    #print(depParse)

    headList = getHeads(sentence, word_tokens)
    #print("Obtaining Heads : Done")
    #print(headList)

    # Process data format to suit the Elastic Search requirements
    count = count + 1
    corpus_dict[count] = {}
            
    corpus_dict[count]["sentence"] = {}
    corpus_dict[count]["sentence"] = sentence
            
    corpus_dict[count]["tokenized_text"] = {}
    corpus_dict[count]["tokenized_text"] = word_tokens
            
    corpus_dict[count]["lemma"] = {}
    corpus_dict[count]["lemma"] = word_lemmas
    
    corpus_dict[count]["ner_tag"] = {}
    if articleName is not None:
        corpus_dict[count]["ner_tag"] = str(dict(word_NEtags))
    else:
        corpus_dict[count]["ner_tag"] = dict(word_NEtags)
            
    corpus_dict[count]["tagged"] = {}
    corpus_dict[count]["tagged"] = word_POStags
            
    corpus_dict[count]["dependency_parse"] = {}
    corpus_dict[count]["dependency_parse"] = depParse
            
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
    
    # For question, we don't have the article name and then it will have a questionType
    if articleName is not None:
        corpus_dict[count]["file_name"] = {}
        corpus_dict[count]["file_name"] = articleName
        
        
    # For question, we should add the question type
    else:
        tokens = nltk.word_tokenize(sentence)
        questionTypes = ["who", "when", "what", "whom"]
        queType = [i for i in questionTypes if i in tokens]
        corpus_dict[count]["type_of_question"] = {}
        corpus_dict[count]["type_of_question"] = queType
    
    return count, corpus_dict
    
    
# Main method
if __name__ == "__main__":
    # List of all article names in the repository
    articleNames = ["109.txt", "111.txt", "151.txt", "160.txt", "177.txt", 
                    "179.txt","181.txt", "196.txt", "199.txt", "220.txt", 
                    "222.txt", "226.txt", "247.txt", "273.txt", "281.txt", 
                    "282.txt", "285.txt", "287.txt", "288.txt", "297.txt", 
                    "304.txt", "342.txt", "347.txt", "360.txt", "390.txt", 
                    "400.txt", "428.txt", "56.txt", "58.txt", "6.txt"] 
    fileCount = len(articleNames)
    
    content = ""
    folderPath = "https://raw.githubusercontent.com/SaiManasaVedantam/NLP-QA-System-Datasets/main/Articles/"
    for i in range(fileCount):
        print("\nStarted Processing File : " + articleNames[i])
        fileName = folderPath + articleNames[i]
        response = urllib.request.urlopen(fileName)
        webContents = response.read()
        stringTypeData = webContents.decode("utf-8")
        content = stringTypeData
        count = 0
        corpus_dict = {}

        # Obtain wordnet lemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()

        # Get tokenized content
        sentences = []
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentences.extend(tokenizer.tokenize(content))

        # Sentence count
        #print("Total Sentences After splitting the document: ", len(sentences))
        print("Extracting features for each sentence in the file...")
    
        # Extracting words
        for sen in sentences:
            count, corpus_dict = NLP_Pipeline(sen, count, corpus_dict, articleNames[i])
                
        output_name = '../Pipeline-Output/Parsed-' + articleNames[i]
        with open(output_name, 'w+', encoding='utf8') as output_file:
            json.dump(corpus_dict, output_file,  indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        
        print("Completed Processing File : " + articleNames[i])
        
    print("\nTask 1 Successfully Completed !!!")
    


# In[3]:


# Turn off unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import all the required packages
import ssl
import json
import urllib
import requests
from elasticsearch import Elasticsearch
from elasticsearch import RequestsHttpConnection

# Main method
if __name__ == "__main__":
    # List of all article names in the repository
    articleNames = ["109.txt", "111.txt", "151.txt", "160.txt", "177.txt", 
                    "179.txt","181.txt", "196.txt", "199.txt", "220.txt", 
                    "222.txt", "226.txt", "247.txt", "273.txt", "281.txt", 
                    "282.txt", "285.txt", "287.txt", "288.txt", "297.txt", 
                    "304.txt", "342.txt", "347.txt", "360.txt", "390.txt", 
                    "400.txt", "428.txt", "56.txt", "58.txt", "6.txt"] 
    fileCount = len(articleNames)
    
    # Setup Elastic Search
    elastic = Elasticsearch([{'host': 'localhost', 'port': 9200, 'use_ssl' : False, 'ssl_verify' : False}], timeout=30, max_retries=10)
    
    # Obtain requests from the page
    req = requests.get("http://localhost:9200", verify=False)
    
    # Use indexing
    idx = 1
    
    content = ""
    folderPath = "https://raw.githubusercontent.com/SaiManasaVedantam/NLP-QA-System-Datasets/main/Pipeline-Output/Parsed-"
    for i in range(fileCount):
        print("\nStarted Processing File : " + articleNames[i])
        fileName = folderPath + articleNames[i]
        response = urllib.request.urlopen(fileName)
        webContents = response.read()
        stringTypeData = webContents.decode("utf-8")
        content = stringTypeData
        
        # Obtain Json data from file contents
        jsonFile = json.loads(content)
        
        # Creating new index "articles" for each line in the article
        for key, value in jsonFile.items():
            elastic.index(index = "articles", doc_type = "text", id = idx, body = value)
            # print("Here")
            idx += 1
            
        print("Finished Processing File : " + articleNames[i])
        
    print("Elastic Search Successfully Completed !!!")
    
    


# In[4]:


# Obtains features in the question by using the result obtained from the NLP Pipeline
def questionFeatures(question):
    # Get all the wordnet features
    WNfeatures = question[1]['synonyms'] + question[1]['meronyms'] + question[1]['hyponyms'] + question[1]['holonyms'] + question[1]['hypernyms']
       
    # Create hints for easy search using Named Entities and the Sentence head
    head = question[1]['head_word'][0]
    NEs = question[1]['ner_tag']
    NEhints = ""
    namedEntities = []
    
    for word, entity in NEs.items():
        namedEntities.append(entity)
        if entity == 'ORGANIZATION' or entity == 'LOCATION' or entity == 'PERSON':
            NEhints += " " + word + " "
            
    NEhints += " " + head + " "
    
    # Obtain question type and other features
    queType = question[1]['type_of_question']
    lemmas = question[1]['lemma']
    depParse = question[1]['dependency_parse']

    depList = list(list(x) for x in depParse)
    depElements = []
    
    for i in depList:
        if i[1] == 'nsubj' or i[1] == 'dobj':
            depElements.append(i[0])
     
    # Retrieve main elements from the dependency parse result
    dependencyList = list(list(x) for x in depElements)

    return NEhints, WNfeatures, queType, lemmas, dependencyList


# Check and obtain matched sentences using the query string
def GetMatchedSentences(queryStr, dependencyList):
    querybody = {
        "query": {
            "dis_max": {
                "queries": [
                    # { "match": { "lemma": {"query": spclQuery,"boost": 2}  }},
                    {"multi_match": {'query': queryStr, "fields": [
                        # "lemma^2.0", "synonyms^0.5", "meronyms^0.1", "holonyms^0.1", "hypernyms^0.1", "hyponyms^0.1"]}},
                        "lemma^2", "ner_tag", "synonyms", "meronyms^0.5", "holonyms^0.5", "hypernyms^0.5", "hyponyms^0.5"]}},
                    ]
                }
            }
        }

    result = elastic.search(index = "articles", body=querybody)
    answers = result['hits']['hits']
    depParses, sentences, scores, articles, NEs = [], [], [], [], []
    
    for i in range(len(answers)):
        sentence = result['hits']['hits'][i]['_source']['sentence']
        sentences.append(sentence)
        
        score = result['hits']['hits'][i]['_score']
        scores.append(score)
        
        depParse = result['hits']['hits'][i]['_source']['dependency_parse']
        depParses.append(depParse)
        
        article = result['hits']['hits'][i]['_source']['file_name']
        articles.append(article)
        
        NE = result['hits']['hits'][i]['_source']['ner_tag']
        NEs.append(NE)
        
    return sentences, scores, depParses, articles, NEs


# In[19]:


# Find the match score to know how well a statement is matched
def FindScore(queType, NEhints, sentences, scores, depParses, articles, NEs, dependencyList):
    # Add additional World Knowledge to implement a much deeper NLP pipeline
    # Named Entities
    organizations = ['ORGANIZATION']
    persons = ['PERSON']
    locations = ['LOCATION', 'PLACE', 'CITY', 'COUNTRY', 'STATE_OR_PROVINCE']
    times = ['TIME', 'DATE', 'NUMBER']
    
    # Feeding world knowledge for a deeper pipeline
    deaths = ['die', 'died', 'assassination']
    births = ['born', 'birth', 'life']
    keywords = NEhints.split()
    keywords = [item.lower() for item in keywords]
    
    # Obtain relations using Dependency Parse result
    count = 0
    relations = []
    for dep in depParses:
        for i in dep:
            if i[1] == 'nsubj' or i[1] == 'dboj':
                if i[0] in dependencyList:
                    relations.append([count,i[0]])
        count += 1

    # Get question type
    questionType = queType[0].lower()
    answers = []
    
    # Set for relation
    for reln in relations:
        idx = relations[0]
        scores[idx] += 100
        # print(sentenses[ano[0]])

    # Handle different question types
    if questionType == 'who' or questionType == 'whom':
        for NE in NEs:
            # Obtain all the named entities which are initially stored as a stringified dictionary
            NEdict = eval(NE)
            ans = []
            for key, value in NEdict.items():
                if value in persons or organizations:
                    ans.append(key)
                if (ans != [] and key == ',') or (ans != [] and key == 'and'):
                    ans.append(key)
                    
            answers.append(' '.join(ans))

    if questionType == 'when':
        for NE in NEs:
            # Obtain all the named entities which are initially stored as a stringified dictionary
            NEdict = eval(NE)
            ans = []
            for key, value in NEdict.items():
                if value in times and dateparser.parse(key) is not None:
                    ans.append(key)

            answers.append(' '.join(ans))

    """if questionType == 'what':
        for NE in NEs:
            # Obtain all the named entities which are initially stored as a stringified dictionary
            NEdict = eval(NE)
            ans = []
            for key, value in NEdict.items():
                if value in locations or value in organizations:
                    ans.append(key)
                if (ans != [] and key == ',') or (ans != [] and key == 'and'):
                    ans.append(key)
                    
            answers.append(' '.join(ans))"""

    
    for idx in range(len(answers)):
        if len(answers[idx]) < 3:
            scores[idx] -= 100
    
    """# Level 2 handling for When questions as it can also be about births & deaths
    dieconcept = 0       
    if questionType == 'when':
        for key in range(len(sentences)):
            for j in deaths:
                if j in sentences[key]:
                    pattern = r"\((.*?)\)"
                    try:
                        matched = re.findall(pattern, sentences[key])
                        splits = matched[0].split(' ')
                        splitjoin = ' '.join(splits[4:])
                        answers[key] = splitjoin
                        dieconcept = 1
                    except:
                        pass
                    scores[key] += 50
                    
        if dieconcept == 0:
            for key in range(0,len(sentences)):
                for j in births:
                    if j in sentences[key]:
                        pattern = r"\((.*?)\)"
                        try:
                            matched = re.findall(pattern, sentences[key])
                            splits = matched[0].split(' ')
                            splitjoin = ' '.join(splits)
                            answers[key] = splitjoin
                            dieconcept = 1
                        except:
                            pass
                        scores[key] += 10"""

    results = zip(sentences, scores)
    sortedResults = sorted(results, key = lambda x: x[1])

    return reversed(sortedResults)


# In[28]:


question = "What was the capital of the Safavid Dynasty?"
count = 0

# Pass the question through NLP pipeline
count, queFromPipeline = NLP_Pipeline(question.lower(), count, corpus_dict, None)

# Obtain features of the question which already passed through the NLP pipeline
NEhints, WNfeatures, queType, lemmas, dependencyList = questionFeatures(queFromPipeline)

# Form a query string with best possible features for reliable answers
queryStr = NEhints + " " +' '.join(WNfeatures) + " " + ' '.join(lemmas)

# Run the match query against indexed articles and obtain matched sentences
sentences, scores, depParses, articles, NEs = GetMatchedSentences(queryStr, dependencyList)

# Obtain only the relevant sentences
relevantSentences = FindScore(queType, NEhints, sentences, scores, depParses, articles, NEs, dependencyList)

for sent in relevantSentences:
    print(sent, "\n")
    
    


# In[ ]:



"""
WORK TO DO:

1. Handle case-sensitive questions.
-> Currently converting to lower. Need to check if it's ok.

2. Handle WHAT case.
-> Currently considering locations & organizations but that's not good.

3. Handle output format.
-> We should produce output as specified in the description.

4. Handle multi-type questions.
-> Easy, Medium, Hard.

5. Find a way to produce top ranked result.
-> Currently unreliable system.

"""

