#!/usr/bin/env python
# coding: utf-8

# # Closed-Domain Question-Answering System

# ### All required package imports

# In[2]:


# Turn off unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import all the required packages
import csv
import json
import nltk
import string
import urllib
import os.path
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.wsd import lesk
from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer


# ### Global declarations 

# In[3]:


# Start common things globally
stop_words = stopwords.words('english') + list(string.punctuation)
dependencyParser = CoreNLPDependencyParser(url='http://localhost:9000')
namedEntityTagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
wordnet_lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

UNKNOWN_ANSWER = "****** Our System did not learn the knowledge required to answer this query ******"
UNAVAILABLE_ARTICLE = "unavailable"


# ### Methods to perform Tokenization, Lemmatization, Stemming and Part-of-Speech (POS) tagging

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


# Performs Stemming : Uses word stem
def Stemming(word_tokens):
    stems = [porter.stem(word) for word in word_tokens]
    return stems


# Performs POS tagging
def POSTagging(sentence):
    word_tokens = [i for i in nltk.word_tokenize(sentence.lower()) if i not in stop_words]
    POStags = nltk.pos_tag(word_tokens)
    return POStags   


# ### Methods to perform Dependency Parsing and Named Entity identification

# In[5]:


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


# ### Method to obtain sentence Heads

# In[6]:


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


# ### Method to obtain WordNet features like Synonyms, Meronyms, Hypernyms, Hyponyms & Holonyms 

# In[7]:


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
   


# ### Method to build NLP Pipeline end-to-end using all the features we used above

# In[8]:


# NLP pipeline through which all the articles & question will pass
def NLP_Pipeline(sentence, count, data_dict, articleName = None):
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
    
    word_stems = Stemming(word_tokens)
    #print("Word Stemming : Done")
    #print(word_stems)

    word_POStags = POSTagging(sentence)
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
    data_dict[count] = {}
            
    data_dict[count]["sentence"] = {}
    data_dict[count]["sentence"] = sentence
            
    data_dict[count]["tokenized_text"] = {}
    data_dict[count]["tokenized_text"] = word_tokens
            
    data_dict[count]["lemma"] = {}
    data_dict[count]["lemma"] = word_lemmas
    
    data_dict[count]["stems"] = {}
    data_dict[count]["stems"] = word_stems
    
    data_dict[count]["ner_tag"] = {}
    if articleName is not None:
        data_dict[count]["ner_tag"] = str(dict(word_NEtags))
    else:
        data_dict[count]["ner_tag"] = dict(word_NEtags)
            
    data_dict[count]["tags"] = {}
    data_dict[count]["tags"] = word_POStags
            
    data_dict[count]["dependency_parse"] = {}
    data_dict[count]["dependency_parse"] = depParse
            
    data_dict[count]["synonyms"] = {}
    data_dict[count]["synonyms"] = synonyms
            
    data_dict[count]["hypernyms"] = {}
    data_dict[count]["hypernyms"] = hypernyms
            
    data_dict[count]["hyponyms"] = {}
    data_dict[count]["hyponyms"] = hyponyms
            
    data_dict[count]["meronyms"] = {}
    data_dict[count]["meronyms"] = meronyms
            
    data_dict[count]["holonyms"] = {}
    data_dict[count]["holonyms"] = holonyms
            
    data_dict[count]["head_word"] = {}
    data_dict[count]["head_word"] = headList[0]
    
    # For question, we don't have the article name and then it will have a questionType
    if articleName is not None:
        data_dict[count]["file_name"] = {}
        data_dict[count]["file_name"] = articleName
        
        
    # For question, we should add the question type
    else:
        tokens = nltk.word_tokenize(sentence)
        questionTypes = ["who", "when", "what", "whom", "whose"]
        queType = [i for i in questionTypes if i in tokens]
        data_dict[count]["question_type"] = {}
        data_dict[count]["question_type"] = queType
    
    return count, data_dict
    


# ## TASK 1 - Building an NLP Pipeline end-to-end in training phase by using 30 articles

# In[9]:


# Builds NLP Pipeline in the Task 1

def task1():
    # List of all article names in the repository
    articleNames = ["109.txt", "111.txt", "151.txt", "160.txt", "177.txt", 
                    "179.txt","181.txt", "196.txt", "199.txt", "220.txt", 
                    "222.txt", "226.txt", "247.txt", "273.txt", "281.txt", 
                    "282.txt", "285.txt", "287.txt", "288.txt", "297.txt", 
                    "304.txt", "342.txt", "347.txt", "360.txt", "390.txt", 
                    "400.txt", "428.txt", "56.txt", "58.txt", "6.txt"] 
    fileCount = len(articleNames)
    
    content = ""
    urlPath = "https://raw.githubusercontent.com/SaiManasaVedantam/NLP-QA-System-Datasets/main/Articles/"

    for i in range(fileCount):
        print("\nStarted Processing File : " + articleNames[i])
        fileName = urlPath + articleNames[i]
        response = urllib.request.urlopen(fileName)
        webContents = response.read()
        stringTypeData = webContents.decode("utf-8")
        content = stringTypeData
        count = 0
        data_dict = {}

        # Get tokenized sentences
        sentences = []
        sentences.extend(tokenizer.tokenize(content))

        # Sentence count
        #print("Total Sentences After splitting the document: ", len(sentences))
        print("Extracting features for each sentence in the file...")

        # Extracting words
        for sen in sentences:
            count, data_dict = NLP_Pipeline(sen, count, data_dict, articleNames[i])

        output_name = '../Pipeline-Output/Parsed-' + articleNames[i]
        with open(output_name, 'w+', encoding='utf8') as output_file:
            json.dump(data_dict, output_file,  indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)

        print("Completed Processing File : " + articleNames[i])

    print("\nTask 1 Successfully Completed !!!")


# ## TASK 2 - Indexing and obtaining answer for the given query

# In[10]:


# Import all the required packages
import ssl
import json
import urllib
import requests
from elasticsearch import Elasticsearch
from elasticsearch import RequestsHttpConnection

# Setup Elastic Search
elastic = Elasticsearch([{'host': 'localhost', 'port': 9200, 'use_ssl' : False, 'ssl_verify' : False}], timeout=30, max_retries=10)

# Obtain requests from the page
req = requests.get("http://localhost:9200", verify=False)


# ### Implements sentence indexing

# In[11]:


# Indexing using Elastic Search

def task2_part1():
    # List of all article names in the repository
    articleNames = ["109.txt", "111.txt", "151.txt", "160.txt", "177.txt", 
                    "179.txt","181.txt", "196.txt", "199.txt", "220.txt", 
                    "222.txt", "226.txt", "247.txt", "273.txt", "281.txt", 
                    "282.txt", "285.txt", "287.txt", "288.txt", "297.txt", 
                    "304.txt", "342.txt", "347.txt", "360.txt", "390.txt", 
                    "400.txt", "428.txt", "56.txt", "58.txt", "6.txt"] 
    fileCount = len(articleNames)

    # Use indexing
    idx = 1

    content = ""
    urlPath = "https://raw.githubusercontent.com/SaiManasaVedantam/NLP-QA-System-Datasets/main/Pipeline-Output/Parsed-"

    for i in range(fileCount):
        print("\nStarted Processing File : " + articleNames[i])
        fileName = urlPath + articleNames[i]
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

    print("\nElastic Search Successfully Completed !!!")


# ### Uses the indexed content, generate question features from the pipeline & obtain the result

# In[12]:


# Import all necessary packages
import dateparser
import ast
import re


# ### Extracts features associated with the query by passing it through the NLP pipeline

# In[13]:


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
        if entity == 'TIME' or entity == 'DATE' or entity == 'NUMBER':
            NEhints += " " + word + " "
        
    NEhints += " " + head + " "
    
    # Obtain question type and other features
    queType = question[1]['question_type']
    lemmas = question[1]['lemma']
    stems = question[1]['stems']
    depParse = question[1]['dependency_parse']

    depList = list(list(x) for x in depParse)
    depElements = []
    
    for i in depList:
        if i[1] == 'nsubj' or i[1] == 'dobj':
            depElements.append(i[0])
     
    # Retrieve main elements from the dependency parse result
    dependencyList = list(list(x) for x in depElements)

    return NEhints, WNfeatures, queType, lemmas, stems, dependencyList


# ### Query the indexed content with features & their importance

# In[14]:


# Check and obtain matched sentences using the query string
def GetMatchedSentences(queryStr, dependencyList):
    # Used Lemmas with 2.2 importance
    # Named Entities, Synonyms with 1.9 importance
    # Holonyms, Meronyms, POS tags with 0.2 importance
    # Hypernyms, Hyponyms with 0.4 importance
    # Heads with 1.6 importance
    querybody = {
        "query": {
            "dis_max": {
                "queries": [
                    # Boost the value of each feature as per the need
                    {"multi_match": {'query': queryStr, "fields": [
                        "lemma^2.2", "ner_tag^1.9", "synonyms^1.9", "holonyms^0.2", "meronyms^0.2", "hypernyms^0.4", "hyponyms^0.4", "head_word^1.6", "tags^0.2"]}},
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


# ### Find scores associated with each sentence that satisfies the query & implement deeper NLP pipeline

# In[15]:


# Find the match score to know how well a statement is matched
def FindScore(queType, NEhints, sentences, scores, depParses, articles, NEs, dependencyList):
    # Add additional World Knowledge to implement a much deeper NLP pipeline
    # Named Entities
    
    # IMPLEMENTS A DEEPER NLP PIPELINE USING THE ADDITIONAL FEATURES
    organizations = ['ORGANIZATION']
    persons = ['PERSON']
    locations = ['LOCATION', 'PLACE', 'CITY', 'COUNTRY', 'STATE_OR_PROVINCE']
    times = ['TIME', 'DATE', 'NUMBER']
    # times2 = ['BC', 'AD', 'CENTURY']
    
    # Feeding world knowledge for a deeper pipeline
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
    if len(queType) == 0:
        return None
    
    #print(type(queType))
    questionType = queType[0].lower()
    answers = [] 

    # Handle different question types
    if questionType == 'who' or questionType == 'whom' or questionType == 'whose':
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
            
    for idx in range(len(answers)):
        if len(answers[idx]) < 3:
            scores[idx] -= 100

    results = zip(sentences, articles, scores)
    sortedResults = sorted(results, key = lambda x: x[2])

    return reversed(sortedResults)


# ### Method to retrieve data from the Validation set

# In[16]:


# Obtains contents from validation set & returns list of questions
def getValidationData():
    valFile = open("Validation-Data.txt", encoding='UTF-8')
    valData = valFile.read()
    valData = valData.strip()
    valList = valData.split("\n")
    
    totalQue = []
    totalAns = []
    
    for articleQueList in valList:
        queList = articleQueList.split("]]")
        questions = ast.literal_eval(queList[0] + "]]")
        
        for QApair in questions[1]:
            question = re.sub('\?', '', QApair[0])
            totalQue.append(question)
            answer = QApair[1]
            totalAns.append(answer)
 
    return totalQue, totalAns


# ### Method to obtain the best possible answer for the query

# In[17]:


# Obtains best possible answer for the query
def getAnswer(question):
    count = 0
    data_dict = {}

    # Pass the question through NLP pipeline
    count, queFromPipeline = NLP_Pipeline(question.lower(), count, data_dict, None)

    # Obtain features of the question which already passed through the NLP pipeline
    NEhints, WNfeatures, queType, lemmas, stems, dependencyList = questionFeatures(queFromPipeline)

    # Form a query string with best possible features for reliable answers
    queryStr = NEhints + " " +' '.join(WNfeatures) + " " + ' '.join(lemmas) +  " " +' '.join(stems)

    # Run the match query against indexed articles and obtain matched sentences
    sentences, scores, depParses, articles, NEs = GetMatchedSentences(queryStr, dependencyList)
    #print(articles)

    # Obtain only the relevant sentences
    relevantSentences = FindScore(queType, NEhints, sentences, scores, depParses, articles, NEs, dependencyList)
    if relevantSentences is None:
        return UNKNOWN_ANSWER, UNAVAILABLE_ARTICLE
    
    #print(tuple(relevantSentences))

    answer_candidates = []
    article_candidates = []

    for ans in relevantSentences:
        #print(ans)
        answer_candidates.append(ans[0])
        article_candidates.append(ans[1])

    # Result sentence
    answer = None if len(answer_candidates) == 0 else answer_candidates[0]
    article = None if len(article_candidates) == 0 else article_candidates[0]
    
    return answer, article


# ### Method to run the NLP pipeline on Validation set & obtain accuracy

# In[18]:


# Runs the pipeline on the validation set and obtains accuracy
def validateAndGetAccuracy():
    questions, answers = getValidationData()
    total = len(questions)
    correct = 0
    idx = 1
  
    for que, expectedAns in zip(questions, answers):
        #print("\n", que)
            
        obtainedAns, obtainedArticle = getAnswer(que)
        #print(obtainedAns)
        #print(obtainedArticle)

        if obtainedAns is None:
            continue
                
        elif expectedAns in obtainedAns:
            correct += 1
            
        # Tracks how many questions are completed & prints status for every 500 questions
        if idx % 500 == 0:
            print("Completed answering", idx, "questions in Validation Data")
        idx += 1
        
    errors = total - correct
    accuracy = (correct / total) * 100
    print("Correct: ", correct, "\t Total: ", total, "\t Incorrect: ", errors)
    print("Validation Accuracy: ", round(accuracy, 2), "%")
    


# ### Runs the NLP pipeline on the sample input file

# In[19]:


# Runs the pipeline on the sample questions with different levels of complexity
def runPipelineOnSample(inputFile, outputFile):
    questions = readInput(inputFile)
    answers = readInput(outputFile)
    for question, answer in zip(questions, answers):
        obtainedAns, obtainedArticle = getAnswer(question)
        print("\nExpected: ", answer)
        print("\nObtained: ", obtainedAns)
    


# ## TASK 3 - Read an input file, run the NLP pipeline, obtain answers in csv format

# ### Methods to read input data & to check if the file specified is valid

# In[20]:


# Reads content from the input file using fileName & returns questions
# It considers the relative path to be in the same location as this ipynb
def readInput(fileName):
    inputData = open(fileName).read()
    inputData = inputData.strip()
    questions = inputData.splitlines()
   
    return questions

# Checks if the given file exists in the path
def checkFile(fileName):
    if os.path.isfile(fileName):
        return True
    return False


# ### Method to process the input file and generate output in csv format

# In[21]:


# Produces output in the required format & save as .csv
def processAndGenerateOutput(questions):
    # Saves the output for all questions in a list
    headers = ["Question", "Answer's Article-ID", "Answer"]
    finalOutput = []
    finalOutput.append(headers)
    
    for que in questions:
        obtainedAns, obtainedArticle = getAnswer(que)
        
        # Stores output for each question in a list
        outputData = []
        outputData.append(que)
        outputData.append(obtainedArticle)
        outputData.append(obtainedAns)

        # Appends each question's output to the final output list
        finalOutput.append(outputData)
        
        with open('Output.csv', 'w+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerows(finalOutput)
        
    print("The output CSV file is ready!!")


# ## TRAINING & TESTING

# ### Use a flag to train the system if it isn't already trained

# In[22]:


# Setup a boolean flag to check if the System is already trained
isTrained = False


# ### Method to train the System

# In[23]:


def TrainSystem():
    # Average training time to pass articles through Pipeline : 25 mins
    task1()
    
    # Average training time to index the articles : 7 mins
    task2_part1()
    
    # Sets the flag to True to avoid training as long as the session is active
    global isTrained
    isTrained = True


# ### Main method

# In[24]:


if __name__ == '__main__':  
    # Train the system once
    if isTrained is False:
          TrainSystem()
        
    print("\nTraining the Question Answering System is successfully completed !!")
    
    # Filename for testing: Sample-Questions.txt  (8 Correct, 3 Incorrect)
    fName = input("\nEnter input file name along with extension (.txt only): ")
    exists = True
    exists = checkFile(fName)
    
    if exists == False:
        print("\nFile does not exist in the expected path !!")
        print("Retry with valid file name !!")
        
    else:
        questions = readInput(fName)
        print("\nStarted obtained answers for the questions posed...")
        processAndGenerateOutput(questions)
        
        # Average time to run on the Validation dataset : 5 mins
        print("\nAccuracy on Validation Dataset: ")
        validateAndGetAccuracy()
        


# ## Demonstrating the capability of our System in handling several features & scenarios 

# ### Handling questions that doesn't come under the scope of question

# In[25]:


# No article contains the answer 
# We don't know the answer to this question because it is beyond the scope of data we fed to system

print(getAnswer("IIIIII DDDDD KKKKKK MMMMMM LLLLLLLL"))
print(getAnswer("Where did Bell spend time in canadian Home?"))


# ### Handling questions that have more confusing question type

# In[26]:


print(getAnswer("what what why why how how when where"))


# ### Handling Synonyms

# In[27]:


# Pros = Advantages, Neutralize = Offset
# Expected answer:
# These advantages offset the high stress, physical exertion costs, and other risks of the migration, 109.txt

print(getAnswer("What pros neutralize risk of migration?"))


# ### Handling Hyponyms

# In[28]:


# Magazine is the direct hyponym of Publication
# Expected answer:
# Japanese comics magazine typically run to hundreds of pages.

print(getAnswer("What Japanese publication run to many pages?"))


# ### Handling Hypernyms

# In[29]:


# Provenance is the direct hypernym of Birthplace
# Expected answer:
# Geneva is the birthplace of the Red Cross and Red Crescent Movement and the Geneva Conventions and, since 2006, 
# hosts the United Nations Human Rights Council

print(getAnswer("What is the provenance of Red Cross?"))


# ### Handling Lemmas

# In[30]:


# Question types who, whom and whose are centered at who being the root word
# Pagan's collapse was followed by 250 years of political fragmentation that lasted well into the 16th century.
# Expected answer:

print(getAnswer("Whose collapse caused political fragmentation?"))


# ### Handling Sentence Heads

# In[31]:


# Sentence head : Blend
# Expected answer:
# It may be blended with the hot bitumen in tanks, but its granular form allows it to be fed in the mixer or
# in the recycling ring of normal asphalt plants.

print(getAnswer("What can be blended in tanks?"))


# ### Handling Named Entities

# In[32]:


# 150 or 302 are NUMBER entities
# Expected answer:
# Asphalt/bitumen is typically stored and transported at temperatures around 150 °C (302 °F).

print(getAnswer("At what temperature is Asphalt stored?"))


# ### Handling Meronyms

# In[33]:


# Plumbing is a part of Construction or Building work
# Expected answer:
# Some children undertook work as apprentices to respectable trades, such as building or as domestic servants 
# (there were over 120,000 domestic servants in London in the mid-18th century).

print(getAnswer("When did children start apprentice work in plumbing or house servants for trades?"))


# ### Handling Holonyms

# In[34]:


# Bird is the holonym of feather
# Expected answer:
# In addition, the feathers of a bird suffer from wear-and-tear and require to be molted. 

print(getAnswer("What part of bird need to be molted to reduce suffering?"))


# ### Testing on a sample Easy questions dataset

# In[36]:


runPipelineOnSample("Easy-Que.txt", "Easy-Ans.txt")


# ### Testing on a sample Medium questions dataset

# In[37]:


runPipelineOnSample("Medium-Que.txt", "Medium-Ans.txt")


# ### Testing on a sample Hard questions dataset

# In[41]:


runPipelineOnSample("Hard-Que.txt", "Hard-Ans.txt")


# ### Testing on a sample of mix of questions dataset

# In[40]:


runPipelineOnSample("Sample-Que.txt", "Sample-Ans.txt")

