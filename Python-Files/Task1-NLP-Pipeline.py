#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Turn off unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import all the required packages
import json
import nltk
import urllib
import en_core_web_sm
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.wsd import lesk
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.corpus import stopwords


# Performs Word tokenization on sentences
def Tokenization(sentence):
    tokens = nltk.word_tokenize(sentence)
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
   
    
# Performs Dependency Parsing
def DependencyParsing(sentence):
    dependencyParser = CoreNLPDependencyParser(url='http://localhost:9000')
    parse, = dependencyParser.raw_parse(sentence)
    
    # Dependency parsing to parse tree based patterns as features
    depParseResult = list(parse.triples())
    
    return depParseResult
    
    
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
        print("Started Processing File : " + articleNames[i])
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
            print("\n------SENTENCE------")
            print(sen)

            word_tokens = Tokenization(sen)
            #print("\nWord Tokenization : Done")
            #print(word_tokens)

            word_lemmas = Lemmatization(word_tokens)
            #print("Word Lemmatization : Done")
            #print(word_lemmas)

            word_POStags = POSTagging(word_tokens)
            #print("POS Tagging : Done")
            #print(word_POStags)

            hypernyms, hyponyms, meronyms, holonyms, synonyms = WordNetFeatures(sen, word_tokens)
            #print("WordNet Feature Extraction : Done")
            #print(holonyms)
            
            depParse = DependencyParsing(sen)
            #print("Dependency Parsing : Done")
            #print(depParse)

            headList = getHeads(sen, word_tokens)
            #print("Obtaining Heads : Done")
            #print(headList)

            # Process data format to suit the Elastic Search requirements
            count = count + 1
            corpus_dict[count] = {}
            
            corpus_dict[count]["sentence"] = {}
            corpus_dict[count]["sentence"] = sen
            
            corpus_dict[count]["tokenized_text"] = {}
            corpus_dict[count]["tokenized_text"] = word_tokens
            
            corpus_dict[count]["lemma"] = {}
            corpus_dict[count]["lemma"] = word_lemmas
            
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
            
            corpus_dict[count]["file_name"] = {}
            corpus_dict[count]["file_name"] = articleNames[i]

        output_name = '../Pipeline-Output/Parsed-' + articleNames[i]
        with open(output_name, 'w+', encoding='utf8') as output_file:
            json.dump(corpus_dict, output_file,  indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        
        print("Completed Processing File : " + articleNames[i])
        
    print("Task 1 Successfully Completed !!!")


# In[ ]:




