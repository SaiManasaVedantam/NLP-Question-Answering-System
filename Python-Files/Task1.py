# Turn off unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Import all the required packages
import nltk
import codecs
import urllib
import os, spacy
import en_core_web_sm
from spacy import displacy
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from IPython.core.display import display, HTML
from nltk.tokenize import word_tokenize, sent_tokenize 
nltk.download('punkt')
nltk.download('wordnet')
nlp = en_core_web_sm.load()


# Helper method to append values to duplicate keys in the dictionaries without data loss
def AppendData(mainDictionary, smallDictionary):
    smallKeys = smallDictionary.keys()
    mainKeys = mainDictionary.keys()
    for key in smallKeys:
        if key in mainKeys:
            mainDictionary[key].append(smallDictionary[key])
        else:
            mainDictionary[key] = smallDictionary[key]
    return mainDictionary
            

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


# Obtains WordNet Features
def WordNetFeatures(word_tokens):
    # Creates dictionaries for important word senses
    hypernyms_dict = dict()
    hyponyms_dict = dict()
    meronyms_dict = dict()
    holonyms_dict = dict()
    
    # Populates the above dictionaries according to the word senses associated with them
    for token in word_tokens:
        # For each token, we create & populate the considered word senses & add that to their dictionaries
        token_hypernyms = []
        token_hyponyms = []
        token_meronyms = []
        token_holonyms = []
        
        # Obtain Synsets associated with the token
        synsetList = wn.synsets(token)
        synsetLen = len(synsetList)
        
        if synsetLen != 0:
            for each_syn in synsetList:
                # Appends Hypernym Lemmas to the Hypernym list
                for hypernym in each_syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        token_hypernyms.append(lemma.name())
                
                # Appends Hyponym Lemmas to the Hypernym list
                for hyponym in each_syn.hyponyms():
                    for lemma in hyponym.lemmas():
                        token_hyponyms.append(lemma.name())
                
                # Appends Meronym Lemmas to the Hypernym list
                for meronym in each_syn.part_meronyms():
                    for lemma in meronym.lemmas():
                        token_meronyms.append(lemma.name())
                        
                # Appends Holonym Lemmas to the Hypernym list
                for holonym in each_syn.part_holonyms():
                    for lemma in holonym.lemmas():
                        token_holonyms.append(lemma.name())
                        
        hypernyms_dict[token] = token_hypernyms
        hyponyms_dict[token] = token_hyponyms
        meronyms_dict[token] = token_meronyms
        holonyms_dict[token] = token_holonyms
        
    return hypernyms_dict, hyponyms_dict, meronyms_dict, holonyms_dict
   
    
# Performs Dependency Parsing
def DependencyParsing(sentence):
    depedencyParse = nlp(sentence)
    for token in depedencyParse:
        print(token.text,"----->",token.dep_,"----->",token.pos_,)
    print('\n\n')
    display(depedencyParse)
    html = displacy.render(depedencyParse, style="dep")
    display(HTML(html))
    

# Main method
if __name__ == "__main__":
    # List of all article names in the repository
    articleNames = ["109.txt", "111.txt", "151.txt", "160.txt", "177.txt", 
                    "179.txt","181.txt", "196.txt", "199.txt", "220.txt", 
                    "222.txt", "226.txt", "288.txt", "297.txt", "304.txt", 
                    "342.txt", "347.txt", "360.txt", "390.txt", "400.txt", 
                    "56.txt", "58.txt", "6.txt"]
    fileCount = len(articleNames)
    
    content = ""
    folderPath = "https://raw.githubusercontent.com/SaiManasaVedantam/NLP-QA-System-Datasets/main/Articles/"
    for i in range(fileCount):
        fileName = folderPath + articleNames[i]
        response = urllib.request.urlopen(fileName)
        webContents = response.read()
        stringTypeData = webContents.decode("utf-8")
        content += stringTypeData
   
    # Use this if you want to use a local file on your machine
    """
    content = None
    try:
        f = open("Articles/6.txt", "r")
        content = f.read()
        
    except UnicodeDecodeError:
        f = open("Articles/6.txt", "r", encoding = "utf8")
        content = f.read()
        
    """
    
    # Obtain wordnet lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Get tokenized content
    sentences = []
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences.extend(tokenizer.tokenize(content))
    
    # Sentence count
    print("Total Sentences After splitting the document: ", len(sentences))
    print("Extracting features for each of the sentences and shown below:")
    
    # We maintain all tokens, lemmas etc. in the following lists
    all_word_tokens = []
    all_word_lemmas = []
    all_word_POStags = []
    all_hypernyms = dict()
    all_hyponymns = dict()
    all_meronyms = dict()
    all_holonyms = dict()
    
    
    # Extracting words
    for sen in sentences:
        print("\n------SENTENCE------")
        print(sen)
        
        print("\n----Word Tokenization----")
        word_tokens = Tokenization(sen)
        all_word_tokens += word_tokens
        #print(word_tokens)
        
        print("\n----Word Lemmatization----")
        word_lemmas = Lemmatization(word_tokens)
        all_word_lemmas += word_lemmas
        #print(word_lemmas)
        
        print("\n----POS Tagging----")
        word_POStags = POSTagging(word_tokens)
        all_word_POStags += word_POStags
        #print(word_POStags)
        
        print("\n----WordNet Feature Extraction----")
        hypernyms, hyponyms, meronyms, holonyms = WordNetFeatures(word_tokens)
        all_hypernyms = AppendData(all_hypernyms, hypernyms)
        #all_hypernyms.update(hypernyms)
        #print("===> HYPERNYMS: <===\n", hypernyms, "\n")
        
        all_hyponymns = AppendData(all_hyponymns, hyponyms)
        #print("===> HYPONYMS: <===\n", hyponyms, "\n")
        
        all_meronyms = AppendData(all_meronyms, meronyms)
        #print("===> MERONYMS: <===\n", meronyms, "\n")
        
        all_holonyms = AppendData(all_holonyms, holonyms)
        #print("===> HOLONYMS: <===\n", holonyms)
        
        print("\n----Dependency Parsing----")
        #DependencyParsing(sen)