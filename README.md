# NLP-Question-Answering-System (October 2021)
This project implements a Closed-Domain Question-Answering System using Natural Language Processing techniques. We all know that the use of interactive, intelligent assistants which can respond to human statements/questions is increasing day-by-day. Using NLP, this project aims to answer WHAT, WHEN and WHO
type questions based on 30 articles from the Stanford Question Answering Dataset(SQuAD). 


## Approach
The solution used in the project reads articles one-by-one, tokenize into sentences and then to words, extract features required and stores them. Then, using the features obtained by passing through the NLP pipeline, inverted indexing is done on each sentence using the Elastic Search. Later, a search on the indexed document using a much deeper pipeline with varied weights associated with each feature helps us to obtain top 10 ranked results from which the top solution is considered as the final answer to the question posed. However, the above is implemented in 3 stages.


## Stages
### Stage 1
This phase involves feature extraction from the NLP pipeline. Features considered are:
1. Word and Sentence Tokens : They help us to understand the context & develop NLP model
2. Lemmas : They help us to group different forms of word so that any version of base word can be used in the query and retrieval.
3. Part-of-Speech Tags (POS) : They help us to build Parse trees and can be used as a linguistic criteria.
4. WordNet Features : We extracted the following features from WordNet.
	a. Synonyms : Helps in handling exact substitutes for a word with the same meaning.
	b. Hypernyms : Helps in extracting the included meaning of the words.
	c. Hyponyms : Helps us to find the subcategory of a generic word.
	d. Meronyms : Helps us to identify the constituent part of something.
	e. Holonyms : Helps us to know the whole thing or group to which something belongs.
5. Dependency Parsing : This step helps us to obtain the relationship between different words of a sentence.
6. Head Generation : This helps us to understand the essence of the sentence.
7. Stems : They help us to reduce the inflectional forms of words by considering the roots.
8. Named Entities : They help us to identify the key elements in the sentence like people, place, time, year etc.

From the above, after making critical analysis on the performance on validation set, few features & weights(importance) associated contribute to "Useful features". This can be done on trial & error basis for more understanding.

### Stage 2
This phase involves indexing the output obtained from the NLP pipeline using ElasticSearch, using that to build better query criteria to obtain the most sensible output for the question posed to the system. 

### Stage 3
This stage involves generating answers for a list of questions with different levels of complexity passed as an input to the pipeline and generating the most sensible answer obtained from the model. 


## Tools & Technologies
### Programming Language & Libraries
1. Python
2. NLTK (sentence tokenization, word tokenization, lemmatization, stemming, POStagging, WordNet features)
3. Stanford’s CoreNLP Parser (dependency parsing, Named entity tagging)
4. Elasticsearch module for fetching requests from the server
5. Python’s ‘csv’ library for output formatting
6. Python’s WordNet library
7. Lesk for obtaining the best sense based on the context
8. PorterStemmer for stemming
9. WordNetLemmatizer for extracting lemmas from the sentence
10. Python’s dateparser for parsing times in the when case
11. Python’s ast to process the validation dataset and obtain questions from it

### Tools & IDEs
1. Jupyter Notebook
2. ElasticSearch
3. Github


## Architecture
<img width="375" alt="Architecture" src="https://user-images.githubusercontent.com/28973352/149229899-20c007c0-d740-47b8-bdf0-cb41789bf1b0.png">


## Execution Instructions
Please check "Instructions.txt" for all detailed instructions on How to execute the code, Order of execution & the packages to be installed.
