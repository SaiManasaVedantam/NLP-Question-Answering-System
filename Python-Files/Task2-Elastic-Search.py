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
        print("Started Processing File : " + articleNames[i])
        fileName = folderPath + articleNames[i]
        response = urllib.request.urlopen(fileName)
        webContents = response.read()
        stringTypeData = webContents.decode("utf-8")
        content = stringTypeData
        
        # Convert to Json
        jsonFile = json.loads(content)
        
        # Creating new index QA-Articles
        for key, value in jsonFile.items():
            elastic.index(index = "articles", doc_type = "text", id = idx, body = value)
            # print("Here")
            idx += 1
            
    print("Elastic Search Successfully Completed !!!")
