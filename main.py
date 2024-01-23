# NER project
# by Christian Grech

# Import all libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
import json
import requests
from configobj import ConfigObj
import string
nltk.download('stopwords')
conf = ConfigObj('.env')
API_URL = "https://api-inference.huggingface.co/models/jayant-yadav/roberta-base-multinerd"
headers = {"Authorization": "Bearer "+ conf['HUGGINGFACEHUB_API_TOKEN']}
evaluation = 1 # Evaluate the model (1) or just generate the news-article-linked file
execution = 1

# Function to remove stopwords
def clean_text(df):
    stopword_pattern = {'|'.join([r'\b{}\b'.strip().format(w) for w in stop_words]): ''}
    return (df.assign(text_cleaned=lambda df_: 
                  df_.text.replace(stopword_pattern, regex=True))) 

# Query function to reach API and get predictions
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        keyValList = ['ORG'] # Extract detected organizations
        result = [d for d in response.json() if d['entity_group'] in keyValList and d['score'] > 0.2]
        return result
    except:
        print('Unable to get data.')
        pass
        
if __name__ == "__main__":
    
    
    lines = []
    new = []
    annotations = []
    global_correct_predictions = 0
    total_annotations = 0
    new_companies = []

    # Import the news articles with annotations
    with open(r'news_articles-gold.jsonl') as f:
        lines = f.read().splitlines()

    line_dicts = [json.loads(line) for line in lines]
    df_text = pd.DataFrame(line_dicts) # Place data in a pandas DataFrame

    # Import the testing dataset
    with open(r'news_articles-new.jsonl') as f:
        new = f.read().splitlines()

    line_dicts = [json.loads(line) for line in new]
    df_test = pd.DataFrame(line_dicts)  # Place data in a pandas DataFrame

    # Import the companies dataset
    df_companies = pd.read_json('company_collection.json')
    
    # Preprocess the data
    stop_words = stopwords.words('english')
    df_text = clean_text(df_text)
    

    # Clean company names to allow better matching with predictions and annotations
    df_companies['name'] = df_companies['name'].apply(lambda x: x.strip()) # Strip whitespaces
    df_companies['clean_name'] = [''.join(c for c in s if c not in string.punctuation) for s in df_companies['name'].tolist()] # Remove punctuation
    df_companies['clean_name'] = df_companies['clean_name'].apply(lambda x: x.lower()) # Change to lower case
 
    # Model evaluation
    if evaluation == 1:
        # Iterate all articles
        for idx, row in df_text.iterrows():
            company_dict={}
            output = query({"inputs": row['text']}) # Model output
            predicted_companies = set(d['word'].strip() for d in output) # Strip whitespace
            #predicted_companies = set(i for i in pred_companies if not ('runchbase' in i or 'elegraph' in i)) # Remove Crunchbase and Telegraph entries
            predicted_companies_lower = set(d.lower() for d in predicted_companies) # Switch names to lowercase
            predicted_companies_no_punctuation = [''.join(c for c in s if c not in string.punctuation) for s in predicted_companies_lower] # Remove punctuation
            filtered = df_companies[df_companies['clean_name'].isin(predicted_companies_no_punctuation)].reset_index() # Merge predictions with the company collection to get urls
            clean_company_dict = pd.Series(filtered.url.tolist(),index=filtered.clean_name).to_dict() # Create a dictionary with the lower case company names
            company_dict = pd.Series(filtered.url.tolist(),index=filtered.name).to_dict() # Create a dictionary with the normal case company names
            

            correct_predictions = 0
            for key in row['annotations']:
                clean_key = key.strip().lower() # Lower case
                clean_key = ''.join(c for c in clean_key if c not in string.punctuation) # Remove punctuation
                if clean_key in clean_company_dict.keys(): # Compare clean predictions with clean annotations
                    correct_predictions = correct_predictions + 1 # Count correct predictions in one article
                    global_correct_predictions = global_correct_predictions + 1 # Count correct predictions in all articles

            score = correct_predictions / len(row['annotations']) # Calculate Accuracy
            total_annotations = total_annotations + len(row['annotations']) # global number of annotations
            print('Accuracy: ', str(round((score*100),1)), '%')
            
            for company in predicted_companies: # Iterate predictions to find companies which were not in the collections file
                clean_company_name = company.strip().lower() # Lower case
                clean_company = ''.join(c for c in clean_company_name if c not in string.punctuation) # Remove punctuation
                if clean_company not in filtered['clean_name'].tolist():
                    company_dict[company] = ''
                    new_companies.append(company) 
            annotations.append(company_dict) # All predictions
            
        print('Total Accuracy: ', str(round((global_correct_predictions*100/total_annotations),2)), '%')
        print('New companies found: ', new_companies)
    
    
    # Model execution
    if execution == 1:
        print('Executing model on the news_articles-new.jsonl file.......')
        predictions = []
        annotations = []
        total_annotations = 0
        new_companies = []
        
        # Execute the model on the test dataset for production
        for idx, row in df_test.iterrows():
            company_dict={}
            output = query({"inputs": row['text']})
            predicted_companies = set(d['word'].strip() for d in output) # Strip whitespace
            #predicted_companies = set(i for i in pred_companies if not ('runchbase' in i or 'elegraph' in i)) # Remove Crunchbase and Telegraph entries
            predicted_companies_lower = set(d.lower() for d in predicted_companies) # Lowercase
            predicted_companies_no_punctuation = [''.join(c for c in s if c not in string.punctuation) for s in predicted_companies_lower] # Remove punctation
            filtered = df_companies[df_companies['clean_name'].isin(predicted_companies_no_punctuation)].reset_index() # Merge predictions with the company collection
            clean_company_dict = pd.Series(filtered.url.tolist(),index=filtered.clean_name).to_dict() # Create a dict with the clean names
            company_dict = pd.Series(filtered.url.tolist(),index=filtered.name).to_dict() # Create a dict with the original company names for the output

            for company in predicted_companies:
                clean_company_name = company.strip().lower() # Strip whitespace and lower case
                clean_company = ''.join(c for c in clean_company_name if c not in string.punctuation) # Remove punctation
                if clean_company not in filtered['clean_name'].tolist():    # If company is not in the collection, add to a new list and put an empty string as a URL
                    company_dict[company] = ''
                    new_companies.append(company)
            annotations.append(company_dict)
        df_test['annotations'] = annotations
        
        # Output in JSONL format
        output_path = "news_articles-linked.jsonl"
        try:
            with open(output_path, "w") as f:
                f.write(df_test.to_json(orient='records', lines=True, force_ascii=False))
            print('Successfully saved data in', output_path)
        except:
            print('Error writing to file')
