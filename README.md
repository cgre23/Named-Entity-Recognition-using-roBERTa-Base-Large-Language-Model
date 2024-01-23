# Named Entity Recognition using roBERTa base Large Language Model
### by Christian Grech


## Introduction

The aim is to process news articles from the *news_articles-new.jsonl* dataset and identify and link potential companies that appear in these articles. The results are be a file called *news_articles-linked.jsonl* in the same format as the *news_articles-gold* dataset. The company identifiers are taken from the company_collection but new companies that are not part of the collection are added with an empty string as value, e.g. companies: {'Apple':'apple.com', 'MalWart': ''} . The solution is implemented in Python and uses the roBERTa-Base-Multinerd LLM. This model can be used for Named Entity Recognition and is trained on the MultiNERD dataset. More information on the model in HuggingFace can be found in https://huggingface.co/jayant-yadav/roberta-base-multinerd

## Performance metric

The performance metric used to evaluate the model is accuracy, defined as the correct predictions divided by the total number of annotations in the *news_articles-gold* dataset. This metric is used because alternative companies predicted, that are not in the company collection file might still be appropriate predictions.

## Solution

1. Data is loaded from the jsonl files.
2. If evaluation is set to 1 in the top of the script, the model is evaluated with data from the *news_articles-gold.jsonl* file. The accuracy for each article and the global accuracy is printed in the script.
3. If execution is set to 1, the model executes based on the *new_articles-new.jsonl* articles and the annotations are saved in the *new_articles-linked.jsonl* file


## Instructions

1. Make sure that you place the HUGGINGFACEHUB_API_KEY in the .env file in this folder. Use this format: HUGGINGFACEHUB_API_KEY=xxx. Access the file as follows:
        
        nano .env


2. Activate the virtual environment. If the folder is not available, a new virtual environment can be created using the available *requirements.txt* file

        . venv/bin/activate


3. Run the python script


        python main.py


3. Run the python script


        python main.py
    

4. Wait until the script finishes for a new *news_articles-linked.jsonl* file to be produced.


## Evaluation

Evaluating the model, it results in a 64% accuracy based on the provided annotations. However over 70 new company names were extracted which can be explored as potential good predictions.

## New company suggestions

Following evaluation on the gold dataset, these were the companies proposed by the model which are not listed in company collections:

    New companies found:  {'Yahoo Finance', 'Crunchbase News', 'Health Service', 'AI', 'New York State Energy Research and Development Authority', 'Big', 'The University of Texas at Austin', 'Work', 'Insight Partners', 'Postman API', 'Dark', 'TenEleven Ventures', 'TCL Capital', 'Takasago Thermal Engineering Co.', 'Employment Development Department', 'University of Pennsylvania', 'Abingworth', 'Preston-Werner Ventures 1', 'IF SPV 1st Investment Partnership', 'Harvard Law School', 'Tiger Global', 'OrbiMed Advisors', 'Children’s Hospital Colorado Center for Innovation', 'Regional Office of Africa', 'LUN Partners Group', 'Bill Wood Ventures', 'Companion Fund', 'GIBLIB', 'Crunchbase Daily Progyny', 'Telegraph', 'KKR', 'U.S. Immigration and Customs Enforcement', 'Hippo Insurance', 'World Health Organization', 'Left Lane', 'Centaurus Advisors', 'Nasdaq', 'Austin Ventures', 'Firebolt Ventures', 'The Wall Street Journal', 'Breakthrough Energy Ventures', 'Left Lane Capital', 'Austin Technology Incubator', 'Jeremy and Hannelore Grantham Environmental Trust', 'Space Frontier Fund', 'Life Sciences', 'CNBC', 'o', 'Bio', 'Steady Together Initiative', 'Bio Fund', 'Crunchbase Daily Fund III', 'Thinx Miki Agrawa l', 'OECD', 'The Telegraph', 'XL Ventures', 'Crunchbase Daily', 'University of Colorado', 'Bessemer Venture Partners', 'Andreessen Horowitz', 'Bling Capital', 'Crunchbase Daily Investors', 'Medicare Advantage', 'Forge Global', 'Financial Times', 'Sanofi Ventures', 'Post', 'Money', 'NY', 'The One Health Company', 'Nasdaq Private Market', 'Krispy Kreme Doughnut Corp', 'Recode', 'NFX', 'Pel', 'Wh', 'Founder Collective', 'The Farmer’s Dog', 'Tonal', 'Freight', 'Atlas Venture'}

## Why this method?

Given the time limitations of the task, using a pre-trained model on a related dataset makes sense. Challenges for this task included raw/unclean data (for example a company like WhizAI would be written as Whiz.AI and would be harder to match). This is why company names were stripped from whitespaces, converted to lowercase and punctuation marks extracted.

## Improvements

- Better handling of API errors in case of problems getting this data. If this happens, try running the script multiple times.
- Investigate which companies where not predicted from the gold file, and why this happens.
- Search for a better trained model, based on a larger dataset.
- Potentially fine-tune the model once the collected companies list is large enough. This requires resources such as GPUs/TPUs to be viable in a short amount of time.
