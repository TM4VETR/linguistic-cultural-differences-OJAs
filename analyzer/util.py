import pandas as pd
import os
import re
import torch
import sqlite3
import nltk
from nltk.util import ngrams
nltk.download('punkt')
from datasets import load_dataset
from bs4 import BeautifulSoup
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# connect to SQLite database and return dataframe suitable for analysis
def load_data_from_db(db_path, num_samples=10000, table_name='job_listings', is_html=False, country_code=None):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        if country_code == None:
            query = f'''SELECT id, description, country FROM {table_name};'''
        else:
            query = f'''SELECT id, description FROM {table_name} WHERE country='{country_code}';'''
        df = pd.read_sql_query(query, conn)
    # sample num_samples items
    if 0 < num_samples < df.shape[0]:
        df = df.head(num_samples)
    texts, paragraphs = [], []
    if is_html: # parse html if description+ still contains html tags
        for html_string in df['description']:
            raw_text, ps = parse_html(html_string)
            texts.append(raw_text)
            paragraphs.append(ps)
        data = pd.DataFrame({'id': df['id'],
                             'text': texts,
                             'paragraphs': paragraphs})
    else: # if text description is raw text
        data = df.rename(columns={'description': 'text'})
    return data

# merge data to one database containing equal amount of samples for each country
def merge_db(dir, output_path, table_name='job_listings', samples_per_country=10000):
    dfs = []
    for db_file in os.listdir(dir):
        print(f'PROCESS DB {db_file}')
        db_path = dir+'/'+db_file
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            country_query = f'''SELECT DISTINCT country FROM {table_name};'''
            cursor.execute(country_query)
            res = cursor.fetchall()
            countries = [el[0] for el in res]
            if db_file == 'jobads_final_3.db':
                countries.remove('ES')
            print(f'COUNTRIES will be added {countries}')
            for c in countries:
                print(c)
                query = f'''SELECT * FROM {table_name} WHERE country='{c}';'''
                country_data = pd.read_sql_query(query, conn)
                if 0 < samples_per_country <= country_data.shape[0]:
                    country_data = country_data.sample(samples_per_country)
                dfs.append(country_data)
                print(f'{c} data shape {country_data.shape}')
                print(f'{c} data columns {country_data.columns}')
    full_data = pd.concat(dfs)
    print(f'FULL DATA SHAPE {full_data.shape}')
    print(f'FULL DATA COLUMNS {full_data.columns}')
    with sqlite3.connect(output_path) as out_conn:
        cursor = out_conn.cursor()
        cursor.execute(f'''DROP TABLE IF EXISTS {table_name}''')
        print(full_data.shape)
        full_data.to_sql(table_name, out_conn, if_exists='replace', index=False)
        out_conn.commit()

# get wikilingua data from huggingface datasets
def load_wikilingua_data(country_code, num_samples=10000):
    country_mapping = {'AUT': 'german', 'DE': 'german', 'ES': 'spanish', 'FR': 'french', 'IT': 'italian', 'UK': 'english', 'US': 'english', 'EN': 'english'}
    dataset = load_dataset('wiki_lingua', split='train', name=country_mapping[country_code])
    ids, urls, texts, english_urls = [], [], [], []
    for i, item in enumerate(dataset):
        if i > 0 and i == num_samples:
            break
        doc = item['article']['document']
        if len(doc) < 1: # if document has no text content
            num_samples += 1
            continue
        else:
            texts.append(doc[0])
        ids.append(i+1)
        urls.append(item['url'])
        if 'english_url' in item['article']:
            english_urls.append(item['article']['english_url'])
        else:
            english_urls.append('na')
    df = pd.DataFrame({'id': ids, 'text': texts, 'url': urls, 'english_url': english_urls})
    return df

# parse html and return paragraphs
# paragraphs are separated by <br>, a list is counted as one paragraph
def parse_html(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    tag = soup.find('p', {'class' : 'source'})
    if tag is not None:
        tag.decompose()
    soup.extract(tag)
    raw_text = ''
    paragraphs = []
    section = soup.find('section', {'class' : 'content'})
    if section is not None:
        for tag in section.children:
            if tag.name != 'ul' and tag.name != 'ol': 
                str_content = tag.get_text()
                str_content = re.sub(r'http\S+', '', str_content) # remove urls
                str_content = re.sub(r'\".+\":\s{.+}', '', str_content) # remove unwanted dictionary elements from text
                str_content = str_content.strip().strip('\n')
                if len(str_content) > 1:
                    raw_text += '\n' + str_content
                    paragraphs.append([str_content])
            else:
                list_content = []
                lis = tag.children
                for li in lis:
                    content = li.get_text()
                    content = content.strip().strip('\n')
                    if len(content) > 1:
                        raw_text += '\n' + content
                        list_content.append(content)
                paragraphs.append(list_content)
    return raw_text.strip('\n'), paragraphs

# read input file and preprocess text, 
def preprocess(df, remove_lb=False, lowercase=False, remove_punct=False, num_samples=0):
    df['id'] = df['id'].astype(int)
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str.replace(r'\r', '\n')
    df['text'] = df['text'].str.replace(r'\n+', '\n', regex=True)
    if remove_lb: # remove linebreaks
        df['text'] = df['text'].str.replace('\n', '')
    if lowercase: # lowercase
        df['text'] = df['text'].apply(lambda x: x.lower())
    if remove_punct: # remove punctuation
        df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    if 0 < num_samples <= df.shape[0]:
        df = df.head(n=num_samples)
    return df


# simple tokenization
def space_tokenize(string_list):
    tokenized = []
    for s in string_list:
        t = re.findall(r'[!?.,;:\-%/’$€@]|\w+', s)
        tokenized.append(t)
    return tokenized


# BERT tokenization
def bert_tokenize(string_list, model_name='bert-base-multilingual-cased', print_interval=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'tokenization using device {device}')
    tokenizer = BertTokenizer.from_pretrained(model_name, device=device)
    tokenized = []
    cnt = 0 
    for s in string_list:
        cnt += 1
        t = tokenizer.tokenize(s)
        tokenized.append(t)
    return tokenized


# get ngrams
def get_ngrams(str_list, n, lowercase=False):
    n_grams = []
    for item in str_list:
        if lowercase:
            item = [x.lower() for x in item]
        ng_doc = list(ngrams(item, n))
        n_grams.append(ng_doc)
    return n_grams

# load model and tokenizer for NLI
def load_nli_model(print_mode, modelname='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'):
    # setup NLI model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = modelname
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    if print_mode:
        print()
        print(f'NLI model {modelname} set up on device {device}')
    return model, tokenizer

# perform nli linewise
def nli_binary(premise, hypothesis, model, tokenizer, print_mode, threshold=90.0):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input['input_ids'].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ['entailment', 'neutral', 'contradiction']
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    if prediction['entailment'] >= threshold: # prediction['neutral'] and prediction['entailment'] > prediction['contradiction']:
        if print_mode:
            print(premise)
        return premise
    else:
        return None

# detect language of a text using pretrained huggingface model  
def detect_language(text, modelname='papluca/xlm-roberta-base-language-detection'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = pipeline('text-classification', model=modelname, device=device)
    prediction = pipe(text, top_k=3, truncation=True)
    return prediction
