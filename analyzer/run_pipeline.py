import json
import time
import pandas as pd
from util import preprocess, space_tokenize, bert_tokenize, load_data_from_db, load_wikilingua_data
from analyzer import Analyzer

start_time = time.time()
print(f'Start time: {time.ctime()}')

print()
print('load data')
# parse config file
with open('../config.json', 'r') as f:
    config = json.load(f)

# access config data
db_path = config['db_path']
country = config['country'].upper()
print_mode = config['print_mode']
save_results = config['save_results']
num_samples = config['num_samples']

# run with comparison data if wiki lingua
is_comparison_data = False
if db_path == 'wiki_lingua':
    print(f'load wiki_lingua data')
    is_comparison_data = True
    df = load_wikilingua_data(country, num_samples=num_samples)
else:
    print(f'connect to database {db_path}')
    # get data from sqlite db
    df = load_data_from_db(db_path, num_samples=num_samples, country_code=country, is_html=True)

# load spacy model
with open('../spacy-mapping.json', 'r') as f:
    mapping = json.load(f)

# access config data
if country in mapping:
    spacy_model = mapping[country]

print()
print(f'COUNTRY: {country}')
# df = preprocess(input)
if print_mode:
    print(df.head())
    print(df.shape) 

print()
print('start pipeline')
# create Analyzer object
pipeline = Analyzer(df, country, print_mode, save_results, comparison_data=is_comparison_data)

# get document length using different tokenization methods
pipeline.text_length(tokenization=space_tokenize)
pipeline.text_length(tokenization=bert_tokenize)

# get standardized type token ratio
pipeline.standardized_TTR()

# get vendi-score
pipeline.ngram_vendi_score()
pipeline.embedding_vendi_score()

# get tokens, tags and lemmas from spacy
pipeline.spacy_token_tag_lemma(spacy_model)
pipeline.standardized_TTR(tokenization='lemma')

# POS frequency
pipeline.pos_frequency()

# POS n-gram frequency
pipeline.calculate_pos_ngram_frequency()

# analysis that should be executed on job ads only
if not is_comparison_data:
    # number of lists per doc and average list length
    pipeline.paragraph_list_count_length(tokenization=space_tokenize)
    # salary info
    pipeline.nli_salary_lines(per_paragraph=True)
    # language detection per paragraph
    pipeline.language_detection()

print()
end_time = time.time()
total_seconds = end_time-start_time
print(f'pipeline completed successfully in {int(total_seconds/60)}.{int(total_seconds%60)} minutes')
print(f'End time: {time.ctime()}')