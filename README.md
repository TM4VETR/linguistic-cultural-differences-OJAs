# stea-language-comparison-pipeline
This repository contains a pipeline for comparing online job advertisements obtained from [CareerJet](https://www.careerjet.com/) from seven different countries (Germany, Austria, United Kingdom, United States, Spain, France, Italy) and five languages (German, English, Spanish, French, Italian).

## Description
The following aspects are analyzed in order to compare job ads from different countries:
* document length using BERT and whitespace tokenization
* standardized type-token ratio as a measure for lexical diversity
  * with tokens (space-tokenized) and spacy lemmas
* n-gram [vendi score](https://github.com/vertaix/Vendi-Score) for structural/ lexical diversity
* embedding vendi score for semantic diversity
* POS-tag frequency
* frequency of n-grams of POS-tags
* paragraph count and length (in tokens)
* list (HTML unordered list) count and length (in items)
* salary information (Does the job ad contain info about salary, and in which lines?) using an NLI model
* language detection: How many documents are written in other languages than expected, and which languages?
\\
All analysis except the latter four are executed with the [huggingface Wikilingua dataset](https://huggingface.co/datasets/wiki_lingua) in the examined languages (German, English, Spanish, French, Italian) for comparison.

## Content
* `plots/`: plots generated from the pipeline results, each country has a subfolgder, e.g. `plots/US/`
* `results/`: analysis results will be saved to a country-specific subfolder, e.g. `results/US/`
* `analyzer/`: contains the python scripts used for analysis, including the main script `run_pipeline.py`
* `crawler/`: contains the python scripts used to crawl the CareerJet data, including the main script `careerjet_crawler.py`
* `config.json`: config file that should be modified as needed before running pipeline
* `install-spacy.sh`: bash script to install required spacy models for all of the languages
* `requirements.txt`: lists required python packages
* `spacy-mapping.json`: maps each country to a spacy model of the respective language

## Getting Started

### Dependencies
* The code is written in Python 3.8
* Install required packages using `pip install -r requirements.txt`
* Additionally, download the spacy models by executing the shell script `bash install-spacy.sh`
* Input should be provided as an SQLite database containing at least the columns `id`, `country` and `description`

### Execution

#### Crawler
To execute the crawler script, navigate to `crawler/` and run using `python3 careerjet_crawler.py`.

#### Analyzer
First modify the parameters in `config.json` as needed:
```
 {
  "db_path": "merged_final.db",
  "country": "us",
  "save_results": false,
  "plot_results": false,
  "print_mode": true,
  "num_samples": 10000
}
```
* `db_path`: path to SQLite database containing table `job_listings`, input "wiki_lingua" to run with comparison data
* `country`: origin country of data to be analyzed, choose from: `["aut", "de", "es", "fr", "it", "uk", "us"]`
* `save_results`: if true, results will be saved to the respective country folder in `results/`
* `plot_results`: if true, some results will be plotted and saved to the respective country folder in `plots/`
* `print_mode`: if true, results will be printed to console while the pipeline is running
* `num_samples`: Specify the number of samples to be used for analysis, e.g. `num_samples=10` will run the pipeline using only the first ten samples available for the given country. Choose zero or a negative number if you want to run the pipeline with the full available data.\
\
To execute the pipeline, navigate to `analyzer/` and run the script using `python3 run_pipeline.py`.

## Add a new country
To add data from another country, check if the language is already used in this repository (check language models in `install-spacy.sh`). If yes, add another line to `spacy-mapping.json` with the according country code and model name (e.g. `"US": "en_core_web_sm"`). \
If not, make sure the language is supported by both [spacy](https://spacy.io/usage/models) and [this NLI model](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) to make sure that the pipeline is able to run smoothly. Add a new line to `install-spacy.sh` to install the required spacy model for the language. Add the country code and the suitable spacy model to `spacy-mapping.json`. \
After completing this steps, modify `config.json` with the according country code and the pipeline should run without errors.

## Structure
```
.
├── plots
│   ├── AUT
│   ├── DE
│   ├── ES
│   ├── FR
│   ├── IT
│   ├── paired
│   ├── statistics
│   ├── UK
│   └── US
│   └── wiki_lingua
├── results
│   ├── AUT
│   ├── DE
│   ├── ES
│   ├── FR
│   ├── IT
│   ├── statistics
│   ├── UK
│   └── US
│   └── wiki_lingua
└── analyzer
│   └── analyzer.py
│   └── examine_results.py
│   └── run_pipeline.py
│   └── util.py
└── crawler
│   └── careerjet_crawler.py
│   └── country_config.py
│   └── data_helper.py
├── config.json
├── install-spacy.sh
├── README.md
├── requirements.txt
├── spacy-mapping.json
```
