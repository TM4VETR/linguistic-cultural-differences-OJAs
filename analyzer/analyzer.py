import numpy as np
import pandas as pd
import spacy
import torch
import re
import os
import os.path
from vendi_score import text_utils
from collections import Counter
from util import get_ngrams, space_tokenize, load_nli_model, nli_binary, detect_language


class Analyzer:

    def __init__(self, data, country, print_mode, save_results, comparison_data=False, max_n=3):
        self.ids = data['id']
        self.texts = data['text']
        if 'paragraphs' in data.columns:
            self.paragraphs = data['paragraphs']
        self.country = country
        self.print_mode = print_mode
        self.save_results = save_results
        self.max_n = max_n
        self.tokens = None
        self.tags = None
        self.lemmas = None
        # create directories for country
        if not comparison_data:
            self.result_path = '../results/' + country + '/'
        else: 
            self.result_path = '../results/wiki_lingua/' + country + '/'
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)
            # initialize file
            with open(self.result_path + 'statistics.csv', mode='w', encoding='utf-8') as f:
                f.write('data,num_samples,statistic,value\n')

    # get document length given a tokenization function
    def text_length(self, tokenization):
        tokenized = tokenization(self.texts)
        token_cnt = np.array([len(t) for t in tokenized])
        mean = round(np.mean(token_cnt), 4)
        median = round(np.median(token_cnt), 4)
        std = round(np.std(token_cnt), 4)
        str_description = 'doclen-' + tokenization.__name__ + 'd'
        if self.print_mode:
            print()
            print(f'{tokenization.__name__}d document length mean: {mean}, standard deviation: {std}, median: {median}')
        if self.save_results:
            df = pd.DataFrame({'id': self.ids, 'token_cnt': token_cnt})
            filename = self.result_path + str_description + '.csv'
            df.to_csv(filename, index=False)
            with open(self.result_path + 'statistics.csv', mode='a', encoding='utf-8') as f:
                f.write(str_description + ',' + str(len(self.ids)) + ',mean,' + str(mean) + '\n')
                f.write(str_description + ',' + str(len(self.ids)) + ',std,' + str(std) + '\n')
                f.write(str_description + ',' + str(len(self.ids)) + ',median,' + str(median) + '\n')
        return token_cnt
    
    # get paragraph count and length given a tokenization function
    def paragraph_list_count_length(self, tokenization):
        p_counts, p_lens = [], []
        l_counts, l_lens = [], []
        for i, p_list in enumerate(self.paragraphs):
            cnt = 0
            num_p = len(p_list)
            num_toks = []
            num_lis = []
            for p in p_list:
                # if a paragraph contains multiple strings, it is a list
                if len(p) > 1:
                    cnt += 1
                    num_lis.append(len(p))
                tokens = tokenization([''.join(p)])
                num_toks.append(len(tokens[0]))
            p_counts.append(num_p)
            p_lens.append(num_toks)
            l_counts.append(cnt)
            l_lens.append(num_lis)
        stats = dict()
        measures = ['paragraphs_per_doc', 'lists_per_doc', 'paragraph_len-'+tokenization.__name__+'d', 'elements_per_list']
        lists = [p_counts, l_counts, 
                 [item for sublist in p_lens for item in sublist], 
                 [item for sublist in l_lens for item in sublist]] # flatten length lists to compute statistics
        str_description = 'paragraph-list-cnt-len'
        for i, m in enumerate(measures):
            li = lists[i]
            stats[m] = []
            stats[m].append(round(np.mean(li), 4))
            stats[m].append(round(np.std(li), 4))
            stats[m].append(round(np.median(li), 4))
        if self.print_mode:
            print(str_description)
            for key in stats.keys():
                print(f'{key}   mean: {stats[key][0]}, std: {stats[key][1]}, median {stats[key][2]}')
            print()
        if self.save_results:
            df = pd.DataFrame({'id': self.ids, 'paragraph_cnt': p_counts, 'list_cnt' : l_counts, 'paragraph_lens': p_lens, 'list_lens': l_lens})
            filename = self.result_path + str_description + '.csv'
            df.to_csv(filename, index=False)
            with open(self.result_path + 'statistics.csv', mode='a', encoding='utf-8') as f:
                for key in stats.keys():
                    f.write(key + ',' + str(len(self.ids)) + ',mean,' + str(stats[key][0]) + '\n')
                    f.write(key + ',' + str(len(self.ids)) + ',std,' + str(stats[key][1]) + '\n')
                    f.write(key + ',' + str(len(self.ids)) + ',median,' + str(stats[key][2]) + '\n')
        return p_counts, p_lens

    # type-token-ratio to compare lexical diversity
    def standardized_TTR(self, tokenization=space_tokenize):
        if tokenization == space_tokenize:
            tokenized = tokenization(self.texts)
            tokens = [item.lower() for sublist in tokenized for item in sublist]
            description = tokenization.__name__ + 'd'
        else:
            tokens = [item.lower() for sublist in self.lemmas for item in sublist]
            description = 'lemmas'
        ttr = []
        i = 0
        while i + 1000 < len(tokens):
            toks = tokens[i:i + 1000]
            types = list(set(toks))
            ttr.append(len(types) / len(toks))
            i += 1000
        toks = tokens[i:len(tokens)]
        types = list(set(toks))
        ttr.append(len(types) / len(toks))
        mean_ttr = round(np.mean(ttr), 4)
        if self.print_mode:
            print()
            print('total number of tokens: {}'.format(len(tokens)))
            print(f'standardized TTR: {mean_ttr}')
        if self.save_results:
            with open(self.result_path + 'statistics.csv', mode='a', encoding='utf-8') as f:
                f.write(description + ',' + str(len(self.ids)) + ',STTR,' + str(mean_ttr) + '\n')
        return mean_ttr

    # n-gram vendi-score
    def ngram_vendi_score(self):
        ngram_vs = text_utils.ngram_vendi_score(self.texts, ns=range(1, self.max_n))
        ngram_vs = round(ngram_vs, 4)
        if self.print_mode:
            print()
            print(f'n-gram vendi-score with n={self.max_n}: {ngram_vs}')
        if self.save_results:
            with open(self.result_path + 'statistics.csv', mode='a', encoding='utf-8') as f:
                f.write('ngrams,' + str(len(self.ids)) + ',ngram_vendiscore,' + str(ngram_vs) + '\n')
        return ngram_vs

    # embedding vendi-score
    def embedding_vendi_score(self, modelname='bert-base-multilingual-cased'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'calculate embedding vendi score on device {device}')
        emb_vs = text_utils.embedding_vendi_score(list(self.texts), model_path=modelname, device=device)
        emb_vs = round(emb_vs, 4)
        if self.print_mode:
            print()
            print(f'{modelname} embedding vendi-score : {emb_vs: .4f}')
        if self.save_results:
            with open(self.result_path + 'statistics.csv', mode='a', encoding='utf-8') as f:
                f.write('BERT_embeddings,' + str(len(self.ids)) +',embedding_vendiscore,' + str(emb_vs) + '\n')
        return emb_vs

    # use spacy model to get tokens, POS-tags, and lemmas
    def spacy_token_tag_lemma(self, spacy_model, print_interval=500):
        nlp = spacy.load(spacy_model)
        tokens = []
        tags = []
        lemmas = []
        cnt = 1
        print('start tagging and lemmatizing using spacy')
        for text in self.texts:
            doc = nlp(text)
            toks = [token.text for token in doc]
            tokens.append(toks)
            pos = [token.pos_ for token in doc]  # list of pos tags
            # print(lem)
            assert len(toks) == len(pos)
            tags.append(pos)
            lem = [token.lemma_.lower() for token in doc]
            lemmas.append(lem)
            if cnt % print_interval == 0:
                print(f'{cnt} documents tokenized, tagged and lemmatized')
            cnt += 1
        print('dataset tagging and lemmatization completed')
        self.tokens = tokens
        self.tags = tags
        self.lemmas = lemmas
        return tokens, tags, lemmas

    # get frequency of POS tags
    def pos_frequency(self):
        pos_corpus = [item for sublist in list(self.tags) for item in sublist]
        count_pos = Counter(pos_corpus)
        df = pd.DataFrame(list(count_pos.items()))
        df.columns = ['POS', 'Frequency']
        df = df.sort_values('Frequency', ascending=False).reset_index()
        df['relative_freq'] = df['Frequency'].apply(lambda x: (x / len(pos_corpus)) * 100)
        str_description = 'pos-frequencies'
        if self.print_mode:
            print()
            print(df)
        if self.save_results:
            filename = self.result_path + str_description + '.csv'
            df.drop(columns='index').to_csv(filename, index=False)
        return df

    # get POS-ngram frequencies
    def pos_ngram_frequency(self, n, top=20):
        if n == 2:
            name = 'bigram'
        elif n == 3:
            name = 'trigram'
        else:
            name = n + '-gram'
        # POS ngrams
        pos_ngrams = get_ngrams(self.tags, n)
        token_ngrams = get_ngrams(self.tokens, n, lowercase=True)
        ngrams = [item for sublist in list(pos_ngrams) for item in sublist]  
        # if 'PUNCT' not in item and 'SPACE' not in item] # get flat lists of POS tags, filter out PUNCT
        ngram_corpus = [item for sublist in list(token_ngrams) for item in sublist]  # get flat list of tokens
        count_ngrams = Counter(ngrams)
        df = pd.DataFrame(list(count_ngrams.items()), columns=['ngram', 'Frequency'])
        df = df.sort_values('Frequency', ascending=False).reset_index()
        en_total = sum(count_ngrams.values())
        df['relative_freq'] = df['Frequency'].apply(lambda x: (x / en_total) * 100)
        df = df.drop(columns=['index'])
        str_description = 'pos-' + name
        if self.print_mode:
            print()
            print(df.head(top))
        if self.save_results:
            filename = self.result_path + str_description + '-frequencies.csv'
            df.to_csv(filename, index=False)
        return df
    
    # helper function: calculate POS-ngram frequency for n = 1 to max_n
    def calculate_pos_ngram_frequency(self):
        dataframes = []
        for n in range(2, self.max_n + 1):
            dataframes.append(self.pos_ngram_frequency(n))
        return dataframes
    
    # get lines containing info about salary based on NLI
    def nli_salary_lines(self, per_paragraph=False, print_interval=500):
        model, tokenizer = load_nli_model(self.print_mode)
        hypothesis = 'The line contains information on salary.'
        salary_lines = []
        print(f'LEN PARAGRAPHS: {len(self.paragraphs)}')
        for i, doc in enumerate(self.texts):
            if per_paragraph:
                premises = [' '.join(li) for li in self.paragraphs.iloc[i]]
            else:
                premises = re.split(r'\.|\n', doc)
            lines = []
            for p in premises:
                line = nli_binary(p, hypothesis, model, tokenizer, print_mode=False)
                if line is not None:
                    lines.append(line)
            salary_lines.append(lines)
            if self.print_mode and i % print_interval == 0: 
                print(f'{i} documents checked for salary info')
        num_docs = len(salary_lines) - salary_lines.count([])
        num_lines = [len(l) for l in salary_lines]
        str_type = 'paragraphs' if per_paragraph else 'lines'
        perc = round((num_docs/ len(salary_lines) * 100), 2)
        name = f'salary_{str_type}'
        df = pd.DataFrame({'id': self.ids, name: salary_lines})
        if self.print_mode:
            print()
            print(len(salary_lines))
            print(f'number of documents with salary info {perc}%')
            print(f'average number of {str_type} per document {np.mean(num_lines)}')
        if self.save_results:
            filename = self.result_path + name + '.csv'
            df.to_csv(filename, index=False)
            with open(self.result_path + 'statistics.csv', mode='a', encoding='utf-8') as f:
                f.write(f'salary_info,{str(len(self.ids))},percentage,{perc}\n')
                f.write(f'salary_info,{str(len(self.ids))},mean_lines_per_doc,{np.mean(num_lines)}\n')
        return df
    
    def language_detection(self, print_interval=500):
        pred_lang = []
        pred_score = []
        total_texts = 0
        other_lang_cnt = 0
        langs = set()
        if self.country.lower() == 'aut':
            lang = 'de'
        elif self.country.lower() == 'uk' or self.country.lower() == 'us':
            lang = 'en'
        else:
            lang = self.country.lower()
        for i, doc in enumerate(self.texts):
            if self.print_mode and i % print_interval == 0: 
                print(f'{i} documents language checked')
            t = doc
            total_texts += 1
            pred = detect_language(t)
            pred1 = pred[0]
            label = pred1['label']
            score = round(pred1['score'], 4)
            pred_lang.append(label)
            pred_score.append(score)
            langs.add(label)
            if label != lang:
                other_lang_cnt += 1
                print('OTHER LANGUAGE FOUND')
                print(f'TEXT: {t}')
                print(pred)
                print()
        total_texts = len(self.ids)
        perc = round((other_lang_cnt/ total_texts * 100), 2)
        df = pd.DataFrame({'id': self.ids, 'detected_lang': pred_lang, 'score': pred_score})
        if self.print_mode:
            print(f'{perc}% not in language {lang}')
            print(f'detected languages: {list(langs)}')
        if self.save_results:
            filename = self.result_path + 'language-detection.csv'
            df.to_csv(filename, index=False)
            with open(self.result_path + 'statistics.csv', mode='a', encoding='utf-8') as f:
                f.write(f'other_language_detected,{str(len(self.ids))},percentage,{perc}\n')
        return df