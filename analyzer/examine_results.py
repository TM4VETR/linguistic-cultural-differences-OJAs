import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util import load_data_from_db


def get_other_lang_texts(db_file): 
    countries = ['aut', 'de', 'es', 'fr', 'it', 'uk', 'us']
    for c in countries:
        print(f'COUNTRY: {c.upper()}')
        if c == 'aut':
            lang = 'de'
        elif c == 'uk' or c == 'us':
            lang = 'en'
        else:
            lang = c
        filepath = f'../results/{c.upper()}/language-detection.csv'
        df = pd.read_csv(filepath)
        df_other = df[df['detected_lang']!=lang]
        print(df_other.shape)

        data = load_data_from_db(db_file, is_html=True, country_code=c.upper())
        df_merged = df_other.merge(data,on='id')
        print('MERGED')
        print(df_merged.shape)
        outpath = f'../results/{c.upper()}/other-lang-detected.csv'
        df_merged = df_merged.replace(r'\n',r'\\n', regex=True) 
        df_merged = df_merged.drop(columns=['paragraphs'])
        df_merged.to_csv(outpath, index=False, sep='\t')

        for i, html in enumerate(df_merged['text']):
            detected_lang = df_merged['detected_lang'].iloc[i]
            score = df_merged['score'].iloc[i]
            print(html)
            print(f'language {detected_lang} detected with score {score}')
            print()
        print()

def merge_stats(result_dir, filename='statistics.csv'):
    df = pd.DataFrame(columns=['country', 'data', 'statistic', 'value'])
    for subdir, dirs, files in os.walk(result_dir):
        # if 'wiki_lingua' not in subdir: # skip wikilingua data 
        country = subdir.split('/')[-1]
        print(country)
        for file in files:
            if file == filename:
                filepath = os.path.join(subdir, file)
                country_df = pd.read_csv(filepath)
                country_df['country'] = country
                country_df = country_df.drop(columns=['num_samples'])
                print(country_df.shape)
                df = pd.concat([df, country_df])
    df = df.sort_values(by=['country'], ascending=True)
    outpath = os.path.join(result_dir, 'merged_statistics.csv')
    df.to_csv(outpath, index=False)
    return df

def table_per_stat(dir, infile, num_countries=7):
    file = os.path.join(dir, infile)
    df = pd.read_csv(file)
    measures = df['measure'].unique()
    for measure in measures:
        print(measure)
        new_df = df[df['measure'] == measure]
        new_df = new_df.drop(columns=['measure'])
        if new_df.shape[0] > num_countries:
            stats = new_df['statistic'].unique()
            stat_val = {}
            for s in stats:
                s_list = list(new_df[new_df['statistic']==s]['value'])
                stat_val[s] = s_list
            new_df = pd.DataFrame(stat_val)
            new_df.insert(0,'country','')
            new_df['country'] = df['country'].unique()
        else:
            stat_name = new_df['statistic'].iloc[0]
            new_df.rename(columns = {'value':stat_name}, inplace = True)
            new_df = new_df.drop(columns=['statistic'])
        print(new_df)
        print(new_df.shape)
        outfile = os.path.join(dir, f'{measure}.csv')
        new_df.to_csv(outfile, index=False)

def create_barplots(inpath, name, errorbar=True):
    df = pd.read_csv(inpath)
    print(df)
    print(df.shape)
    sns.set_context('paper', rc={'font.size':15,'axes.titlesize':15,'axes.labelsize':14, 'xtick.labelsize':12, 'ytick.labelsize':12})
    plt.clf()
    # set color palette and order:
    if df.shape[0] == 7:
        palette = {'AUT': '#fab469', 'DE': '#fa860a', 'ES': '#0a42fa', 'FR': '#0a9603', 'IT': '#e80202', 'UK': '#bf86e3', 'US': '#9405ed'}
    else:
        df = df.set_index('country')
        df = df.reindex(index=['DE', 'ES', 'FR', 'IT', 'EN'])
        df = df.reset_index()
        palette = {'DE': '#fa860a', 'ES': '#0a42fa', 'FR': '#0a9603', 'IT': '#e80202', 'EN': '#9405ed'}
    # plot mean with std if available
    if 'mean' in df.columns:
        outpath = inpath[:-4].replace('results', 'plots') + '.png'
        ax = sns.barplot(data=df, x='country', y='mean', hue='country', palette=palette)
        ax.set_title(name)
        if errorbar:
            plt.errorbar(x = df['country'], y = df['mean'],
                yerr=df['std'], fmt='none', c= 'black', capsize = 2)
        print(outpath)
        plt.savefig(outpath)
        plt.show()
    # plot other statistics
    else:
        columns = df.select_dtypes(include=np.number).columns.tolist() # get numeric columns
        for column in columns:
            filename = inpath.split('/')[-1]
            outpath = inpath[:-len(filename)].replace('results', 'plots')
            outpath += filename[:-4]+'.png' #column+'.png'
            ax = sns.barplot(data=df, x='country', y=column, hue='country', palette=palette)
            ax.set_title(name)
            plt.savefig(outpath)
            plt.show()

def plot_salary_info(inpath, name):
    df = pd.read_csv(inpath)
    print(df)
    plt.clf()
    outpath = inpath[:-4].replace('results', 'plots') + '.png'
    ax = sns.barplot(data=df, x='country', y='percentage', hue='type' , palette=sns.color_palette('colorblind'))
    ax.set_title(name)
    plt.savefig(outpath) 
    plt.show()

def create_paired_plot(file1, file2, name, errorbar=True):
    df1 = pd.read_csv(file1)
    df1['data'] = pd.Series(['OJAs' for x in range(len(df1.index))]) 
    df2 = pd.read_csv(file2)
    df2['data'] = pd.Series(['Wikilingua' for x in range(len(df2.index))]) 
    # couple lines for languages with multiple occurrences
    de_line = df2[df2['country'] == 'DE']
    de_line['country'] = 'AUT'
    en_line = df2[df2['country'] == 'EN']
    en_line['country'] = 'US'
    df2 = df2.replace('EN', 'UK')
    df2 = pd.concat([df2, de_line, en_line], ignore_index=True)
    # merge dfs
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.replace('DE', 'DE/DE').replace('AUT', 'AUT/DE').replace('UK', 'UK/EN').replace('US', 'US/EN')
    df = df.sort_values('country', ascending=True)
    print(df)
    print(df.shape)
    sns.set_context('paper', rc={'font.size':15,'axes.titlesize':15,'axes.labelsize':14, 'xtick.labelsize':10, 
                                 'ytick.labelsize':11, 'legend.fontsize': 11, 'legend.title_fontsize': 11})
    plt.clf()
    # plot
    if 'mean' in df.columns:
        ax = sns.barplot(data=df,  x='country', y='mean', hue='data', palette=sns.color_palette('colorblind')[3:])
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_title(name)
        ax.set_ylim(0, 1200)
        plt.tight_layout()
        if errorbar:
            for i, type in enumerate(list(df['data'].unique())):
                add = -0.2 if i==0 else 0.2
                countries = list(df[df['data']==type]['country'])
                x = [countries.index(item) for item in countries]
                x = [item+add for item in x]
                plt.errorbar(x = x, y = df[df['data']==type]['mean'],
                    yerr=df[df['data']==type]['std'], fmt='none', c= 'black', capsize = 2)
        outpath = file1[:-4].replace('results/statistics/', 'plots/paired/') + '.png'
        plt.savefig(outpath)
        plt.show()
    else:
        columns = df.select_dtypes(include=np.number).columns.tolist() # get numeric columns
        for column in columns:
            ax = sns.barplot(data=df,  x='country', y=column, hue='data', palette=sns.color_palette('colorblind')[3:])
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            ax.set_title(name)
            # ax.set_ylim(0, 0.55)
            plt.tight_layout()
            filename = file1.split('/')[-1]
            outpath = file1[:-len(filename)].replace('results/statistics/', 'plots/paired/')
            outpath += column+'.png' 
            plt.savefig(outpath)
            plt.show()