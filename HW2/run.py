import re
import time
import os
import argparse
import pandas as pd

import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords_set_en = set(stopwords.words('english'))

import ufal.morphodita as morphodita
czech_tagger = morphodita.Tagger.load('./czech-morfflex-pdt-161115.tagger')
czech_tokenizer = czech_tagger.newTokenizer().newCzechTokenizer()
english_tagger = morphodita.Tagger.load('./english-morphium-wsj-140407-no_negation.tagger')
english_tokenizer = english_tagger.newTokenizer().newEnglishTokenizer()

with open('stopwords_cs.txt', 'r', encoding='utf-8') as f:
    stopwords_czech = f.read().split('\n')
stopwords_set_cs = set(stopwords_czech)


import pyterrier as pt
if not pt.started():
    pt.init()


def create_doc_df(lst_filename, lang):
    #print(f'create_doc_df: lang: {lang}')
    total_dict = {}
    aaa = time.time()
    #    tf_dict, df_dict = {}, {}
    xml_list = read_lst_file(lst_filename)
    directory = lst_filename.split('.')[0]
    counter = 0
    for xml_fname in xml_list:

        # print(xml_fname)
        with open(f'./{directory}/{xml_fname}', 'r', encoding='utf-8') as f:
            xml = f.read()
        # print('start parsing')
        a = time.time()
        id_doc_list = parse_one_xml(xml, lang=lang)
        for element in id_doc_list:
            # print(id_doc_dict)
            total_dict[counter] = element
            counter += 1
        # print(len(total_dict.keys()))
        b = time.time()
        # print(f'end parsing, time: {b - a}')
        #        tf_dict, df_dict = update_tf_df_dicts(tf_dict, df_dict, id_doc_dict)
        c = time.time()
        # counter += 1
    #        print(f'end updating dict, time: {c - b}')

    # print('start creating matrix')
    d1 = time.time()
    # a, b, c, d, e, t, d = create_sparse_matrix(tf_dict, df_dict)
    # return a, b, c, d, e, t, d
    df = pd.DataFrame.from_dict(total_dict, columns=['docno', 'text'], orient='index')
    #    sparse_matrix, doc_names_list, term_names_list, idf_unique_list = create_sparse_matrix(tf_dict, df_dict)
    d2 = time.time()
    #print(f'end making dataframe, time: {d2 - d1}')
    eee = time.time()
    # print(f'total time: {eee - aaa}')
    return df  # df #sparse_matrix, doc_names_list, term_names_list, idf_unique_list


def read_lst_file(lst_filepath):
    with open(lst_filepath, 'r', encoding='utf-8') as f:
        # files = f.readlines()
        files = f.read().split('\n')
        if files[-1] == '':
            files = files[:-1]
        return files


def tokenize(text, mode='default', lang='cs'):
    '''tokenizing text
        :param str text: input text
        :param str mode: mode of tokenization.
            Values: 'default' (splitting by spaces and punct); 'nltk' (nltk tokenization method),
            'morphodita' (morphodita tokenization method)
        :param str lang: language
    '''
    if mode == 'default':
        punctuation = "[\!\"\#\$%&\'\(\)\*\+\,\-\.:;=\?@\[\]\^_\<\>\{\}\\/]{0,5}\s+[\!\"\#\$%&\'\(\)\*\+\,\-\.:;=\?@\[\]\^_\<\>\{\}\\/]{0,5}"  # add
        text = re.sub(r'^[\!\"\#\$%&\'\(\)\*\+\,\-\.:;=\?@\[\]\^_\<\>\{\}\\/]', r'^', text)
        text = re.sub(r'[\!\"\#\$%&\'\(\)\*\+\,\-\.:;=\?@\[\]\^_\<\>\{\}\\/]$', r'$', text)
        tokenized_text = [i for i in re.split(punctuation, text) if i != '']
    elif mode == 'default__':
        regex = r"\w+|[^\w\s]"
        tokenized_text = re.findall(regex, text)

    elif mode == 'nltk':
        if lang == 'cs':
            language = 'czech'
        elif lang == 'en':
            language = 'english'
        tokenized_text = nltk.word_tokenize(text, language=language)

    elif mode == 'morphodita':
        if lang == 'cs':
            tokenizer = czech_tokenizer
            tagger = czech_tagger
        elif lang == 'en':
            tokenizer = english_tokenizer
            tagger = english_tagger
        #print(tokenizer)
        token_ranges = morphodita.TokenRanges()
        forms = morphodita.Forms()
        lemmas = morphodita.TaggedLemmas()
        tokenizer.setText(text)
        tokenizer.nextSentence(forms, token_ranges)
        tagger.tag(forms, lemmas)
        tokenized_text = [l.lemma for l in lemmas]
        tokenized_text = [t+' ' for t in tokenized_text]
        tokenized_text = ''.join(tokenized_text)
        tokenized_text = re.sub(r'[\!\"\#\$%&\'\(\)\*\+\,\-\.:;=\?@\[\]\^_\<\>\{\}\\/0-9]', r'', tokenized_text)

    elif mode == 'clear_stopwords':  # this option is combination of nltk tokenization and clearing from stopwords
        if lang == 'cs':
            language = 'czech'
            stopwords = stopwords_set_cs
        elif lang == 'en':
            language = 'english'
            stopwords = stopwords_set_en
        tokenized_text_initial = nltk.word_tokenize(text, language=language)
        tokenized_text = [w for w in tokenized_text_initial if w.lower() not in stopwords]

    elif mode == 'nltk_clear_punct':
        if lang == 'cs':
            language = 'czech'
            stopwords = stopwords_set_cs
        elif lang == 'en':
            language = 'english'
            stopwords = stopwords_set_en
        tokenized_text_initial = nltk.word_tokenize(text, language=language)
        tokenized_text = [w for w in tokenized_text_initial if w not in string.punctuation]

    if type(tokenized_text) == list:
        tokenized_text = ' '.join(tokenized_text)
    return tokenized_text


def preprocess_string(string, mode='default'):
    '''for now - only clearing from internal html tags'''
    clear_string = re.sub('<.*?>', '', string)

    clear_string = re.sub("[/\']", ' ', clear_string)

    if mode == 'case-sensitive':
        # clear_string = clear_string.lower()

        clear_string = ' '.join([case_preservation(w) for w in clear_string.split(' ')])
    return clear_string


def case_preservation(word,
                      tags={'title': 'TITLECASE', 'upper': 'UPPERCASE'}):  # {'title':'<TITLE>', 'upper':'<UPPER>'}
    if word.isupper() and len(word) > 1:
        word = tags['title'] + word
    elif word.istitle():
        word = tags['upper'] + word
    return word


def parse_one_doc(doc, lang='cs'):
    # print(doc)
    doc_id = re.search(r'<DOCNO>(.*?)</DOCNO>', doc).group(1)
    if lang == 'cs':
        text_raw = re.sub(r'<DOC(ID|NO)>.*?</DOC(ID|NO)>\n', '', doc)
        text_raw = '\n'.join(
            [m[1] for m in re.findall(r'<(TITLE|HEADING|TEXT)>((.|\n)*?)</(TITLE|HEADING|TEXT)>', text_raw)])
        text_preprocessed = preprocess_string(text_raw, mode=args.Text_Preprocessing_Mode)
    elif lang == 'en':
        text_raw = re.sub(r'<DOC(ID|NO)>.*?</DOC(ID|NO)>\n', '', doc)
        text_raw = '\n'.join([m[1] for m in re.findall(r'<(HD|DH|LD|TE)>((.|\n)*?)</(HD|DH|LD|TE)>', text_raw)])
        text_preprocessed = preprocess_string(text_raw, mode=args.Text_Preprocessing_Mode)

    text = tokenize(text_preprocessed, mode=args.Tokenization_Mode, lang=lang)

    return doc_id, text


def parse_one_xml(xml, lang='cs'):
    docs = [res[0] for res in re.findall(r'<DOC>((.|\n)*?)</DOC>', xml)]
    # print(docs)
    # print(len(docs))
    #    id_doc_dict = {}
    id_doc_list = []
    for i, doc in enumerate(docs):
        doc_id, text = parse_one_doc(doc, lang=lang)
        # if i == 2:
        if type(text) == str:
            pass
        elif type(text) == list:
            text = ' '.join(text)
        id_doc_list.append([doc_id, text])
    #        print(doc_id)
    #        print(text)
    return id_doc_list

def create_index(doc_df, index_path='./index_df', overwrite=True):
    indexer = pt.DFIndexer(index_path, overwrite=overwrite, tokeniser="UTFTokeniser", stopwords=None)#tokeniser="whitespace")
    index_ref = indexer.index(doc_df["text"], doc_df["docno"])
    index_ref.toString()

    index = pt.IndexFactory.of(index_ref)

    print(index.getCollectionStatistics().toString())

    return index


def parse_one_query(string, lang='cs', mode='default'):
    query_id = re.search(r'<num>(.*?)</num>', string).group(1)
    if mode == 'default':
        query_raw = re.search(r'<title>(.*?)</title>', string).group(1)
    elif mode == 'enhanced':
        query_raw = ' '.join(
            [m[1] for m in re.findall(r'<(title|desc)>((.|\n)*?)</(title|desc)>', string)])
    elif mode == 'enhanced_2':
        query_raw = ' '.join(
            [m[1] for m in re.findall(r'<(title|desc|narr)>((.|\n)*?)</(title|desc|narr)>', string)])

    #print(query_raw)
    query_text = preprocess_string(query_raw, mode=args.Text_Preprocessing_Mode)
    #print(query_text)
    query_text = tokenize(query_text, mode=args.Tokenization_Mode, lang=lang)

    #print(query_text)
    #query_text = ' '.join(query_text)
    #query_text = re.sub('[/\']', ' ', query_text)
    return query_id, query_text

def parse_query_doc(fpath, lang='cs'):
    with open(fpath, 'r', encoding='utf-8') as f:
        xml = f.read()
    queries = [res[0] for res in re.findall(f'<top lang="{lang}">((.|\n)*?)</top>', xml)]
    id_query_dict, query_id_dict = {}, {}
    for i, query in enumerate(queries):
        query_id, text = parse_one_query(query, lang=lang, mode=args.Query_Construction_Mode)
        #if i == 2:
        text = text.lower()
        id_query_dict[query_id] = text
        query_id_dict[text] = query_id
    return id_query_dict, query_id_dict

def create_result_dataframe(index, id_query_dict, model, query_expansion=None):
    #result_df = pd.DataFrame(columns=['qid', 'docid', 'docno', 'rank', 'score', 'query'])
    result_df = []
    if query_expansion is not None:
        print('go to query expansion')
        br = pt.BatchRetrieve(index, wmodel=model, controls={"qemodel": query_expansion, "qe": "on"})
    else:
        br = pt.BatchRetrieve(index, wmodel=model)
    for query in id_query_dict.keys():
        curr_df = br.search(id_query_dict[query])
        #print(curr_df.shape)
        #print(id_query_dict[query])
        #print(curr_df)
        result_df.append(curr_df)#, ignore_index=True)
        #print(len(result_df))
    result_df = pd.concat(result_df)
    return result_df

def save_result(result_df, query_id_dict, experiment_id, result_filename, iter_id=0):
    #experiment_id = result_filename.split('_')[0]
    # query_id, 0, docid,  rank, score, experiment_name
    result_df['query_id'] = result_df['query'].apply(lambda x: query_id_dict[x])
    result_df['iter'] = iter_id
    result_df['experiment_name'] = experiment_id
    df_to_save = result_df[['query_id', 'iter', 'docno', 'rank', 'score', 'experiment_name']]
    df_to_save.to_csv(result_filename, header=False, index=False, sep=' ')

    return df_to_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ./run -q topics-train_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_train_cs.res
    parser.add_argument("-q", "--Topics", help="file including topics in the TREC format")
    parser.add_argument("-d", "--Documents", help="file including document filenames")
    parser.add_argument("-r", "--Run", help="string identifying the experiment")
    parser.add_argument("-o", "--Output", help="output file")
    parser.add_argument("-preprocessing", "--Text_Preprocessing_Mode", help="text preprocessing mode: out of {default, case-sensitive}", default="default")
    parser.add_argument("-tokenization", "--Tokenization_Mode", help="tokenization mode: out of {default, nltk, morphodita, clear_stopwords, nltk_clear_punct, nltk_clear_punct_case_insensitive}", default="default")
    parser.add_argument("-weighting", "--Weighting_Mode", help="weighting mode: out of {TF_IDF, Tf, BM25}", default="TF_IDF")
    parser.add_argument("-query_expansion", "--Query_Expansion_Mode", help="Query Construction mode: out of {default, on}", default="default")
    parser.add_argument("-query_construction", "--Query_Construction_Mode", help="Query Construction mode: out of {default, enhanced, enhanced_2}", default="default")
    args = parser.parse_args()
    print(args.Text_Preprocessing_Mode, args.Tokenization_Mode, args.Weighting_Mode, args.Query_Expansion_Mode, args.Query_Construction_Mode)
    # Initializing hyperparams
    LANG = args.Documents.split('.')[0].split('_')[-1]
    print(f'language: {LANG}')

    # Initializing manual hyperparams
    t1 = time.time()
    print('creating dataframe for documents, doing preprocessing')
    cs_doc_df = create_doc_df(args.Documents, LANG)
    print('creating index')
    cs_index = create_index(cs_doc_df, index_path=f'./{LANG}_index', overwrite=True)
    #print(sparse_matrix, doc_names, term_names)
    print('parsing queries')
    cs_id_query_dict, cs_query_id_dict = parse_query_doc(args.Topics, lang=LANG)
    print('doing the search')
    cs_result_df = create_result_dataframe(cs_index, cs_id_query_dict, args.Weighting_Mode, query_expansion=args.Query_Expansion_Mode)
    print('saving result')
    cs_final = save_result(cs_result_df, cs_query_id_dict, args.Run, args.Output)
    t2 = time.time()
    print(f'time for the whole process: {t2-t1} sec')

