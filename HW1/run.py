import re
import string
import time
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
import numpy as np
import os
from collections import Counter
import argparse

import nltk
nltk.download('punkt')


import ufal.morphodita as morphodita
czech_tagger = morphodita.Tagger.load('./morpho/czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger')
czech_tokenizer = czech_tagger.newTokenizer().newCzechTokenizer()
english_tagger = morphodita.Tagger.load('./morpho/english-morphium-wsj-140407/english-morphium-wsj-140407-no_negation.tagger')
english_tokenizer = english_tagger.newTokenizer().newEnglishTokenizer()

with open('stopwords_cs.txt', 'r', encoding='utf-8') as f:
    stopwords_czech = f.read().split('\n')
stopwords_set_cs = set(stopwords_czech)

from nltk.corpus import stopwords
stopwords_set_en = set(stopwords.words('english'))

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
        elif lang == 'en':
            tokenizer = english_tokenizer
        token_ranges = morphodita.TokenRanges()
        forms = morphodita.Forms()
        lemmas = morphodita.TaggedLemmas()
        tokenizer.setText(text)
        tokenizer.nextSentence(forms, token_ranges)
        czech_tagger.tag(forms, lemmas)
        tokenized_text = [l.lemma for l in lemmas]

    elif mode == 'clear_stopwords': # this option is combination of nltk tokenization and clearing from stopwords
        if lang == 'cs':
            language = 'czech'
            stopwords = stopwords_set_cs
        elif lang == 'en':
            language = 'english'
            stopwords = stopwords_set_en
        tokenized_text_initial = nltk.word_tokenize(text, language=language)
        tokenized_text = [w for w in tokenized_text_initial if w.lower() not in stopwords]

    return tokenized_text

def preprocess_string(string, mode='default'):
    '''for now - only clearing from internal html tags'''
    clear_string = re.sub('<.*?>', '', string)
    if mode == 'case_insensitive':
        clear_string = clear_string.lower()
    return clear_string


def parse_one_doc(doc, lang='cs'):
    # print(doc)
    doc_id = re.search(r'<DOCNO>(.*?)</DOCNO>', doc).group(1)
    if lang == 'cs':
        text_raw = re.sub(r'<DOC(ID|NO)>.*?</DOC(ID|NO)>\n', '', doc)
        text_raw = '\n'.join(
            [m[1] for m in re.findall(r'<(TITLE|HEADING|TEXT)>((.|\n)*?)</(TITLE|HEADING|TEXT)>', text_raw)])
        text_preprocessed = preprocess_string(text_raw, mode=TEXT_PREPROCESSING_MODE)
    elif lang == 'en':
        text_raw = re.sub(r'<DOC(ID|NO)>.*?</DOC(ID|NO)>\n', '', doc)
        text_raw = '\n'.join([m[1] for m in re.findall(r'<(HD|DH|LD|TE)>((.|\n)*?)</(HD|DH|LD|TE)>', text_raw)])
        text_preprocessed = preprocess_string(text_raw, mode=TEXT_PREPROCESSING_MODE)

    text = tokenize(text_preprocessed, mode=TOKENIZATION_MODE, lang=lang)

    return doc_id, text

def parse_one_xml(xml, lang='cs'):
    docs = [res[0] for res in re.findall(r'<DOC>((.|\n)*?)</DOC>', xml)]
    #print(docs)
    #print(len(docs))
    id_doc_dict = {}
    for i, doc in enumerate(docs):
        doc_id, text = parse_one_doc(doc, lang=lang)
        #if i == 2:
        id_doc_dict[doc_id] = text
#        print(doc_id)
#        print(text)
    return id_doc_dict

def read_lst_file(lst_filepath):
    with open(lst_filepath, 'r', encoding='utf-8') as f:
        #files = f.readlines()
        files = f.read().split('\n')
        if files[-1] == '':
            files = files[:-1]
        return files

def add_or_update_dict(dictionary, key, value):
    if key in dictionary.keys():
        dictionary[key] += value
    else:
        dictionary[key] = value
    return dictionary

def update_tf_df_dicts(tf_dict, df_dict, corpus):
    for doc_id in corpus.keys():
        doc = corpus[doc_id]
        word_set = list(set(doc))
        for word in doc:
            #print(word)
            #df_dict = add_or_update_dict(df_dict, word, 1)
            if doc_id in tf_dict.keys():
                #print(tf_dict)
                #print(tf_dict[doc_id])
                tf_dict[doc_id] = add_or_update_dict(tf_dict[doc_id], word, 1)

            else:
                tf_dict[doc_id] = {word: 1}
        for unique_word in word_set:
            df_dict = add_or_update_dict(df_dict, unique_word, 1)
    return tf_dict, df_dict

def compute_tf(tf_sequence, mode='default'):
    if mode == 'default':
        tf_array = np.array(tf_sequence)
        tf_array = np.log(tf_array)
        tf_array = tf_array + 1
    elif mode == 'boolean':
        tf_array = [1 if tf > 0 else 0 for tf in tf_sequence]
        tf_array = np.array(tf_array)
    elif mode == 'augmented':
        tf_array = np.array(tf_sequence)
        max_tf = tf_array.max()
        tf_array = 0.5 * tf_array
        tf_array = tf_array / max_tf
        tf_array = 0.5 + tf_array
    return tf_array

def compute_idf(df_sequence, doc_number, mode='default'):
    if mode == 'default':
        df_array = np.array(df_sequence)
        df_array = doc_number / df_array
        df_array = np.log(df_array)
    elif mode == 'none':
        df_array = np.ones(len(df_sequence))
    elif mode == 'prob_idf':
        df_array = np.array(df_sequence)
        df_array = (doc_number - df_array) / df_array
        df_array = np.log(df_array)
        df_array = [max(0, idf) for idf in df_array]
        df_array = np.array(df_array)
    return df_array

def calculate_mean_len(id_doc_dict):
    doc_lens = []
    for idx in id_doc_dict.keys():
        doc = id_doc_dict[idx]
        doc_len = len(doc)
        doc_lens.append(doc_len)
    doc_lens = np.array(doc_lens)
    return np.mean(doc_lens), np.median(doc_lens)

def normalization(matrix, method='default'):
    if method == 'default':
        norm_matrix = normalize(matrix, norm='l2', axis=1)
    elif method == 'pivot':
        pass
    return norm_matrix

def create_sparse_matrix(tf_dict, df_dict):
    a = time.time()
    term_names_list = sorted(df_dict.keys(), key= lambda x: x.lower())
    b = time.time()
    #print(f'sort terms: {b-a}')
    #term_names_dict = {k: term_names_list.index(k) for k in term_names_list}
    term_names_dict = {k: idx for idx, k in enumerate(term_names_list)}
    c = time.time()
    #print(f'create dict: {c-b}')
    doc_names_list = sorted(tf_dict.keys(), key= lambda x: x.lower())
    d = time.time()
    #print(f'sort docs: {d-c}')
    i, j, tf, df = [], [], [], []
    #terms, docs = [], []
    a = time.time()
    for doc_id, doc_name in enumerate(doc_names_list):
        # i - doc axis
        curr_doc_dict = tf_dict[doc_name]
        for term_id, term_name in enumerate(curr_doc_dict):
            # j - term axis
            # j_value = term_names_list.index(term_name)
            j_value = term_names_dict[term_name]
            tf_count = curr_doc_dict[term_name]
            df_count = df_dict[term_name]
            i.append(doc_id)
            j.append(j_value)
            tf.append(tf_count)
            df.append(df_count)
    b = time.time()
    #print(f'time for looping: {b-a}')
    # compute tf, idf separately
    a = time.time()
    tf_array = compute_tf(tf, mode=TF_MODE)
    idf_array = compute_idf(df, len(doc_names_list), mode=IDF_MODE)
    tf_idf_array = np.multiply(tf_array, idf_array)
    b = time.time()
    #print(f'time for tf, idf: {b-a}')
    # create sparse matrix
    matrix = coo_matrix((tf_idf_array, (i, j)), shape=(len(doc_names_list), len(term_names_list)))
    c = time.time()
    #print(f'time for matrix: {c-b}')
    norm_matrix = normalization(matrix, method=NORMALIZATION_MODE)
    d = time.time()
    #print(f'time for normalization: {d-c}')
    # create idf array of unique values:
    df_values_list = [df_dict[k] for k in term_names_list]
    idf_unique_list = compute_idf(df_values_list, len(doc_names_list))
#    return tf_idf_array, i, j, terms, docs, tf, df
    return norm_matrix, doc_names_list, term_names_list, idf_unique_list#i, j, v#term_names_list, doc_names_list


def create_matrix_with_keys(lst_filename, lang):
    aaa = time.time()
    tf_dict, df_dict = {}, {}
    xml_list = read_lst_file(lst_filename)
    directory = lst_filename.split('.')[0]
    for xml_fname in xml_list:
        #print(xml_fname)
        with open(f'./{directory}/{xml_fname}', 'r', encoding='utf-8') as f:
            xml = f.read()
        #print('start parsing')
        a = time.time()
        id_doc_dict = parse_one_xml(xml, lang=lang)
        b = time.time()
        #print(f'end parsing, time: {b - a}')
        tf_dict, df_dict = update_tf_df_dicts(tf_dict, df_dict, id_doc_dict)
        c = time.time()
        #print(f'end updating dict, time: {c - b}')

    #print('start creating matrix')
    d1 = time.time()
    # a, b, c, d, e, t, d = create_sparse_matrix(tf_dict, df_dict)
    # return a, b, c, d, e, t, d
    sparse_matrix, doc_names_list, term_names_list, idf_unique_list = create_sparse_matrix(tf_dict, df_dict)
    d2 = time.time()
    #print(f'end making matrix, time: {d2 - d1}')
    eee = time.time()
    #print(f'total time: {eee - aaa}')
    return sparse_matrix, doc_names_list, term_names_list, idf_unique_list


def parse_one_query(string, lang='cs', mode='default'):
    query_id = re.search(r'<num>(.*?)</num>', string).group(1)
    if mode == 'default':
        query_raw = re.search(r'<title>(.*?)</title>', string).group(1)
    elif mode == 'enhanced':
        query_raw = '\n'.join(
            [m[1] for m in re.findall(r'<(title|desc)>((.|\n)*?)</(title|desc)>', string)])
    elif mode == 'enhanced_2':
        query_raw = '\n'.join(
            [m[1] for m in re.findall(r'<(title|desc|narr)>((.|\n)*?)</(title|desc|narr)>', string)])
    query_text = preprocess_string(query_raw, mode=TEXT_PREPROCESSING_MODE)
    query_text = tokenize(query_text, mode=TOKENIZATION_MODE, lang=lang)
    return query_id, query_text

def parse_query_doc(fpath, lang='cs'):
    with open(fpath, 'r', encoding='utf-8') as f:
        xml = f.read()
    queries = [res[0] for res in re.findall(f'<top lang="{lang}">((.|\n)*?)</top>', xml)]
    id_query_dict = {}
    for i, query in enumerate(queries):
        query_id, text = parse_one_query(query, lang=lang, mode=QUERY_CONSTRUCTION_MODE)
        #if i == 2:
        id_query_dict[query_id] = text
    return id_query_dict


def create_sparse_matrix_queries(id_query_dict, term_names_list, idf_unique_list):
    term_names_set = set(term_names_list)  # for faster search if word is in training data
    query_id_list = sorted(id_query_dict.keys(), key=lambda x: x.lower())
    query_term_set = set()
    for query_name in query_id_list:
        cleared_query_list = []
        for word in id_query_dict[query_name]:
            if word in term_names_set:
                cleared_query_list.append(word)
                query_term_set.add(word)
            else:
                print(query_name, word)
        id_query_dict[query_name] = cleared_query_list
    # query_term_set = list(query_term_set) # let's try firstly without creating new ds
    term_ids_dict = {q_term: term_names_list.index(q_term) for q_term in query_term_set}
    term_idf_dict = {q_term: idf_unique_list[term_ids_dict[q_term]] for q_term in query_term_set}
    i, j, tf_idf = [], [], []
    for query_id, query_name in enumerate(query_id_list):
        # i - doc axis
        current_query_counter = dict(Counter(id_query_dict[query_name]))
        for query_term in current_query_counter.keys():
            # j - term axis
            j_value = term_ids_dict[query_term]
            if TF_MODE == 'default' or TF_MODE == 'augmented':
                tf_value = 1 + np.log(current_query_counter[query_term])
            elif TF_MODE == 'boolean':
                tf_value = 1
            idf_value = term_idf_dict[query_term]
            tf_idf_value = tf_value * idf_value
            i.append(query_id)
            j.append(j_value)
            tf_idf.append(tf_idf_value)

    matrix = coo_matrix((tf_idf, (i, j)), shape=(len(query_id_list), len(term_names_list)))
    return matrix, query_id_list, term_names_list

def create_cos_sim_vectors(sparse_matrix_docs, sparse_matrix_queries):
    cos_sim_vectors = []
    a = time.time()
    for i in range(sparse_matrix_queries.shape[0]):
        one_query_vector = sparse_matrix_queries.tocsr()[i,:] # doing slice of query matrix for one query
        #print(type(one_query_vector))
        one_query_matrix = vstack([one_query_vector for i in range(sparse_matrix_docs.shape[0])])
        #print(type(one_query_matrix))
        cos_sim_matrix = sparse_matrix_docs.multiply(one_query_matrix)
        #print(type(cos_sim_matrix))
        cos_sim_vector = cos_sim_matrix.sum(axis=1)
        #print(type(cos_sim_vector))
        cos_sim_vectors.append(cos_sim_vector)
    b = time.time()
    return cos_sim_vectors


def rank_most_similar_documents(cos_sim_vectors, doc_names_list, query_id_list):
    query_rel_docs_dict = {}
    for query_number, cos_sim_vector in enumerate(cos_sim_vectors):
        relevant_doc_num_max = min(1000, np.count_nonzero(cos_sim_vector))
        if relevant_doc_num_max > 0:
            # print(type(cos_sim_vector))
            cos_sim_vector = np.array(cos_sim_vector)
            # print(type(cos_sim_vector))
            argsort = cos_sim_vector.flatten().argsort(kind='mergesort')
            argsort_desc = argsort[::-1]
            argsort_top = argsort_desc[:relevant_doc_num_max]

            query_id = query_id_list[query_number]
            top_rank_values = [float(cos_sim_vector[arg]) for arg in argsort_top]
            #print(len(top_rank_values))
            top_rank_docs = [doc_names_list[arg] for arg in argsort_top]
            query_rel_docs_dict[query_id] = [top_rank_values, top_rank_docs]
        else:
            #print(0)
            pass # maybe return something? Mark it somehow?

    return query_rel_docs_dict

def save_to_file(fname, ranking, experiment_id, iter_id=0):
    experiment_id = experiment_id.split('_')[0]
    # iter_id hardcoded, unnecessary for this hw
    f = open(fname, 'w', encoding='utf-8').close()
    for query_id in ranking.keys():
        scores, doc_ids = ranking[query_id]
        for rank in range(len(scores)):
            doc_id = doc_ids[rank]
            score = scores[rank]
            line = f'{query_id} {iter_id} {doc_id} {rank} {score} {experiment_id}\n'
            #print(line)
            with open(fname, 'a', encoding='utf-8') as f:
                f.write(line)
    return 'done'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ./run -q topics-train_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_train_cs.res
    parser.add_argument("-q", "--Topics", help="file including topics in the TREC format")
    parser.add_argument("-d", "--Documents", help="file including document filenames")
    parser.add_argument("-r", "--Run", help="string identifying the experiment")
    parser.add_argument("-o", "--Output", help="output file")
    args = parser.parse_args()

    # Initializing hyperparams
    LANG = args.Documents.split('.')[0].split('_')[1]

    # Initializing manual hyperparams
    TEXT_PREPROCESSING_MODE = 'case_insensitive'
    TOKENIZATION_MODE = 'nltk'
    IDF_MODE = 'default'
    TF_MODE = 'default'
    NORMALIZATION_MODE = 'default'
    QUERY_CONSTRUCTION_MODE = 'enhanced'
    t1 = time.time()
    print('creating sparse matrix for documents')
    sparse_matrix, doc_names, term_names, idf_unique_list = create_matrix_with_keys(args.Documents, LANG)
    #print(sparse_matrix, doc_names, term_names)
    print('parsing queries')
    id_query_dict = parse_query_doc(args.Topics, lang=LANG)
    print('creating sparse matrix for queries')
    sparce_matrix_queries, query_id_list, term_names_list = create_sparse_matrix_queries(id_query_dict,
                                                                                                  term_names,
                                                                                                  idf_unique_list)
    print('creating cosine similarity vectors')
    cos_sim_vectors = create_cos_sim_vectors(sparse_matrix, sparce_matrix_queries)
    print('ranking')
    ranking = rank_most_similar_documents(cos_sim_vectors, doc_names, query_id_list)
    save_to_file(args.Output, ranking, args.Run)
    t2 = time.time()
    print(f'time for the whole process: {t2-t1} sec')

# TODO:
# 1. parsing input files, extracting language
# 1.5. reuse the preprocessed documents for test inference?
# 2. writing to file
# 3. running the test
# 4. function for processing run arguments
# 5. some code for runs of test variants

# ablations:
# preprocessing: nltk.tokenize, lemmatization, stopwords filtering + (nltk, lemmatization), case normalization, lemma+tags
# query construction: description, narrative
# term weighting (as much as possible)
# idf weighting (as much  as possible)

# python run.py -q topics-train_en.xml -d documents_en.lst -r morphodita -o morphodita_train_en.res