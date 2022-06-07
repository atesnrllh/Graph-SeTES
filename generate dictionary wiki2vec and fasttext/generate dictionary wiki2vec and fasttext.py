import wikipedia2vec
import tagme
import WIKI
import pickle
import os, sys
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("/home/nurullah/fastText/"))))
import fasttext
wiki2vec = wikipedia2vec.Wikipedia2Vec.load("/home/nurullah/Desktop/enwiki_20180420_300d.pkl")
tagme.GCUBE_TOKEN = "29ebf7ce-ad64-4790-9d0a-7a2ba38e40d0-843339462"
w2v = [wiki2vec, tagme.GCUBE_TOKEN]

def ent_dict():
    path = "/media/nurullah/E/datasets/Gayo-Avello dataset/webis-smc-12/webis-smc-12.csv"


    df1 = pd.read_csv(path, sep='\t',encoding='utf-8')
    Query_list1 = df1['Query'].values.tolist()


    #joint_ann_score = [0.15,0.2,0.23,0.25,0.27,0.3,0.35]
    joint_ann_score = [0.35, 0.3, 0.27, 0.25, 0.23, 0.2, 0.15]
    #Query_list1 = list(set(Query_list1))
    for js in joint_ann_score:
        Query_emb_dict1 = {}
        for i, query in enumerate(tqdm(Query_list1)):
            if query not in Query_emb_dict1:
                Query_emb_dict1[query]= WIKI.use_wikipedia2vec2(w2v, query, 0.2, 0.2)

        with open(str(js)+'dict.pickle', 'wb') as f:
            pickle.dump(Query_emb_dict1, f, pickle.HIGHEST_PROTOCOL)


    with open( 'dict.pickle', 'rb') as f:
        x = pickle.load(f)
    print()

import requests, time
def internet_on():
    url = "http://www.kite.com"
    timeout = 5
    try:
        request = requests.get(url, timeout=timeout)
        return False
    except:
        time.sleep(3)
        return True

def ent_dict_task():
    path = "/media/nurullah/E/agnostic_bert/search-master1/datasets/volske/d1_session1.csv"
    df1 = pd.read_csv(path, sep=',',encoding='utf-8')
    Query_list1 = df1['Query'].values.tolist()

    path = "/media/nurullah/E/agnostic_bert/search-master1/datasets/volske/d2_trec1.csv"
    df1 = pd.read_csv(path, sep=',',encoding='utf-8')
    Query_list1 += df1['Query'].values.tolist()

    path = "/media/nurullah/E/agnostic_bert/search-master1/datasets/volske/d3_wikihow1.csv"
    df1 = pd.read_csv(path, sep=',',encoding='utf-8')
    Query_list1 += df1['Query'].values.tolist()

    # path = "/media/nurullah/E/datasets/Gayo-Avello dataset/webis-smc-12/webis-smc-12.csv"
    # df1 = pd.read_csv(path, sep='\t',encoding='utf-8')
    # Query_list1 = df1['Query'].values.tolist()

    #joint_ann_score = [0.10, 0.15,0.2,0.23,0.25,0.27,0.3,0.35]
    joint_ann_score = [0.23]
    Query_list1 = list(set(Query_list1))
    for js in joint_ann_score:
        Query_emb_dict1 = {}
        for i, query in enumerate(tqdm(Query_list1)):
            try:
                if query not in Query_emb_dict1:
                    Query_emb_dict1[query]= WIKI.use_wikipedia2vec2(w2v, query, js, js)
            except:
                while internet_on():
                    a=1
                continue
        with open("fast_dict_task_map" + str(js)+'dict.pickle', 'wb') as f:
            pickle.dump(Query_emb_dict1, f, pickle.HIGHEST_PROTOCOL)


    # with open( 'dict.pickle', 'rb') as f:
    #     x = pickle.load(f)
    print()



def fast_dict():

    path = "/home/nurullah/Dropbox/tez/all_models/gayo/siamese and decider/upsample_fasttext_group/no_entity/9_task_mapping/datasets/super/test/d1_session.csv"
    df1 = pd.read_csv(path, sep='\t',encoding='utf-8')
    Query_list1 = df1['Query'].values.tolist()

    path = "/home/nurullah/Dropbox/tez/all_models/gayo/siamese and decider/upsample_fasttext_group/no_entity/9_task_mapping/datasets/super/test/d2_trec.csv"
    df1 = pd.read_csv(path, sep='\t',encoding='utf-8')
    Query_list1 += df1['Query'].values.tolist()

    path = "/home/nurullah/Dropbox/tez/all_models/gayo/siamese and decider/upsample_fasttext_group/no_entity/9_task_mapping/datasets/super/test/d3_wikihow.csv"
    df1 = pd.read_csv(path, sep='\t',encoding='utf-8')
    Query_list1 += df1['Query'].values.tolist()


    w2v_fastext = fasttext.load_model('/home/nurullah/cc.en.300.bin')
    fast_dict = {}


    Query_list1 = list(set(Query_list1))

    for query in tqdm(Query_list1):
        query = list(query.split(" "))
        for word in query:
            if word not in fast_dict:
                fast_dict[word] = w2v_fastext.get_word_vector(word)


    with open('/home/nurullah/Desktop/gayo_entity2dic_and_fasttext_small/fast_dict_task_map.pickle', 'wb') as f:
        pickle.dump(fast_dict, f, pickle.HIGHEST_PROTOCOL)


    with open('/home/nurullah/Desktop/gayo_entity2dic_and_fasttext_small/fast_dict_task_map.pickle', 'rb') as f:
        x = pickle.load(f)
    print()

if __name__ == '__main__':

    ent_dict()
    #fast_dict()
    #ent_dict()
    #ent_dict_task()
    # df1 = pd.read_csv('/home/nurullah/Dropbox/tez/conference-tasking/all_model_cross_validation/gayo/siamese and decider/upsample_fasttext/data.csv', sep=',',encoding='utf-8')
    #
    # for i, row in df1.iterrows():
    #     print(row["la"])
    #     vv = WIKI3.use_wikipedia2vec(w2v, row["q1"], 0.3, 0.3)
    #     vv = WIKI3.use_wikipedia2vec(w2v, row["q2"], 0.3, 0.3)
    #     print()

