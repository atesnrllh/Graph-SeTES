
import  wikipediaapi
import wikipedia
wiki_wiki  =  wikipediaapi.Wikipedia ('en')
wikipedia.set_lang("en")

from tqdm import tqdm
import tagme
import numpy as np

def print_categories(query):
    page_py = wiki_wiki.page(query)
    categories = page_py.categories
    cat = []
    for title in sorted(categories.keys()):
        cat.append(categories[title].title.replace("Category:",""))
    return cat
def is_query_one_entity(query, relatedness_score):

    try:
        tex, sug = wikipedia.search(query, results = 5,suggestion = True)
        if len(tex) > 0:
            if tex[0].lower() == query.lower():
                #print("tex 2 ",tex[0])
                query = tex[0]
                return query , sug

        if len(tex) > 2:
            res = 0
            cat = 0
            for i in range(0,1):
                for j in range(i+1,5):
                    try:
                        #rel = tagme.relatedness_title((tex[i], tex[j])).relatedness[0].rel
                        rel = tagme.relatedness_title((wiki_wiki.page(tex[i]).title, wiki_wiki.page(tex[j]).title)).relatedness[0].rel
                        cat += len(list(set(     print_categories(wiki_wiki.page(tex[i]).title)) &    set(print_categories(wiki_wiki.page(tex[j]).title)) ) )
                        #print(i,j,rel)
                        if rel is not None:
                            res += rel
                    except:
                        continue
            #print("cat :",cat)
            if res > relatedness_score:
                title = wiki_wiki.page(tex[0]).title
                #print("title 1.5 ", title)
                return title, sug
            else:
                return "", sug
        else:
            if sug is not None:
                #print("------------------------------------------------------------------------------------------------suggest 3 "," ",sug)
                return "",sug

            return "",""
    except:
        return "",""

def is_query_one_entity1(query):
    page_py = wiki_wiki.page(query)

    if page_py.exists():
        #print("title 1 ", page_py.title)
        query = page_py.title
        #print(query, " için sayfa bulunmuştur")
    else:
        try:
            tex, sug = wikipedia.search(query, results = 5,suggestion = True)
            wikipedia.page()
            if len(tex) > 4:
                res = 0
                for i in range(0,5):
                    categories1 = print_categories(tex[i])
                    for j in range(i+1,5):
                        categories2 = print_categories(tex[j])
                        res += len(set(categories1) & set(categories2)) / (len(categories1)**(1./3.) + len(categories2)**(1./3.))

                if res/5 > 1:
                    title = wikipedia.page(tex[0]).title
                    #print("title 1.5 ", title)
                    return title
                else:
                    return query
            else:
                if len(tex) > 0:
                    if tex[0].lower() == query.lower():
                        #print("tex 2 ",tex[0])
                        query = tex[0]
                        return query
                if sug is not None:
                    #print("suggest 3 "," ",sug)
                    query = sug

        except:
            if len(tex) > 0:
                if tex[0].lower() == query.lower():
                    #print("tex 2 ",tex[0])
                    query = tex[0]
                    return query
            if sug is not None:
                #print("suggest 3 "," ",sug)
                query = sug
            else:
                if len(tex) > 0:
                    #print("tex 4",tex[0], "len ",len(tex))
                    if len(tex) > 4:
                        return tex[0]
    return query


def use_wiki_word(query, wiki):

    #pivot_list = pivot_str.strip().split(" ")
    #s = set(pivot_list)
    diff_list  = query.strip().split(" ")
    #print("-------------word------------------")
    vector_list = []
    counter = 0
    check_list = np.zeros(len(diff_list))
    for i, item in enumerate(diff_list):
        try:
            vector_list.append( wiki.get_word_vector(item.lower()) )
            #print(item)
            counter += 1
            check_list[i] = 1
        except:
            check_list[i] = 0
            #print("no word vector")
            continue
    return vector_list, check_list


def use_wiki_word_sug(query, pivot_str, wiki, check_list):
    query_list = query.split(" ")
    pivot_list = pivot_str.strip().split(" ")
    s = set(pivot_list)
    diff_list = [x for x in query_list if x not in s]
    #print("-------------word------------------")
    vector_list = []
    counter = 0
    for i, item in enumerate(diff_list):
        if check_list[i] == 0:
            try:
                vector_list.append( wiki.get_word_vector(item.lower()) )
                #print(item)
            except:
                #print("no word vector")
                continue
    return vector_list, check_list


def delete_same_entities(text_a_ann):
    if type(text_a_ann) is not str:
        i= -1
        while len(text_a_ann.annotations):
            i = i+1
            j = -1
            if len(text_a_ann.annotations) == i:
                break
            while len(text_a_ann.annotations):
                j = j+1
                if len(text_a_ann.annotations) == j:
                    break
                if j > i:
                    if len(text_a_ann.annotations[j].mention.split()) < len(text_a_ann.annotations[i].mention.split()):
                        if text_a_ann.annotations[j].mention in text_a_ann.annotations[i].mention:
                            text_a_ann.annotations.remove(text_a_ann.annotations[j])
                            j = j - 1
                    else:
                        if text_a_ann.annotations[i].mention in text_a_ann.annotations[j].mention:
                            text_a_ann.annotations.remove(text_a_ann.annotations[i])
                            i = i -1
                            break
    return text_a_ann


def get_entity_vec(wiki2vec, text_a_ann, query, entity_vec_threshold):
    #print("---------",query,"------entities-----------------")
    vector_list1 = []
    pivot_str1 = ""
    pivot_str11= ""
    if type(text_a_ann) is not str:
        for ann in text_a_ann.get_annotations(entity_vec_threshold):
            try:
                vec = wiki2vec.get_entity_vector(ann.entity_title)
                if ann.mention in query:
                    vector_list1.append(vec)
                    pivot_str1 +=" " + ann.mention
                    #print(ann.mention,"------>",ann.entity_title)
            except KeyError:
                continue

    return vector_list1, pivot_str1.strip()


def joint_assessment(q1,q2, joint_ann_score):
    #text_a_ann = tagme.annotate(query)
    #print("iki sorgunun ortak entity lerini bulma", joint_ann_score)
    query = q1 + " and " + q2
    text_r_ann = tagme.annotate(query)
    query = q2 + " and " + q1
    text_l_ann = tagme.annotate(query)
    text_x_ann = tagme.annotate("good")
    text_x_ann.annotations = []

    for it1 in text_r_ann.get_annotations(joint_ann_score):
        check = True
        for it2 in text_l_ann.get_annotations(joint_ann_score):
            if it1.mention == it2.mention:
                if it1.score > it2.score:
                    text_x_ann.annotations.append(it1)
                else:
                    text_x_ann.annotations.append(it2)
                check = False
        if check:
            text_x_ann.annotations.append(it1)

    q1_go_is_one = True
    q2_go_is_one = True
    for ann in text_x_ann.get_annotations(joint_ann_score):
        if ann.mention in q1:
            q1_go_is_one = False

        if ann.mention in q2:
            q2_go_is_one = False

    return q1_go_is_one, q2_go_is_one, text_x_ann

from time import sleep
def check_page_in_wiki_direct(q2):

    try:
        page_py = wiki_wiki.page(q2)
        #page = wikipedia.page(q2,auto_suggest=True)
        text_b_ann = ""
        q2_page = True
        if page_py.text != "":
            text_b_ann = tagme.annotate(page_py.title)
            q2 = page_py.title
            text_b_ann.annotations[0].score = 1
            #print(q2, " için sayfa bulunmuştur")
            q2_page = False
    except:
        sleep(1000)
        page_py = wiki_wiki.page(q2)
        #page = wikipedia.page(q2,auto_suggest=True)
        text_b_ann = ""
        q2_page = True
        if page_py.text != "":
            text_b_ann = tagme.annotate(page_py.title)
            q2 = page_py.title
            text_b_ann.annotations[0].score = 1
            #print(q2, " için sayfa bulunmuştur")
            q2_page = False

    return q2, text_b_ann, q2_page

def correct_query_by_wiki(q1,text_x_ann,relatednes_score):
    sug1 = ""

    #print(q1, " için sorgu düzeltme")
    qx1, sug1 = is_query_one_entity(q1,relatednes_score)
    text_a_ann = text_x_ann
    if qx1 != "":
        text_a_ann = tagme.annotate(qx1)
        q1 = qx1
    elif sug1 != "" and sug1 != None:
        qx1, sug1 = is_query_one_entity(sug1,relatednes_score)
        if qx1 != "":
            text_a_ann = tagme.annotate(qx1)
            q1 = qx1

    return q1, text_a_ann, sug1

def first_correct_query_by_wiki(q1, relatedness_score):
    sug1 = ""
    #print(q1, " için sorgu düzeltme")
    qx1, sug1 = is_query_one_entity(q1, relatedness_score)
    if qx1 != "":
        text_a_ann = tagme.annotate(qx1)
        q1 = qx1
    elif sug1 != "":
        qx1, sug1 = is_query_one_entity(sug1,relatedness_score)
        if qx1 != "":
            text_a_ann = tagme.annotate(qx1)
            q1 = qx1

    return q1, text_a_ann, sug1

def wiki_word_vector_sug(q1, pivot_str1,wiki2vec,q1_page, sug1, vector_list1):

    s = pivot_str1.split(" ")
    for x in s:
        q1 = q1.replace(x,'')
    check_alpha = any(c.isalpha() for c in q1)

    if check_alpha:
        word_vec1, sug1_list = use_wiki_word(q1, wiki2vec)
        #if q1_page and q1_go_is_one:
            #if sug1 != None and sum(sug1_list) < len(sug1_list):
            #    word_vec11, sug11_count = use_wiki_word_sug(sug1, pivot_str1, wiki2vec, sug1_list)
            #    word_vec1 += word_vec11
        vector_list1 = vector_list1 + word_vec1
    return vector_list1

def use_wikipedia2vec1(w2v, q1,joint_ann_score, relatedness_score):


    wiki2vec, tagme.GCUBE_TOKEN = w2v[0], w2v[1]


    #sorguyu direct olarak wikipedia da varsa direk o bilgiyi kullanmak için, başka aramalara gerek olmaz çünkü wikipedia da direk karşılığı varsa güçlü bir sorgu
    q1, text_a_ann, q1_page = check_page_in_wiki_direct(q1)

    #context den öğrenmek için iki sorguyu beraber incelemek için
    #joint_ann_score = 0.1

    #düzeltme belki burada olabilir
    # sorguda yazım yanlışı varsa, büyük küçük harf farkı, anlamı google da olan ama wikide biraz farklı olan, entity si bulunma ihtimali olan sorgular için
    sug1, sug2 = "", ""
    #relatedness_score = 0.5
    if q1_page:
        q1, text_a_ann, sug1 = correct_query_by_wiki(q1, text_a_ann,relatedness_score)

    if q1_page:
        text_a_ann = delete_same_entities(text_a_ann)

    if text_a_ann == "":
        text_a_ann = tagme.annotate(q1)# entity score bazen düşük oluyor ama çoğu kelimeyi içeriyor

    vector_list1, pivot_str1 = get_entity_vec(wiki2vec, text_a_ann, q1, joint_ann_score)

    # word to vec
    vector_list1 = wiki_word_vector_sug(q1, pivot_str1,wiki2vec,q1_page, sug1, vector_list1)


    #print("------------------------------------------------------------")
    if len(vector_list1) == 0:
        #print("no wiki entity and word")
        return list([np.zeros(300)])

    return vector_list1

def use_wikipedia2vec(w2v, q1,joint_ann_score, relatedness_score):


    wiki2vec, tagme.GCUBE_TOKEN = w2v[0], w2v[1]

    text_a_ann = tagme.annotate(q1)# entity score bazen düşük oluyor ama çoğu kelimeyi içeriyor
    text_a_ann = delete_same_entities(text_a_ann)

    vector_list1, pivot_str1 = get_entity_vec(wiki2vec, text_a_ann, q1, joint_ann_score)

    # word to vec
    vector_list1 = wiki_word_vector_sug(q1, pivot_str1,wiki2vec,True, True, vector_list1)


    #print("------------------------------------------------------------")
    if len(vector_list1) == 0:
        #print("no wiki entity and word")
        return list([np.zeros(300)])

    return vector_list1

import helper
def data_loader(w2v, bs, t_samples, t_labels, t_session, query_wiki):


    b_v1, b_v2 = [], []
    t_v1, t_v2 = [], []

    for q1, q2 in tqdm(t_samples):
        #print(q1, "   ", q2)
        v1, v2 = query_wiki[q1], query_wiki[q2]
        if v1 != None:
            t_v1.append(v1)
            t_v2.append(v2)

    s1,s2, label, session, t_samples  = helper.batchify( t_v1, t_v2, t_labels, t_session, bs,t_samples)

    return s1, s2, label, session, t_samples

def data_loader_test(bs, t_samples, t_labels, t_session, query_wiki):


    b_v1, b_v2 = [], []
    t_v1, t_v2 = [], []

    for q1, q2 in tqdm(t_samples):
        #print(q1, "   ", q2)
        v1, v2 = query_wiki[q1], query_wiki[q2]
        if v1 != None:
            t_v1.append(v1)
            t_v2.append(v2)

    s1,s2, label, session, t_samples  = helper.batchify_test( t_v1, t_v2, t_labels, t_session, bs,t_samples)

    return s1, s2, label, session, t_samples
#e-commerce, commerce online


def get_cat_info(q1, q2):

    #rel = tagme.relatedness_title((tex[i], tex[j])).relatedness[0].rel

    q1 = print_categories(wiki_wiki.page(q1).title)
    q2 = print_categories(wiki_wiki.page(q2).title)
    intersect = len(list(     set(q1) &    set(q2) ) )
    payda = (len(q1) + len(q2) - intersect)
    if payda == 0:
        return 0
    cat =  intersect / payda

    return cat

