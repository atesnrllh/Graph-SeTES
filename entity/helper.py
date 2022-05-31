
import nltk, os, torch
import pandas as pd
import numpy as np
from random import sample
import helper
import WIKI3
nltk.download('punkt')


def tokenize(s, tokenize):
    """Tokenize string."""
    if tokenize:
        return nltk.word_tokenize(s)
    else:
        return s.split()



def get_test_data(directory, Query_emb_dict):

    df = pd.read_csv(directory, sep='\t',encoding='utf-8')
    Query_list = df['Query'].values.tolist()
    UserID_list = df['UserID'].values.tolist()
    Mission_list = df['MissionID'].values.tolist()
    time_list = df['TimeStamp'].values.tolist()


    query_list, labels, userid, time = [], [], [], []

    for i in range(len(Query_list)-1):
        if Query_list[i] != Query_list[i+1]:
            query_list.append(Query_list[i])
            labels.append(Mission_list[i])
            userid.append(UserID_list[i])
            time.append(time_list[i])

    if Query_list[-1] != Query_list[-2]:# last query
        query_list.append(Query_list[-1])
        labels.append(Mission_list[-1])
        userid.append(UserID_list[-1])
        time.append(time_list[-1])

    print("sample size: ",len(query_list))

    q1, q2, lab, ses= [], [], [], []
    for i in range(len(query_list)): # from first to last-1 query
        for j in range(i+1,len(query_list)): # from first to last-1 query
            if (userid[i] != userid[j]):
                break
            q1.append(query_list[i])
            q2.append(query_list[j])
            if diff_time(time[j], time[i]) < 26:
                ses.append(-1)
            else:
                ses.append(1)
            if labels[i] == labels[j]:
                lab.append(1)
            else:
                lab.append(0)


    #t_v1, t_v2, label,  session,   samples  = WIKI3.data_loader_test (256, list(zip(q1,q2)), lab, ses,  Query_emb_dict)

    return  userid, labels, list(zip(q1, q2)), lab,  ses

def load_word_embeddings(directory, file, dictionary):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            embeddings_index[word] = np.array(list(map(float, vec.split())))
    f.close()
    return embeddings_index

def div_data(samples, ind):
    train, test = [], []
    for i, item in enumerate(samples):
        if i in ind:
            train.append(item)
        else:
            test.append(item)

    return train, test
def data_convert_train_valid(samples, pairs, pairs_labels,time1_samples, time2_samples):

    l = len(pairs_labels)
    f = int(l*9 / 10)  # number of elements you need
    indices = sample(range(l), f)
    # indices.sort()

    train_samples, test_samples = div_data(samples, indices)

    train_pairs, test_pairs = div_data(pairs, indices)
    train_labels,test_labels = div_data(pairs_labels, indices)
    train_time1_samples,test_time1_samples = div_data(time1_samples, indices)
    train_time2_samples,test_time2_samples = div_data(time2_samples, indices)

    return list(train_samples), list(test_samples), list(train_labels), list(test_labels), list(train_time1_samples),\
                              list(test_time1_samples),list(train_time2_samples),list(test_time2_samples)


def batchify(data1, labels, sessions, bsz):
    """Transform data into batches."""
    # np.random.shuffle(data)
    batched_data1 = []
    batched_data2 = []
    batched_labels = []
    batched_t1 = []
    batched_sesion = []
    sample_b= []
    for i in range(len(data1)):
        if i % bsz == 0:
            batched_data1.append([data1[i]])
            #batched_data2.append([data2[i]])
            batched_labels.append([labels[i]])
            batched_sesion.append([sessions[i]])
            #sample_b.append([sample[i]])
        else:
            batched_data1[len(batched_data1) - 1].append(data1[i])
            #batched_data2[len(batched_data2) - 1].append(data2[i])
            batched_labels[len(batched_labels) - 1].append(labels[i])
            batched_sesion[len(batched_sesion) - 1].append(sessions[i])
            #sample_b[len(sample_b) - 1].append(sample[i])

    return batched_data1[:-1], batched_labels[:-1], batched_sesion[:-1]

def test_batchify(data1, labels, sessions, bsz):
    """Transform data into batches."""
    # np.random.shuffle(data)
    batched_data1 = []
    batched_data2 = []
    batched_labels = []
    batched_t1 = []
    batched_sesion = []
    sample_b= []
    for i in range(len(data1)):
        if i % bsz == 0:
            batched_data1.append([data1[i]])
            #batched_data2.append([data2[i]])
            batched_labels.append([labels[i]])
            batched_sesion.append([sessions[i]])
            #sample_b.append([sample[i]])
        else:
            batched_data1[len(batched_data1) - 1].append(data1[i])
            #batched_data2[len(batched_data2) - 1].append(data2[i])
            batched_labels[len(batched_labels) - 1].append(labels[i])
            batched_sesion[len(batched_sesion) - 1].append(sessions[i])
            #sample_b[len(sample_b) - 1].append(sample[i])

    return batched_data1, batched_labels, batched_sesion

def batchify_test(data1, data2, labels, sessions, bsz, sample):
    """Transform data into batches."""
    # np.random.shuffle(data)
    batched_data1 = []
    batched_data2 = []
    batched_labels = []
    batched_t1 = []
    batched_sesion = []
    sample_b= []
    for i in range(len(data1)):
        if i % bsz == 0:
            batched_data1.append([data1[i]])
            batched_data2.append([data2[i]])
            batched_labels.append([labels[i]])
            batched_sesion.append([sessions[i]])
            sample_b.append([sample[i]])
        else:
            batched_data1[len(batched_data1) - 1].append(data1[i])
            batched_data2[len(batched_data2) - 1].append(data2[i])
            batched_labels[len(batched_labels) - 1].append(labels[i])
            batched_sesion[len(batched_sesion) - 1].append(sessions[i])
            sample_b[len(sample_b) - 1].append(sample[i])

    return batched_data1, batched_data2, batched_labels,  batched_sesion, sample_b   #, batched_sesion


import wikipedia2vec
import tagme

'''
def use_wikipedia2vec(w2v, query):
    #query = " ".join(query)
    wiki2vec, tagme.GCUBE_TOKEN = w2v[0], w2v[1]
    text_a_ann = tagme.annotate(query)
    #print("###################################")
    #print("###################################")
    #print(query)

    entity_word_list = []
    pivot_str = ""
    vector_list = []
    #print("---------entities-----------------")
    for ann in text_a_ann.get_annotations(0.1):
        try:
            vector_list.append( wiki2vec.get_entity_vector(ann.entity_title) )
            pivot_str +=" " + ann.mention
            #print(ann.mention)
        except KeyError:
            #print("no entity vector")
            continue

    query_list = query.split(" ")
    pivot_list = pivot_str.strip().split(" ")
    s = set(pivot_list)
    diff_list = [x for x in query_list if x not in s]
    #print("-------------word------------------")
    for item in diff_list:
        try:
            vector_list.append( wiki2vec.get_word_vector(item) )
            #print(item)
        except:
            #print("no word vector")
            continue

    #print("------------------------------------------------------------")
    if len(vector_list) == 0:
        print("no wiki entity or word")
    return vector_list
'''
def use_wiki_word(query, pivot_str, wiki):
    query_list = query.split(" ")
    pivot_list = pivot_str.strip().split(" ")
    s = set(pivot_list)
    diff_list = [x for x in query_list if x not in s]
    print("-------------word------------------")
    vector_list = []
    for item in diff_list:
        try:
            vector_list.append( wiki.get_word_vector(item) )
            print(item)
        except:
            #print("no word vector")
            continue
    return vector_list
def use_wikipedia2vec(w2v, query,q1,q2):
    #query = " ".join(query)
    wiki2vec, tagme.GCUBE_TOKEN = w2v[0], w2v[1]
    text_a_ann = tagme.annotate(query)
    ann_list = []
    for i, ann1 in enumerate(text_a_ann.annotations):
        for j, ann2 in enumerate(text_a_ann.annotations):
            if i > j :
                if ann1.mention in ann2.mention:
                    text_a_ann.annotations.remove(ann1)
                    break
            else:
                break
    #text_a_ann.annotations = ann_list

    print("###################################")
    print("###################################")
    print(query)

    pivot_str1 = ""
    pivot_str2 = ""
    vector_list1 = []
    vector_list2 = []
    print("---------entities-----------------")

    for ann in text_a_ann.get_annotations(0.1):
        try:
            vec = wiki2vec.get_entity_vector(ann.entity_title)
            if ann.mention in q1:
                vector_list1.append(vec)
                pivot_str1 +=" " + ann.mention
            if ann.mention in q2:
                vector_list2.append(vec)
                pivot_str2 +=" " + ann.mention
            print(ann.mention,"------>",ann.entity_title)

            #print(ann.mention)
        except KeyError:
            print("no entity vector")
            continue

    word_vec1 = use_wiki_word(q1, pivot_str1, wiki2vec)
    word_vec2 = use_wiki_word(q2, pivot_str2, wiki2vec)

    vector_list1 = vector_list1 + word_vec1
    vector_list2 = vector_list2 + word_vec2

    print("------------------------------------------------------------")
    if len(vector_list1) == 0 or len(vector_list2) == 0:
        print("no wiki entity or word")
    return vector_list1, vector_list2


# def sequence_to_tensors(w2v, sequence):
#
#     vec = use_wikipedia2vec(w2v,sequence)
#
#     return vec

def sequence_to_tensorsf(w2v,max_q_len, sequence):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.FloatTensor(max_q_len,300).zero_()
    seq_word = ""

    # sen_rep[0] = torch.LongTensor(w2v[dictionary.start])
    # seq_word += " " + dictionary.start
    counter = 0
    for j in range(0,(max_q_len)):
        try:
            sen_rep[j] += torch.FloatTensor(w2v[sequence[j]])
            seq_word += " "+ sequence[j]
            counter += 1
        except:
            sen_rep[j] += torch.FloatTensor([0.0 for i in range(300)])
            seq_word += " "+"unk"

    # sen_rep = torch.div(sen_rep, len(sequence))
    # sen_rep[len(sequence)+1] = torch.LongTensor(w2v[dictionary.stop])
    # seq_word += " " + dictionary.stop
    return sen_rep, seq_word, counter


def sequence_to_tensors(w2v,max_q_len, sequence):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.FloatTensor(max_q_len,300).zero_()
    seq_word = ""


    # sen_rep[0] = torch.LongTensor(w2v[dictionary.start])
    # seq_word += " " + dictionary.start
    counter = 0
    for j in range(0,(max_q_len)):
        try:
            #sen_rep[j] += torch.FloatTensor(w2v.wv.word_vec(sequence[j]))
            sen_rep[j] += torch.FloatTensor(w2v[sequence[j]])
            seq_word += " "+ sequence[j]
            counter += 1
        except:
            sen_rep[j] += torch.FloatTensor([0.0 for i in range(300)])
            seq_word += " "+"unk"
            continue

    # sen_rep = torch.div(sen_rep, len(sequence))
    # sen_rep[len(sequence)+1] = torch.LongTensor(w2v[dictionary.stop])
    # seq_word += " " + dictionary.stop
    return sen_rep, seq_word, counter

def time_to_tensors(sequence):
    """Convert a sequence of words to a tensor of word indices."""
    # sen_rep = torch.FloatTensor(300).zero_()
    sequence = torch.from_numpy(np.array(sequence))
    #sequence = sequence.repeat(300)

    return sequence

def char_sequence_to_embedding_tensors(sequence, max_sent_length, dictionary):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.FloatTensor(max_sent_length, len(dictionary)).zero_()
    seq_word = ""

    #sen_rep[0][dictionary.word2idx[dictionary.start]] = 1
    # seq_word += " " + dictionary.start
    for i in range(0,len(sequence)):
        if dictionary.contains(sequence[i]):
            sen_rep[i][dictionary.word2idx[sequence[i]]] = 1.0
            seq_word += " "+ sequence[i]
        else:
            sen_rep[i][dictionary.word2idx[dictionary.unk_token] ]= 1.0
            seq_word += " "+ dictionary.unk_token

    return sen_rep, seq_word


def get_wiki_vectors(batch):

    vec_list1, vec_list2 = [], []
    for v1,v2 in batch:
        vec_list1.append(v1)
        vec_list2.append(v2)

    return vec_list1, vec_list2

def batch_to_tensor_fasttext(batch, w2v, w2v1):
    batch_queries = []
    batch_sequneces_length = []
    for sequence in batch:
        for query in sequence:
            query = list(query.split(" "))
            batch_sequneces_length.append(len(query))
            batch_queries.append(query)
    # batch_queries = np.array(batch_queries)
    # batch_queries = batch_queries.reshape(len(batch),2)
    max_q_len = 0
    # char_max_q_len = 0
    for query in batch_queries:
        if max_q_len < len(query):
            max_q_len = len(query)
    #     if char_max_q_len < len(list(" ".join(query))):
    #         char_max_q_len = len(list(" ".join(query)))
    # if max_q_len < 5:
    #     max_q_len = 5

    q1 = torch.FloatTensor(len(batch), max_q_len, (300))
    q1_lenght = q_length = np.zeros(len(batch), dtype=np.int)

    q2 = torch.FloatTensor(len(batch), max_q_len, (300))
    q2_lenght = q_length = np.zeros(len(batch), dtype=np.int)
    # char_query_tensor = torch.FloatTensor(len(batch_queries), char_max_q_len, len(char_dictionary))
    t1_tensor =  torch.FloatTensor(len(batch),1)
    session_id_tensor = torch.FloatTensor(len(batch), 1)
    # t2_tensor =  torch.FloatTensor(len(batch_t2), max(batch_sequneces_length))

    q_length = np.zeros(len(batch_queries), dtype=np.int)
    # char_q_length = np.zeros(len(batch_queries), dtype=np.int)
    index_q1 = 0
    index_q2 = 0
    #for i in range(len(batch)):
     #   t1_tensor [i] =  time_to_tensors (batch_t1[i])
      #  session_id_tensor [i] =  time_to_tensors (batch_t1[i])

        # t2_tensor [i] =  time_to_tensors (batch_t2[i], max(batch_sequneces_length))

    for i in range(len(batch)*2):
        q = batch_queries[i]
        # char_q1 = list(" ".join(q1))

        q_length[i] = len(q1)
        # char_q_length[i] = len(list(" ".join(q1)))

        q1_terms = q

        if i % 2 == 0:
            q1[index_q1], seq_input, count_word = sequence_to_tensorsf(w2v,max_q_len, q1_terms)
            # if count_word < 1:
            #     count_word = 1
            q1_lenght[index_q1] = count_word
            index_q1 += 1
        else:
            q2[index_q2], seq_input, count_word = sequence_to_tensorsf(w2v, max_q_len, q1_terms)
            # if count_word < 1:
            #     count_word = 1
            q2_lenght[index_q2] = count_word
            index_q2 += 1

        # char_query_tensor[i], char_seq_input = char_sequence_to_embedding_tensors(char_q1, char_max_q_len, char_dictionary)

    # q1_q2_t = torch.cat((query_tensor.view(2,len(batch),300), t1_tensor), dim=0)

    return q1,q2, q1_lenght, q2_lenght

# def wiki2tensor(w2v, batch, batch_t1):
#     batch_queries = []
#     batch_sequneces_length = []
#     for sequence in batch:
#         for query in sequence:
#             #query = list(query.split(" "))
#             batch_sequneces_length.append(len(query))
#             batch_queries.append(query)
#
#     max_q_len = 0
#     # char_max_q_len = 0
#     for query in batch_queries:
#         if max_q_len < len(query):
#             max_q_len = len(query)
#     #     if char_max_q_len < len(list(" ".join(query))):
#     #         char_max_q_len = len(list(" ".join(query)))
#     # if max_q_len < 5:
#     #     max_q_len = 5
#
#     q1 = torch.FloatTensor(len(batch), max_q_len, (300))
#     q1_lenght = q_length = np.zeros(len(batch), dtype=np.int)
#
#     q2 = torch.FloatTensor(len(batch), max_q_len, (300))
#     q2_lenght = q_length = np.zeros(len(batch), dtype=np.int)
#     # char_query_tensor = torch.FloatTensor(len(batch_queries), char_max_q_len, len(char_dictionary))
#     t1_tensor =  torch.FloatTensor(len(batch),1)
#     session_id_tensor = torch.FloatTensor(len(batch), 1)
#     # t2_tensor =  torch.FloatTensor(len(batch_t2), max(batch_sequneces_length))
#
#     q_length = np.zeros(len(batch_queries), dtype=np.int)
#     # char_q_length = np.zeros(len(batch_queries), dtype=np.int)
#     index_q1 = 0
#     index_q2 = 0
#     for i in range(len(batch)):
#         t1_tensor [i] =  time_to_tensors (batch_t1[i])
#         session_id_tensor [i] =  time_to_tensors (batch_t1[i])
#
#         # t2_tensor [i] =  time_to_tensors (batch_t2[i], max(batch_sequneces_length))
#
#     for i in range(len(batch)*2):
#         q = batch_queries[i]
#         # char_q1 = list(" ".join(q1))
#
#         q_length[i] = len(q1)
#         # char_q_length[i] = len(list(" ".join(q1)))
#
#         q1_terms = q
#
#         if i % 2 == 0:
#             for j in range(0,len(w2v[q1_terms])):
#                 q1[i][j] = torch.FloatTensor(samples[0][i][j])
#
#             q1[index_q1], seq_input = sequence_to_tensors(w2v,max_q_len, q1_terms)
#             q1_lenght[index_q1] = len(q1_terms)
#             index_q1 += 1
#         else:
#             q2[index_q2], seq_input = sequence_to_tensors(w2v, max_q_len, q1_terms)
#             q2_lenght[index_q2] = len(q1_terms)
#             index_q2 += 1
#
#         # char_query_tensor[i], char_seq_input = char_sequence_to_embedding_tensors(char_q1, char_max_q_len, char_dictionary)
#
#     # q1_q2_t = torch.cat((query_tensor.view(2,len(batch),300), t1_tensor), dim=0)
#
#     return q1,q2, session_id_tensor, q1_lenght, q2_lenght


def wiki2tensor(samples, batch_t1, WikiDict):
    max_list = []
    batch = len(batch_t1)
    #vec_list1, vec_list2 = get_wiki_vectors(samples)


    for i in range(batch):
        wiki_arr = WikiDict[samples[i][0]]
        max_list.append(len(wiki_arr))
        wiki_arr = WikiDict[samples[i][1]]
        max_list.append(len(wiki_arr))

    max_len = max(max_list)

    # if max_len < 5:
    #    max_len = 5

    q1 = torch.zeros(batch, max_len, 300, dtype=torch.float32)
    q2 = torch.zeros(batch, max_len, 300, dtype=torch.float32)
    q1_lenght  = np.zeros(batch, dtype=np.float32)
    q2_lenght  = np.zeros(batch, dtype=np.float32)


    for i in range(batch):

        wiki_arr = WikiDict[samples[i][0]]
        q1_lenght[i] = len(wiki_arr)
        q1[i][0:len(wiki_arr)] = torch.FloatTensor(np.array(wiki_arr))

        wiki_arr = WikiDict[samples[i][1]]
        q2_lenght[i] = len(wiki_arr)
        q2[i][0:len(wiki_arr)] = torch.FloatTensor(np.array(wiki_arr))


       # else:

        #if len(samples[0][i]) == 0:
         #   q1_lenght[i] = 1


        #else:

        #if len(samples[1][i]) == 0:
        #    q2_lenght[i] = 1


        if q1_lenght[i] == 0 or q2_lenght[i] == 0:
            print("sıfır boyutlu ")


    session_id_tensor = torch.FloatTensor(batch, 1)

    for i in range(batch):
        session_id_tensor [i] =  time_to_tensors (batch_t1[i])


    return q1,q2, session_id_tensor, q1_lenght, q2_lenght



def time_normalize(day_data):
    t1,t2 = [],[]
    t1.append(0)
    for i in range(len(day_data) - 1):
        t1.append(diff_time(day_data[i+1], day_data[i]))
        t2.append(diff_time(day_data[i+1 ], day_data[i ]))
    t2.append(0)
    # t1= time_to_normal_distribution(np.asarray(t1))
    # t2 = time_to_normal_distribution(np.asarray(t2))
    return t2

'''
import random
def reshaped_data1111(directory,m,n):
    assert os.path.exists(directory)
    df = pd.read_csv(directory, sep='\t',encoding='utf-8', index_col=[0])
    df2 = df[1:]
    df3 = df2.groupby(['UserID'], sort=True, as_index=False).count()
    df_filtered = df3[df3['Query'] >= 0] # each sample shoul have at leath 2 query pair
    user_id_list = df_filtered["UserID"].to_numpy()
    df5 = df2.loc[df2['UserID'].isin(user_id_list)]

    query_list = df5['Query'].values.tolist()
    MissionID_list = df5['MissionID'].values.tolist()
    UserID_list = df5['UserID'].values.tolist()
    time_data = df5['TimeStamp'].values.tolist()
    t2 = time_normalize(time_data)
    unique_queries = list(set(query_list))

    samples1 = []
    pairs1 =  []
    samples2 = []
    pairs2 =  []

    pairs_id = []
    session_segment_id = []
    session_segment_counter = 0
    task_id_in_user = []
    time1_samples = []
    time2_samples = []
    outF = open("myOutFile.txt", "w")
    session_numbers = 0
    session_numbers_list = []
    for i in range(len(query_list)-1): # from first to last-1 query
        example = []
        time1_example = []
        time2_example = []
        pair = []
        u_id = UserID_list[i]
        pivot_n = 0
        pivot_m = 0
        n_index = i - 1
        m_index = i + 1
        if query_list[i] == "webunlock":
            print()
        if UserID_list[i+1] == u_id: # if there is pair, each pair has same user id

            samples1.append(unique_queries.index(query_list[i]))
            samples2.append(unique_queries.index(query_list[i+1]))

            pairs1.append(unique_queries.index(query_list[i]))
            pairs2.append(unique_queries.index(query_list[i+1]))

            if MissionID_list[i] == MissionID_list[i+1]:
                pairs_id.append(0)
            else:
                pairs_id.append(1)

            time1_samples.append(t2[i])
            time2_samples.append(t2[i])

            session_segment_id.append(session_segment_counter)
            session_numbers += 1
            #task_id_in_user.append()
        else:
            session_segment_counter += 1
            session_numbers_list.append(session_numbers)
            session_numbers = 0
    # sperate test and train###########################3
    # session_numbers_list.sort(reverse=True)
    selection_list = []
    for i in range(len(session_numbers_list)):
        if session_numbers_list[i] < 131:
            selection_list.append(i)
        print(i,session_numbers_list[i])

    test_session_ids = random.sample(selection_list, k=30)
    # for i in range(len(test_session_ids)):
    #     print(session_numbers_list[test_session_ids[i]])
    ##############################################

    samples = list(zip(samples1, samples2))
    pairs   = list(zip(pairs1, pairs2))

    time1_samples= time_to_normal_distribution(np.asarray(time1_samples))

    print("sample size: ",len(samples))
    return  samples, pairs, unique_queries, pairs_id, time1_samples, time2_samples, session_segment_id,test_session_ids
'''

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
def time_to_normal_distribution(time,mean,std,tort):
    # scaler = MinMaxScaler()
    # data = time.flatten()
    if tort == 0:
        data = - time
        mean_gap = np.mean(data)
        std_gap = np.std(data)
        data -= mean_gap
        data /= std_gap
        std = std_gap
        mean = mean_gap
    else:
        data = - time
        data -= mean
        data /= std


    # data = (((data - np.min(data) )) / (np.max(data) - np.min(data)))
    # time1_samples = time1_samples.reshape(-1, 1)
    # data = preprocessing.scale(data)
    # scaler.fit(time1_samples)
    # time1_samples = scaler.transform(time1_samples)
    # data = np.reshape(data, (-1, n + m + 1))
    return data,mean, std
import datetime

def diff_time(t2,t1):
    datetimeFormat = '%Y-%m-%d %H:%M:%S'
    diff = datetime.datetime.strptime(t2, datetimeFormat) - datetime.datetime.strptime(t1,datetimeFormat)
    a = diff.total_seconds() / 60
    return np.abs(a)

def count_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = np.prod(param.size())
    return param_dict

def initialize_out_of_vocab_words(dimension, choice='zero'):
    """Returns a vector of size dimension given a specific choice."""
    if choice == 'random':
        """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
        return np.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return np.zeros(shape=dimension)

def pad_sequence(h_n, batch_sequneces_length):

    query_tensor = torch.zeros(len(batch_sequneces_length),max(batch_sequneces_length), h_n.size(1),dtype=torch.float32)
    query_tensor = query_tensor.cuda()
    counter = 0
    for j in range(query_tensor.size(0)):
        for i in range(batch_sequneces_length[j]):
            query_tensor[j][i] = h_n[counter]
            counter += 1
    return query_tensor


def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs

# def device = torch.device("cuda:0")curacy(preds, labels, valid_pred):
#     valid_pred.append(preds)
#     return np.sum(preds == labels) / len(labels), valid_pred

def flat_accuracy(preds, labels, valid_pred,valid_labels):
    valid_pred.append(preds)
    valid_labels.append(labels)
    # return np.sum(preds == labels) / len(labels), valid_pred
    compare_order = [1 if i == j else 0 for i, j in zip(preds, labels)]
    return sum(compare_order) , len(preds), valid_pred, valid_labels

def count_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = np.prod(param.size())
    return param_dict

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def get_word2vec():
    glove2word2vec('/home/nurullah/Desktop/search-master/datasets/glove/glove.42B.300d.txt',
                   '/home/nurullah/Desktop/search-master/datasets/glove/glove.42B.300d_word2vec.txt')
    # vectors = KeyedVectors.load_word2vec_format('/home/nurullah/Desktop/search-master/datasets/glove/glove.42B.300d_word2vec.txt')
    vectors = KeyedVectors.load_word2vec_format('/media/nurullah/E/datasets/dictionary/glove.840B')
    return vectors

import numpy as np

'''
def loadGloveModel(File="/media/nurullah/E/datasets/dictionary/glove.840B"):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel
'''

import pickle
def loadGloveModel():
    with open('/media/nurullah/E/datasets/dictionary/small_dict/small_dict.txt', 'rb') as handle:
        return pickle.loads(handle.read())

import torch.nn as nn
def init_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.kaiming_uniform_(param)

    return lstm



def calculate_f1(valid_pred, valid_mission_label):
    fp = 0.0  # FalsePositive
    fn = 0.0  # FalseNegative
    tp = 0.0
    tn = 0.0
    for j in range(len(valid_pred)):
        if valid_mission_label[j] != valid_pred[j]:

            # print(q1[test_index[i]], " ", q2[test_index[i]], " ", mission_label[test_index[i]], " ", items[j])
            if valid_mission_label[j] == 0:
                fp += 1.0
            else:
                fn += 1.0

        else:
            if valid_mission_label[j] == 0:
                tn += 1.0
            else:
                tp += 1.0
    if tp + fp == 0:
        p = 0
    else:
        p = tp / (tp + fp)
    if tp + fn == 0:
        r = 0
    else:
        r = tp / (tp + fn) # 1e-10
    if p + r == 0:
        f1 = 0
    else:
        f1 = (2 * p * r) / (p + r)
    accu = (tn + tp) / (tn + tp + fn + fp + 0.00000000001)

    return f1, accu , tp, fp, fn, tn

'''
def pairwise_counts(predicted_labels, true_labels):

    tn, fn, fp, tp = 0.0, 0.0, 0.0, 0.0
    assert (len(true_labels) == len(predicted_labels))

    for i in range(0, len(true_labels) - 1):
        for j in range(i + 1, len(true_labels)):

            if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                tp += 1.0
            elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                tn += 1.0
            elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                fn += 1.0
            elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                fp += 1.0
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    pairwise_fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + fn + tn)
    print(acc)
    print(tp, fp, fn, tn)
    return pairwise_fscore,acc, tp,fp,fn,tn
'''
import random


# def reshaped_data(w2v, directory,tort,js,rs):
#
#     Query_list = df['Query'].values.tolist()
#     #Task_list = df['TaskID'].values.tolist()
#     UserID_list = df['UserID'].values.tolist()
#     #Session_list = df['SessionID'].values.tolist()
#     Mission_list = df['MissionID'].values.tolist()
#     time_list = df['TimeStamp'].values.tolist()
#
#     samples1 = []
#     samples2 = []
#     labels = []
#     session = []
#
#     samples1_n = []
#     samples2_n = []
#     labels_n = []
#     session_n = []
#
#
#     for i in range(len(Query_list)-1): # from first to last-1 query
#         #if Query_emb_dict[Query_list[i]] == None:
#           #  continue
#         for j in range(i+1,len(Query_list)): # from first to last-1 query
#             #if Query_emb_dict[Query_list[j]] == None:
#                 #continue
#             if (UserID_list[i] == UserID_list[j]) and (Query_list[i] != Query_list[j]):
#                 if Mission_list[i] == Mission_list[j]:
#                     labels.append(0)
#                     samples1.append(Query_list[i])
#                     samples2.append(Query_list[j])
#                     if diff_time(time_list[j], time_list[i]) < 26:
#                         session.append(-1)
#                     else:
#                         session.append(1)
#                 else:
#                     labels_n.append(1)
#                     samples1_n.append(Query_list[i])
#                     samples2_n.append(Query_list[j])
#                     if diff_time(time_list[j], time_list[i]) < 26:
#                         session_n.append(-1)
#                     else:
#                         session_n.append(1)


def reshaped_data_valid(directory):
    assert os.path.exists(directory)
    df = pd.read_csv(directory, sep='\t',encoding='utf-8')

    Query_list = df['Query'].values.tolist()
    Task_list = df['TaskID'].values.tolist()
    UserID_list = df['UserID'].values.tolist()
    Session_list = df['SessionID'].values.tolist()
    Mission_list = df['MissionID'].values.tolist()

    samples1 = []
    samples2 = []
    labels = []
    session = []

    samples1_n = []
    samples2_n = []
    labels_n = []
    session_n = []

    for i in range(len(Query_list)-1): # from first to last-1 query


            if Mission_list[i] == Mission_list[j]:
                labels.append(0)
                samples1.append(Query_list[i])
                samples2.append(Query_list[j])
                if str(UserID_list[i]) + str(Session_list[i]) == str(UserID_list[j]) + str(Session_list[j]):
                    session.append(1)
                else:
                    session.append(-1)
            else:
                labels_n.append(1)
                samples1_n.append(Query_list[i])
                samples2_n.append(Query_list[j])
                if str(UserID_list[i]) + str(Session_list[i]) == str(UserID_list[j]) + str(Session_list[j]):
                    session_n.append(1)
                else:
                    session_n.append(-1)

    rn = random.sample(range(0, len(samples1_n)), len(samples1))
    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1_n[i])
        s2.append(samples2_n[i])
        pi.append(labels_n[i])
        se.append(session_n[i])
    samples1_n,samples2_n,labels_n,session_n= s1,s2,pi,se


    samples1 = samples1 + samples1_n
    samples2 = samples2 + samples2_n
    labels = labels + labels_n
    session = session + session_n

    if tort == 0:
        rn = random.sample(range(0, len(samples1)), len(samples1))
        s1,s2,pi,t1,se = [],[],[],[],[]
        for i in rn:
            s1.append(samples1[i])
            s2.append(samples2[i])
            pi.append(labels[i])
            se.append(session[i])
        samples1,samples2,labels,session= s1,s2,pi,se


    samples = list(zip(samples1, samples2))
    print("sample size: ",len(samples))
    return  samples, labels, session


def reshaped_combine_seperate_gayo():

    df1 = pd.read_csv("/media/nurullah/E/Dropbox/tez/conference - tasking/sperate_data_train_test/trainset_gayo_copy/train1.csv", sep='\t',encoding='utf-8')
    df2 = pd.read_csv("/media/nurullah/E/Dropbox/tez/conference - tasking/sperate_data_train_test/trainset_gayo_copy/validation1.csv",  sep='\t',encoding='utf-8')
    frames = [df1, df2]
    df = pd.concat(frames)

    Query_list = df['Query'].values.tolist()
    UserID_list = df['UserID'].values.tolist()
    time_list = df['TimeStamp'].values.tolist()
    Mission_list = df['MissionID'].values.tolist()

    samples1 = []
    samples2 = []
    labels = []
    session = []

    samples1_n = []
    samples2_n = []
    labels_n = []
    session_n = []

    for i in range(len(Query_list)-1): # from first to last-1 query
        for j in range(i+1,len(Query_list)): # from first to last-1 query
            if UserID_list[i] == UserID_list[j]:
                if Mission_list[i] == Mission_list[j]:
                    labels.append(0)
                    samples1.append(Query_list[i])
                    samples2.append(Query_list[j])
                    if diff_time(time_list[i+1], time_list[i]) < 26:
                        session.append(1)
                    else:
                        session.append(-1)
                else:
                    labels_n.append(1)
                    samples1_n.append(Query_list[i])
                    samples2_n.append(Query_list[j])
                    if diff_time(time_list[i+1], time_list[i]) < 26:
                        session_n.append(1)
                    else:
                        session_n.append(-1)




    rn = random.sample(range(0, len(samples1_n)), len(samples1))
    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1_n[i])
        s2.append(samples2_n[i])
        pi.append(labels_n[i])
        se.append(session_n[i])
    samples1_n,samples2_n,labels_n,session_n= s1,s2,pi,se


    samples1 = samples1 + samples1_n
    samples2 = samples2 + samples2_n
    labels = labels + labels_n
    session = session + session_n

    rn = random.sample(range(0, len(samples1)), len(samples1))
    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1[i])
        s2.append(samples2[i])
        pi.append(labels[i])
        se.append(session[i])
    samples1,samples2,labels,session= s1,s2,pi,se

    return  samples1, samples2, labels, session


def reshaped_combine_seperate(args, df, w2v, bs,Query_emb_dict, tort):

    Query_list = df['Query'].values.tolist()
    #Task_list = df['TaskID'].values.tolist()
    UserID_list = df['UserID'].values.tolist()
    #Session_list = df['SessionID'].values.tolist()
    Mission_list = df['MissionID'].values.tolist()
    time_list = df['TimeStamp'].values.tolist()

    query_list, labels, userid, time = [],[],[],[]

    for i in range(len(Query_list)-1):
        if Query_list[i] != Query_list[i+1]:
            query_list.append(Query_list[i])
            labels.append(Mission_list[i])
            userid.append(UserID_list[i])
            time.append(time_list[i])

    if Query_list[-1] != Query_list[-2]:
        query_list.append(Query_list[-1])
        labels.append(Mission_list[-1])
        userid.append(UserID_list[-1])
        time.append(time_list[-1])

    Query_list = query_list
    UserID_list = userid
    Mission_list = labels
    time_list = time




    samples1 = []
    samples2 = []
    labels = []
    session = []

    samples1_n = []
    samples2_n = []
    labels_n = []
    session_n = []


    for i in range(len(Query_list)-1): # from first to last-1 query
        #if Query_emb_dict[Query_list[i]] == None:
          #  continue
        for j in range(i+1,len(Query_list)): # from first to last-1 query
            #if Query_emb_dict[Query_list[j]] == None:
                #continue
            if (UserID_list[i] != UserID_list[j]):
                break
            #if(Query_list[i] != Query_list[j]):
            if Mission_list[i] == Mission_list[j]:
                labels.append(1)
                samples1.append(Query_list[i])
                samples2.append(Query_list[j])
                if diff_time(time_list[j], time_list[i]) < 26:
                    session.append(-1)
                else:
                    session.append(1)
            else:
                labels_n.append(0)
                samples1_n.append(Query_list[i])
                samples2_n.append(Query_list[j])
                if diff_time(time_list[j], time_list[i]) < 26:
                    session_n.append(-1)
                else:
                    session_n.append(1)



    dfx1 = pd.DataFrame(list(zip(samples1, samples2, labels, session)), columns =['s1', 's2','la','se'])
    dfx1 = dfx1.drop_duplicates(subset = ['s1', 's2'],keep = 'last').reset_index(drop = True)
    m = pd.DataFrame(np.sort(dfx1[['s1','s2']], axis=1), index=dfx1.index).duplicated()
    dfx1 = dfx1[~m]
    samples1, samples2, labels, session = dfx1["s1"].tolist(), dfx1["s2"].tolist(), dfx1["la"].tolist(), dfx1["se"].tolist()

    dfxx1 = pd.DataFrame(list(zip(samples1_n, samples2_n, labels_n, session_n)), columns =['s1', 's2','la','se'])
    dfxx1 = dfxx1.drop_duplicates(subset = ['s1', 's2'],keep = 'last').reset_index(drop = True)
    m = pd.DataFrame(np.sort(dfxx1[['s1','s2']], axis=1), index=dfxx1.index).duplicated()### bu satır ve alt satır, çapraz olarak benzer olanları siler (x y , y x) den birini siler
    dfxx1 = dfxx1[~m]
    samples1_n, samples2_n, labels_n, session_n = dfxx1["s1"].tolist(), dfxx1["s2"].tolist(), dfxx1["la"].tolist(), dfxx1["se"].tolist()



    if tort == False:
        rn = 0
        if len(samples1) < len(samples1_n):
            rn = random.sample(range(0, len(samples1_n)), len(samples1))
            print("equal")
        if rn == 0:
            rn = random.sample(range(0, len(samples1_n)), len(samples1_n))

    else:
        rn = 0
        if len(samples1) < len(samples1_n):
            for b in range(1,11):
                if len(samples1_n) >  len(samples1)*(11-b):
                    rn = random.sample(range(0, len(samples1_n)), len(samples1)*(11-b))
                    print(str(11-b)+" katı")
                    break
        if rn == 0:
            rn = random.sample(range(0, len(samples1_n)), len(samples1_n))

    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1_n[i])
        s2.append(samples2_n[i])
        pi.append(labels_n[i])
        se.append(session_n[i])
    samples1_n,samples2_n,labels_n,session_n= s1,s2,pi,se



    rn = random.sample(range(0, len(samples1)), len(samples1))
    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1[i])
        s2.append(samples2[i])
        pi.append(labels[i])
        se.append(session[i])
    samples1,samples2,labels,session= s1,s2,pi,se


    samples1 = samples1 + samples1_n
    samples2 = samples2 + samples2_n
    labels = labels + labels_n
    session = session + session_n

    rn = random.sample(range(0, len(samples1)), len(samples1))
    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1[i])
        s2.append(samples2[i])
        pi.append(labels[i])
        se.append(session[i])


    samples1,samples2,labels,session= s1,s2,pi,se

    dfx = pd.DataFrame(list(zip(samples1, samples2, labels)),
               columns =['q1', 'q2', 'la'])
    dfx.to_csv("data.csv")

    '''
    rn = random.sample(range(0, len(samples1)), int(len(samples1)/10))
    v_sample1 = list(np.array(samples1)[rn])
    v_sample2 = list(np.array(samples2)[rn])
    v_labels = list(np.array(labels)[rn])
    v_session = list(np.array(session)[rn])

    for index in sorted(rn, reverse=True):
        del samples1[index]
        del samples2[index]
        del labels[index]
        del session[index]
    '''
    print("sample size :", len(samples1))
    #t_v1, t_v2, label,  session,   samples  = WIKI3.data_loader (w2v,bs, list(zip(samples1, samples2)), labels, session,  Query_emb_dict)
    #v_v1, v_v2, v_label,v_session, v_sam = WIKI3.data_loader (w2v,bs, list(zip(v_sample1, v_sample2)), v_labels, v_session,    Query_emb_dict)

    return list(zip(samples1, samples2)),labels,session #, list(zip(v_v1, v_v2)),v_label,v_session, v_sam


def reshaped_combine_seperate_seq(args, df, w2v, bs,Query_emb_dict):



    Query_list = df['Query'].values.tolist()
    #Task_list = df['TaskID'].values.tolist()
    UserID_list = df['UserID'].values.tolist()
    #Session_list = df['SessionID'].values.tolist()
    Mission_list = df['MissionID'].values.tolist()
    time_list = df['TimeStamp'].values.tolist()

    samples1 = []
    samples2 = []
    labels = []
    session = []

    samples1_n = []
    samples2_n = []
    labels_n = []
    session_n = []


    for i in range(len(Query_list)-1): # from first to last-1 query
    #for i, item in enumerate(tqdm(Query_list)):
        #if Query_emb_dict[Query_list[i]] == None:
          #  continue
        #for j in range(i+1,len(Query_list)): # from first to last-1 query
            #if Query_emb_dict[Query_list[j]] == None:
                #continue
        j = i + 1
        if (UserID_list[i] == UserID_list[j]) and (Query_list[i] != Query_list[j]):
            if Mission_list[i] == Mission_list[j]:
                labels.append(0)
                samples1.append(Query_list[i])
                samples2.append(Query_list[j])
                if diff_time(time_list[i+1], time_list[i]) < 26:
                    session.append(-1)
                else:
                    session.append(1)
            else:
                labels_n.append(1)
                samples1_n.append(Query_list[i])
                samples2_n.append(Query_list[j])
                if diff_time(time_list[i+1], time_list[i]) < 26:
                    session_n.append(-1)
                else:
                    session_n.append(1)
    print("ilk")
    rn = 0
    if len(samples1) < len(samples1_n):
        for b in range(1,11):
            if len(samples1_n) >  len(samples1)*(11-b):
                rn = random.sample(range(0, len(samples1_n)), len(samples1)*(11-b))
                print(str(11-b)+" katı")
                break
    if rn == 0:
        rn = random.sample(range(0, len(samples1_n)), len(samples1_n))

    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1_n[i])
        s2.append(samples2_n[i])
        pi.append(labels_n[i])
        se.append(session_n[i])
    samples1_n,samples2_n,labels_n,session_n= s1,s2,pi,se



    rn = random.sample(range(0, len(samples1)), len(samples1))
    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1[i])
        s2.append(samples2[i])
        pi.append(labels[i])
        se.append(session[i])
    samples1,samples2,labels,session= s1,s2,pi,se


    samples1 = samples1 + samples1_n
    samples2 = samples2 + samples2_n
    labels = labels + labels_n
    session = session + session_n

    rn = random.sample(range(0, len(samples1)), len(samples1))
    s1,s2,pi,t1,se = [],[],[],[],[]
    for i in rn:
        s1.append(samples1[i])
        s2.append(samples2[i])
        pi.append(labels[i])
        se.append(session[i])
    samples1,samples2,labels,session= s1,s2,pi,se

    '''
    rn = random.sample(range(0, len(samples1)), int(len(samples1)/10))
    v_sample1 = list(np.array(samples1)[rn])
    v_sample2 = list(np.array(samples2)[rn])
    v_labels = list(np.array(labels)[rn])
    v_session = list(np.array(session)[rn])

    for index in sorted(rn, reverse=True):
        del samples1[index]
        del samples2[index]
        del labels[index]
        del session[index]
    '''
    print("sample size :", len(samples1))
    t_v1, t_v2, label,  session,   samples  = WIKI3.data_loader (w2v,bs, list(zip(samples1, samples2)), labels, session,  Query_emb_dict)
    #v_v1, v_v2, v_label,v_session, v_sam = WIKI3.data_loader (w2v,bs, list(zip(v_sample1, v_sample2)), v_labels, v_session,    Query_emb_dict)

    return list(zip(t_v1, t_v2)),label,session, samples #, list(zip(v_v1, v_v2)),v_label,v_session, v_sam


'''
from tqdm import tqdm
def reshaped_data(directory, tort):

    df1 = pd.read_csv(directory, sep='\t',encoding='utf-8')
    df2 = pd.read_csv(directory, sep='\t',encoding='utf-8')
    q1,q2, labels, userid, session = [], [], [], [], []

    for index1, row1 in tqdm(df1.iterrows(), total=df1.shape[0]):
        for index2, row2 in df2.iterrows():
            if index1 < index2:
                if str(row1["UserID"]) == str(row2["UserID"]):

                    q1.append(row1["Query"])
                    q2.append(row2["Query"])
                    userid.append(row1["UserID"])

                    if diff_time(row2["TimeStamp"],row1["TimeStamp"])  < 26:
                        session.append(1)
                    else:
                        session.append(-1)

                    if str(row1["UserID"])+str(row1["MissionID"]) == str(row2["UserID"])+str(row2["MissionID"]):
                        labels.append(0)
                    else:
                        labels.append(1)
                    break
                else:
                    break
    if tort == 0:
        rn = random.sample(range(0, len(q1)), len(q1))
        s1,s2,pi,t1,se = [],[],[],[],[]
        for i in rn:
            s1.append(q1[i])
            s2.append(q2[i])
            pi.append(labels[i])
            se.append(session[i])
        q1,q2,labels,session= s1,s2,pi,se
    else:
        print()
        #time_diff, mean, std= time_to_normal_distribution(np.asarray(time_diff),mean,std,tort)

    print("sample size: ",len(q1))
    return  list(zip(q1,q2)), labels, session
'''

def reshaped_data11(directory,m,n,unique_queries, pairs,mean,std,tort):
    assert os.path.exists(directory)
    df = pd.read_csv(directory, sep='\t',encoding='utf-8')
    df2 = df[:]
    df3 = df2.groupby(['UserID'], sort=True, as_index=False).count()
    df_filtered = df3[df3['Query'] >= 0] # each sample shoul have at leath 2 query pair
    user_id_list = df_filtered["UserID"].to_numpy()
    df5 = df2.loc[df2['UserID'].isin(user_id_list)]

    query_list = df5['Query'].values.tolist()
    MissionID_list = df5['MissionID'].values.tolist()
    UserID_list = df5['UserID'].values.tolist()
    time_data = df5['TimeStamp'].values.tolist()
    t1,t2 = time_normalize(time_data)
    new_list = list(set(query_list))
    l3 = [x for x in new_list if x not in unique_queries]
    unique_queries = unique_queries + l3
    pairs = pairs

    samples1 = []
    pairs1 = []
    samples2 = []
    samples3 = []
    pairs2 = []
    distance = []

    pairs_id = []

    time1_samples = []
    time2_samples = []
    time3_samples = []

    outF = open("myOutFile.txt", "w")
    for i in range(len(query_list) - 1):  # from first to last-1 query
        example = []
        time1_example = []
        time2_example = []
        pair = []
        u_id = UserID_list[i]
        pivot_n = 0
        pivot_m = 0
        n_index = i - 1
        m_index = i + 1
        #neg_ind = 0

        if UserID_list[i + 1] == u_id:  # if there is pair, each pair has same user id
            stop = 0

            neg_ind = i + 2
            for j in range(i + 1, len(query_list) - 1):
                s11, s12, s21, s22, p11, p12, p21, p22, p1, p2, t11, t12, t21, t22 = "", "", "", "", "", "", "", "", -1, -1, -1, -1, -1, -1

                if UserID_list[i] != UserID_list[j] or stop > 1:
                    break

                if (MissionID_list[i] == MissionID_list[j]):

                    stop = stop - 1
                    s11 = unique_queries.index(query_list[i])
                    s12 = unique_queries.index(query_list[j])

                    p11 = (unique_queries.index(query_list[i]))
                    p12 = (unique_queries.index(query_list[j]))

                    if MissionID_list[i] == MissionID_list[j]:
                        p1 = (0)

                    else:
                        p1 = (1)

                    # distance.append(j)
                    t11 = diff_time(time_data[j], time_data[i])
                    #t11 = t2[i]/(j-i)
                    #t12 = t2[i]


                for k in range(neg_ind, len(query_list) - 1):
                    if UserID_list[i] != UserID_list[k]:
                        break
                    if (MissionID_list[i] != MissionID_list[k]):
                        neg_ind = k + 1
                        stop = stop - 1
                        s21 = unique_queries.index(query_list[i])
                        s22 = unique_queries.index(query_list[k])

                        p21 = (unique_queries.index(query_list[i]))
                        p22 = (unique_queries.index(query_list[k]))

                        if MissionID_list[i] == MissionID_list[k]:
                            p2 = (0)

                        else:
                            p2 = (1)

                        # distance.append(j)
                        t22 = diff_time(time_data[k], time_data[i])
                        # t22 = (diff_time(time_data[j], time_data[k]))
                        break

                if p1 == 0 and p2 == 1:
                    samples1.append(s11)
                    samples2.append(s12)
                    samples1.append(s11)
                    samples2.append(s22)
                    #samples3.append(s22)

                    pairs1.append(p11)
                    pairs2.append(p12)

                    pairs_id.append(p1)
                    pairs_id.append(p2)

                    # distance.append(j)
                    time1_samples.append(t11)
                    #time2_samples.append(t12)
                    time1_samples.append(t22)

                else:
                    break

    samples = list(zip(samples1, samples2))
    pairs   = list(zip(pairs1, pairs2))

    if tort == 0:
        time1_samples, mean, std= time_to_normal_distribution(np.asarray(time1_samples),mean,std,tort)
    else:
        time1_samples, mean, std= time_to_normal_distribution(np.asarray(time1_samples),mean,std,tort)
    return  samples, pairs, unique_queries, pairs_id, time1_samples, time1_samples, 0, mean,std






def reshaped_data_augment(directory,m,n,unique_queries, pairs,mean,std,tort):
    assert os.path.exists(directory)
    df = pd.read_csv(directory, sep='\t',encoding='utf-8')
    df2 = df[:]
    df3 = df2.groupby(['UserID'], sort=True, as_index=False).count()
    df_filtered = df3[df3['Query'] >= 0] # each sample shoul have at leath 2 query pair
    user_id_list = df_filtered["UserID"].to_numpy()
    df5 = df2.loc[df2['UserID'].isin(user_id_list)]

    query_list = df5['Query'].values.tolist()
    MissionID_list = df5['MissionID'].values.tolist()
    UserID_list = df5['UserID'].values.tolist()
    time_data = df5['TimeStamp'].values.tolist()
    t1,t2 = time_normalize(time_data)
    new_list = list(set(query_list))
    l3 = [x for x in new_list if x not in unique_queries]
    unique_queries = unique_queries + l3
    pairs = pairs

    samples1 = []
    pairs1 = []
    samples2 = []
    samples3 = []
    pairs2 = []
    distance = []

    pairs_id = []

    time1_samples = []
    time2_samples = []
    time3_samples = []

    outF = open("myOutFile.txt", "w")
    for i in range(len(query_list) - 1):  # from first to last-1 query
        example = []
        time1_example = []
        time2_example = []
        pair = []
        u_id = UserID_list[i]
        pivot_n = 0
        pivot_m = 0
        n_index = i - 1
        m_index = i + 1
        #neg_ind = 0

        if UserID_list[i + 1] == u_id:  # if there is pair, each pair has same user id
            stop = 0

            neg_ind = i + 2
            for j in range(i + 1, len(query_list) - 1):
                s11, s12, s21, s22, p11, p12, p21, p22, p1, p2, t11, t12, t21, t22 = "", "", "", "", "", "", "", "", -1, -1, -1, -1, -1, -1

                if UserID_list[i] != UserID_list[j] or stop > 1:
                    break

                if (MissionID_list[i] == MissionID_list[j]):

                    stop = stop - 1
                    s11 = unique_queries.index(query_list[i])
                    s12 = unique_queries.index(query_list[j])

                    p11 = (unique_queries.index(query_list[i]))
                    p12 = (unique_queries.index(query_list[j]))

                    if MissionID_list[i] == MissionID_list[j]:
                        p1 = (0)

                    else:
                        p1 = (1)

                    # distance.append(j)
                    t11 = diff_time(time_data[j], time_data[i])/(j-i)
                    t12 = t2[i]


                for k in range(neg_ind, len(query_list) - 1):
                    if UserID_list[i] != UserID_list[k]:
                        break
                    if (MissionID_list[i] != MissionID_list[k]):
                        neg_ind = k + 1
                        stop = stop - 1
                        s21 = unique_queries.index(query_list[i])
                        s22 = unique_queries.index(query_list[k])

                        p21 = (unique_queries.index(query_list[i]))
                        p22 = (unique_queries.index(query_list[k]))

                        if MissionID_list[i] == MissionID_list[k]:
                            p2 = (0)

                        else:
                            p2 = (1)

                        # distance.append(j)
                        t21 = diff_time(time_data[k], time_data[i])/(k-i)
                        # t22 = (diff_time(time_data[j], time_data[k]))
                        break

                if p1 == 0 and p2 == 1:
                    samples1.append(s11)
                    samples2.append(s12)
                    samples1.append(s11)
                    samples2.append(s22)
                    #samples3.append(s22)

                    pairs1.append(p11)
                    pairs2.append(p12)

                    pairs_id.append(p1)
                    pairs_id.append(p2)

                    # distance.append(j)
                    time1_samples.append(t11)
                    #time2_samples.append(t12)
                    time1_samples.append(t21)

                elif p1 == 0:
                    samples1.append(s11)
                    samples2.append(s12)

                    pairs1.append(p11)
                    pairs2.append(p12)

                    pairs_id.append(p1)


                    # distance.append(j)
                    time1_samples.append(t11)

                elif p2 == 1:

                    samples1.append(s21)
                    samples2.append(s22)
                    #samples3.append(s22)

                    pairs1.append(p11)
                    pairs2.append(p12)


                    pairs_id.append(p2)

                    # distance.append(j)

                    #time2_samples.append(t12)
                    time1_samples.append(t21)
                else:
                    break

    samples = list(zip(samples1, samples2))
    pairs   = list(zip(pairs1, pairs2))

    if tort == 0:
        time1_samples, mean, std= time_to_normal_distribution(np.asarray(time1_samples),mean,std,tort)
    else:
        time1_samples, mean, std= time_to_normal_distribution(np.asarray(time1_samples),mean,std,tort)
    return  samples, pairs, unique_queries, pairs_id, time1_samples, time1_samples, 0, mean,std




def generate_train_valid( samples, pairs, pairs_labels, time1_samples, time2_samples, rn, cross_index, session_segment_id,test_session_ids):

    start_index = int(len(rn) / 10) * cross_index
    stop_index = start_index + int(len(rn) / 10)

    train_samples, train_time1_samples, train_labels, train_time2_samples = [],[], [],[]
    test_samples, test_time1_samples, test_labels, test_time2_samples     = [], [],[],[]
    test_segment_id = []
    for i in range(len(samples)):
        if session_segment_id[i] in test_session_ids:

            test_samples.append(samples[i])
            test_time1_samples.append(time1_samples[i])
            test_labels.append(pairs_labels[i])
            test_time2_samples.append(time2_samples[i])
            test_segment_id.append(session_segment_id[i])

        else:
            train_samples.append(samples[i])
            train_time1_samples.append(time1_samples[i])
            train_labels.append(pairs_labels[i])
            train_time2_samples.append(time2_samples[i])

    # return  test_q11 , test_q1_time, test_q1_mislabel , test_q12 , test_q2_time , test_q2_mislabel, \
    #         test_mission_label , test_interval_time ,test_user_id, test_rand_numbers, \
    #         train_q11, train_q1_time, train_q1_mislabel , train_q12 , train_q2_time, train_q2_mislabel,\
    #         train_mission_label, train_interval_time, train_user_id, train_rand_numbers
    return  train_samples, train_time1_samples, train_labels, train_time2_samples, \
            test_samples,  test_time1_samples,  test_labels,  test_time2_samples,test_segment_id

def query2vec(query,w2v):
    re_parse_query = []
    query = query.strip()
    quer_length = len(query.split())
    i = 0
    while i < quer_length:
        flag = False
        for j in range(0, quer_length - i):
            query_parse = ' '.join(query.split()[i:quer_length - j])
            query_parse = query_parse.replace(" ", "-")
            if query_parse in w2v:  # if whole query has a vector
                re_parse_query.append(query_parse)
                i = quer_length - j
                flag = True
                break

        if not flag:
            query_parse = ' '.join(query.split()[i:i + 1])
            part_of_query(query_parse,w2v,re_parse_query)
            i += 1
    return re_parse_query
import re
def part_of_query(query,w2v, re_parse_query):

    for token in query.split():
        if token in w2v:
            re_parse_query.append(token)
        else:  # unk or include number
            token = re.sub(r'([^\n\t\w]|_)+', ' ', token)  # parse according to .,: etc.
            for it in token.split():
                if it in w2v:
                    re_parse_query.append(it)
                else:
                    it2 = re.split(r'(\d+)', it)

                    for item in it2:
                        if "" != item and " " != item:
                            if item in w2v:

                                re_parse_query.append(item)
                            else:
                                re_parse_query.append(item)


def prune_graph(G, threshold):

    weak_edges = []
    for u, v, weight in G.edges.data('weight'):
      if weight < threshold or np.isnan(weight):
        weak_edges.append((u, v))
    #print(G.number_of_edges())
    G.remove_edges_from(weak_edges)
    #print(G.number_of_edges())
    return G


def label_graph(G, labels):
    cluster = 0
    predicted_labels = np.zeros(len(labels))
    say = 0
    for component in nx.connected_components(G):
      for idx in component:
        predicted_labels[idx] = cluster
      cluster += 1
    predicted_labels = np.asarray(predicted_labels, dtype=np.int32)
    return predicted_labels


def pairwise_counts1(true_labels, predicted_labels):
    tn, fn, fp, tp = 0.0, 0.0, 0.0, 0.0

    assert (len(true_labels) == len(predicted_labels))

    for i in range(0, len(true_labels) - 1):
        for j in range(i + 1, len(true_labels)):

            if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                tp += 1.0
            elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                tn += 1.0
            elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                fn += 1.0
            elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                fp += 1.0
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    pairwise_fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    # print("precision", precision)

    return pairwise_fscore
def pairwise_counts(true_labels, predicted_labels,user_list):

    tn, fn, fp, tp = 0.0, 0.0, 0.0, 0.0
    assert (len(true_labels) == len(predicted_labels))

    for i in range(0, len(true_labels) - 1):
        for j in range(i + 1, len(true_labels)):
            if user_list[i] != user_list[j]:
                break
            if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                tp += 1.0
            elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                tn += 1.0
            elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
                fn += 1.0
            elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
                fp += 1.0
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    pairwise_fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + fn + tn)
    #print(acc)
    #print(tp, fp, fn, tn)
    return pairwise_fscore

def cal_f1(pre, gro):
    tags = set(gro)
    predicted = set(pre)

    tp = len(tags & predicted)
    fp = len(predicted) - tp
    fn = len(tags) - tp
    #     print(tp,fp,fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)


from collections import Counter
def lucc_f11(target, pre):
    f1_list= []
    start = 0
    f1_sum = 0
    sum_weight = 0

    ta1 = pd.Series(range(len(target))).groupby(target, sort=False).apply(list).tolist()
    pr1 = pd.Series(range(len(pre))).groupby(pre, sort=False).apply(list).tolist()
    for pr_l in pr1:
        biggest = 0
        for ta_l in ta1:
            f1 = cal_f1(ta_l, pr_l)

            if biggest < f1:
                biggest = f1
        #         print(tj,qk)
        f1_sum += biggest * len(pr_l)
        sum_weight += len(pr_l)


        #start = start +  user_counter[item]

    if sum_weight == 0:
        return 0
    else:
        return f1_sum / sum_weight
        f1_list.append(f1_sum / sum_weight)
def lucc_f1(target, pre, user):
    user_counter = Counter(user)
    f1_list= []
    start = 0
    f1_sum = 0
    sum_weight = 0
    for item in user_counter:

        ta = target[start: start + user_counter[item] ]
        pr = pre   [start: start + user_counter[item] ]

        ta1 = pd.Series(range(len(ta))).groupby(ta, sort=False).apply(list).tolist()
        pr1 = pd.Series(range(len(pr))).groupby(pr, sort=False).apply(list).tolist()
        for pr_l in pr1:
            biggest = 0
            for ta_l in ta1:
                f1 = cal_f1(ta_l, pr_l)

                if biggest < f1:
                    biggest = f1
            #         print(tj,qk)
            f1_sum += biggest * len(pr_l)
            sum_weight += len(pr_l)


        start = start +  user_counter[item]

    if sum_weight == 0:
        print(0)
        return 0
    else:
        return f1_sum / sum_weight
        f1_list.append(f1_sum / sum_weight)

import networkx as nx
from tqdm import tqdm

def end_to_end_cat_sim(query_list, labels,session,tr,js,rs, w2v_fast, Query_emb_dict, userid):


    alphas = [alpha for alpha in np.arange(0.0, 1.01, 0.1)]
    cat_sim, cos_sim = [],[]
    counter = 0
    for i, item in enumerate(tqdm(query_list)):
        #if Query_emb_dict[query_list[i]] == None:
            #continue
        for j in range(i+1, len(query_list)):
            #if Query_emb_dict[query_list[j]] == None:
                #continue
            if (userid[j] == userid[i]):
                se = -1 if session[i] == session[j] else 1
                #t_1= diff_time(times[j], times[i])
                #session = 1 if t_1 < 26  else -1
                #print(query_list[i],"  ", query_list[j],"  ",labels[i],"  ",labels[j])
                v1, v2 = Query_emb_dict[query_list[i]], Query_emb_dict[query_list[j]]

                sim, cos = tr.one_shot(v1, v2, se, [(query_list[i],query_list[j])], w2v_fast)
                cat_sim.append( WIKI3.get_cat_info(query_list[i],query_list[j]))

                #print(sim)
                sim = 1- sim[0]
                #if cos[0][0] < 0:
                #    cos[0][0] = - cos[0][0]
                cos = 1 - cos[0][0]

                cos = (cos + sim)/2
                cos_sim.append (cos)
                counter += 1

    lucc_list = []
    flist = []
    for alpha in alphas:
        G=nx.Graph()

        counter = 0
        for i, item in enumerate(tqdm(query_list)):
            for j in range(i+1, len(query_list)):
                total_sim = alpha * cos_sim[counter] + (1- alpha) * cat_sim[counter]
                G.add_edge(i,j,weight=total_sim)
                counter += 1

        for i in range(0,11):

            G1 = prune_graph(G, i*0.1)
            pre = label_graph(G1,labels)

            f1 = pairwise_counts(labels, pre,userid)
            print("alpha %d threshold %d", alpha,i)
            print("f1: ", f1)
            flist.append(f1)

            f1 = lucc_f1(labels , pre)
            print("---f1: ", f1)
            lucc_list.append(f1)
    print("lucc : %d, flist :%d", max(lucc_list),max(flist))
def end_to_end_FAST(query_list, labels,session,tr,w2v,js,rs):

    #query_list = query_list [:10]
    Query_emb_dict = {}
    for i, query in enumerate(tqdm(query_list)):
        Query_emb_dict[query] = WIKI3.use_wikipedia2vec(w2v, query, js,rs)


    for ii in range(0, 11):
        G=nx.Graph()
        for i, item in enumerate(tqdm(query_list)):
            for j in range(i+1, len(query_list)):
                if session[i] == session[j]:
                    se = 1 if session[i] == session[j] else -1
                    #t_1= diff_time(times[j], times[i])
                    #session = 1 if t_1 < 26  else -1
                    #print(query_list[i],"  ", query_list[j],"  ",labels[i],"  ",labels[j])
                    v1, v2 = Query_emb_dict[query_list[i]], Query_emb_dict[query_list[i]]
                    sim, cos = tr.one_shot(v1, v2, se)
                    #print(sim)
                    sim = 1- sim[0]
                    cos = 1- cos[0][0]
                    if cos > ii* 0.1:
                        G.add_edge(i,j,weight=cos)

        G2 = G.copy()
        while True:
            try:
                component = next(nx.connected_components(G))
            except:
                break
            #print(component)
            for node in component:
                G.remove_node(node)
            G1 = G.copy()
            while True:
                try:
                    component1 =  next(nx.connected_components(G1))
                except:
                    break
                for node in component1:
                    G1.remove_node(node)
                check_node = False
                for com in component:
                    for com1 in component1:
                        v1, v2 = Query_emb_dict[query_list[com]], Query_emb_dict[query_list[com1]]
                        sim, cos = tr.one_shot(v1, v2, -1) # session = -1
                        sim = 1- sim[0]
                        cos = 1- cos[0][0]
                        if cos > ii* 0.1:
                            G2.add_edge(com,com1,weight=cos)
                            for node in component1:
                                component.add(node)
                                G.remove_node(node)
                            check_node = True
                            break
                    if check_node:
                        break


        #for i in range(0,11):

            #G1 = prune_graph(G2, i*0.1)
        pre = label_graph(G2,labels)

        f1 = pairwise_counts(labels, pre)
        print("f1: ", f1)

        f1 = lucc_f1(labels , pre)
        print("---f1: ", f1)



def end_to_end(all_userid, all_label, tr, samples_b, label,  session, w2v_fast, w2v1):

    predict = tr.validate_oneshot(samples_b, label, session, w2v_fast, w2v1)

    G=nx.Graph()
    count = 0
    for i, item in enumerate(tqdm(all_userid)):
        for j in range(i+1, len(all_userid)):

            if all_userid[i] != all_userid[j]:
                break

            if label[count] == 1:
                print(session[count], samples_b[count], label[count], predict[count])
            G.add_edge(i,j,weight=predict[count])
            count += 1


    print("count: ", count)
    cat_list = []
    sim_list = []
    for i in range(0,100):

        G1 = prune_graph(G, i*0.01)
        pre = label_graph(G1,all_label)

        f1 = pairwise_counts(all_label, pre, all_userid)
        sim_list.append(f1)
        print(round(i*0.01,3),"  ",round(f1,3))

        f1 = lucc_f1(all_label , pre, all_userid)
        cat_list.append(f1)
        print(round(i*0.01,3),"  ",round(f1,3))
    print("sim : ,cat : " +str(round(max(sim_list),3)) +"\t"+ str(round(max(cat_list),3)) )










