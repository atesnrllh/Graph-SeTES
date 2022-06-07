import pandas as pd
from random import randrange
import random


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    return list_object

import pandas as pd
from tqdm import tqdm
def sampling(df, file_name):

    df_test = pd.DataFrame()
    df_train_valid = pd.DataFrame()


    grouped = df.groupby("Task")
    for name, group in grouped:
        test_size = int(len(group) / 5) # %20
        rn_number = random.sample(range(group.axes[0][0], group.axes[0][0] + len(group)), test_size)
        rn_number.sort()

        if test_size < 2:
            df_train_valid = pd.concat([df_train_valid,group], axis=0)
            continue

        df_test_pivot        = group.loc[rn_number]
        df_test = pd.concat([df_test,df_test_pivot], axis=0)

        df_train_valid_pivot = group.drop(rn_number)
        df_train_valid = pd.concat([df_train_valid,df_train_valid_pivot], axis=0)


    df_test.to_csv(path+"test/"+file_name, sep='\t',index=False)


    samples1, samples2, labels = [],[], []
    Query_list = df_train_valid["Query"].tolist()
    Task_list = df_train_valid["Task"].tolist()
    delete_item = []

    #delete long sentences
    delete_item = []
    for i in range(len(Query_list)):
        if len(Query_list[i].split(" ")) > 10:
            delete_item.append(i)
    delete_item.sort(reverse=True)
    for i in delete_item:
        del Query_list[i], Task_list[i]


    for i in tqdm(range (len(Query_list)-1)): # from first to last-1 query
        for j in range(i+1,len(Query_list)): # from first to last-1 query
            if Task_list[i] == Task_list[j]:
                labels.append(1)
                samples1.append(Query_list[i])
                samples2.append(Query_list[j])

                while True: # negatif sampling
                    rnd = randrange(len(Query_list))
                    if Task_list[i] != Task_list[rnd]:
                        labels.append(0)
                        samples1.append(Query_list[i])
                        samples2.append(Query_list[rnd])
                        break
                    else:
                        continue
            else:
                break

    df = pd.DataFrame(list(zip(samples1, samples2, labels)), columns =['s1', 's2', 'la'])
    rest_valid = df.sample(frac = 0.1)
    rest_train = df.drop(rest_valid.index)

    rest_train.to_csv(path+"train/"+file_name, sep='\t',index=False)
    rest_valid.to_csv(path+"validation/"+file_name, sep='\t',index=False)

    print()
    # myfile = open(path+"train/"+file_name, 'w')
    # for i in range(len(samples1)):
    #     if i not in rnd_list:
    #         myfile.write(str(samples1[i])+"\t"+str(samples2[i])+"\t"+str(labels[i])+'\n')
    #
    # myfile = open(path+"validation/"+file_name, 'w')
    # for i in rnd_list:
    #     myfile.write(str(samples1[i])+"\t"+str(samples2[i])+"\t"+str(labels[i])+'\n')


path = "/home/nurullah/Dropbox/tez/all_models/gayo/siamese and decider/upsample_fasttext_group/no_entity/9_task_mapping/datasets/super/"

# df = pd.read_csv("/media/nurullah/E/agnostic_bert/search-master1/datasets/volske/d1_session.csv", sep=',',encoding='utf-8')
# Query_list1 = df['Query'].values.tolist()
# Task_list1 = df['Task'].values.tolist()

# df = pd.read_csv("/media/nurullah/E/agnostic_bert/search-master1/datasets/volske/d2_trec.csv", sep=',',encoding='utf-8')
# Query_list2 = df['Query'].values.tolist()
# Task_list2 = df['Task'].values.tolist()
#
df = pd.read_csv("/media/nurullah/E/agnostic_bert/search-master1/datasets/volske/d3_wikihow.csv", sep=',',encoding='utf-8')
# Query_list3 = df['Query'].values.tolist()
# Task_list3 = df['Task'].values.tolist()


# sampling( df, "d1_session.csv")
sampling( df, "d3_wikihow.csv")
# sampling(Query_list1,Task_list1,Query_list2,Task_list2, "d1_session_d2_trec.csv")
#
# sampling(Query_list1,Task_list1,Query_list3,Task_list3, "d1_session_d3_wikihow.csv")
#
# sampling(Query_list2,Task_list2,Query_list3,Task_list3, "d2_trec_d3_wikihow.csv")


