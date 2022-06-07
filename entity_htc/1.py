import util, data, helper, train
import os, numpy, torch, sys
from model import Siamese
from torch import optim
from train import Train
import WIKI3
import random
import math
import pandas as pd
random.seed(1)
import copy
import os, sys
import pickle
#load fasttext

import wikipedia2vec
from gensim.models import Word2Vec
import wikipedia2vec
import tagme
from tqdm import tqdm
import numpy as np

def main():
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #model = Word2Vec.load("/media/nurullah/E/datasets/dictionary/en_1000_no_stem/en.model")
    #wiki2vec = wikipedia2vec.Wikipedia2Vec.load("/home/nurullah/Downloads/enwiki_20180420_lg1_300d.pkl")
    cross_validation = False

    args = util.get_args()
    batch_size = [ 32, 64, 128, 256]
    learning_rate = [1e-2, 1e-3, 1e-4, 1e-5] #,2e-3, 2e-2
    batch_size = [128] #128
    learning_rate = [1e-3] # 4e-5

    joint_ann_score = [0.15,0.2,0.23,0.25,0.27,0.3,0.35]#
    joint_ann_score = [0.23]
    relatedness_score = [0.3]# 0.9, 1.5
    #relatedness_score = [0.3]
    #w2v = fasttext.load_model('/home/nurullah/cc.en.300.bin')  #helper.loadGloveModel()
    #w2v = fasttext.load_model("/home/nurullah/Downloads/crawl-300d-2M-subword.bin")
    w2v1 = helper.loadGloveModel()
    w2v = 0
    #w2v_fastext = fasttext.load_model('/home/nurullah/cc.en.300.bin')  #helper.loadGloveModel()
    with open('/home/nurullah/Desktop/gayo_entity2dic_and_fasttext_small/fast_dict.pickle', 'rb') as f:
        w2v_fastext = pickle.load(f)

    acc_list = []
    f = open("demofile2.txt", "a")
    best_dev_acc = 0
    if cross_validation:
        for cr in range(0, 1):

            print("----------------------------------------------------------------------------------cr :", cr)
            df1 = pd.read_csv(args.train_data+str(cr)+'.csv', sep='\t',encoding='utf-8')
            df2 = pd.read_csv(args.valid_data+str(cr)+'.csv', sep='\t',encoding='utf-8')

            for js in joint_ann_score:
                with open( '/home/nurullah/Desktop/gayo_entity2vec_and_word2vec/yes_w2v/'+str(js)+'dict.pickle', 'rb') as f1:
                    Query_emb_dict1 = pickle.load(f1)
                for rs in relatedness_score:

                    for bs in batch_size:
                        t_samples,label,session      = helper.reshaped_combine_seperate(args,df1, w2v,bs, Query_emb_dict1, True)
                        v_samples,v_label,v_session  = helper.reshaped_combine_seperate(args,df2, w2v,32, Query_emb_dict1, False)

                        for lr in learning_rate:
                            print("js %s rs %s bs %s lr %s" %(js, rs, bs, lr))

                            model = Siamese(args)
                            #param_dict = helper.count_parameters(model)
                            #print('number of trainable parameters = ', numpy.sum(list(param_dict.values())))
                            model = model.cuda()
                            #model.load_state_dict(copy.deepcopy(torch.load("siamese_best_for_test.pth", torch.device("cuda:0"))))
                            tr= Train(model, bs, lr, args, w2v, Query_emb_dict1)
                            best_dev_acc, epoch_best = tr.train_epochs(best_dev_acc, t_samples,label,session, v_samples,v_label,v_session,w2v_fastext, w2v1)

                            f.write(str(cr) +" "+str(js)+"  "+str(rs)+"  " +str(bs) +" "+str(lr)+" "+str(epoch_best)+"\n")

    ################################################ TEST #####################


    else:
        js, rs = 0.23 , 0.9
        with open( '/home/nurullah/Desktop/gayo_entity2vec_and_word2vec/yes_w2v/'+str(js)+'dict.pickle', 'rb') as f:
            Query_emb_dict1 = pickle.load(f)
        model = Siamese(args)
        model = model.cuda()
        #tr= Train(model, 64, 0.002, args, w2v)

        model.load_state_dict(copy.deepcopy(torch.load("siamese_best_for_test.pth", torch.device("cuda:0"))))
        model.eval()
        tr= Train(model, 1, 2e-4, args, w2v, Query_emb_dict1)

        #v_samples,v_label,v_session, v_sam = helper.reshaped_data(w2v, args.test_data, 1, js, rs)

        #t_v1,t_v2,label,session, samples = WIKI.data_loader (w2v,[1], query_te, label_te, session_te)

        #_, _, _ ,best_dev_acc = tr.validate( v_samples,v_label,v_session, v_sam, w2v_fastext)
        print("test acc: ",best_dev_acc)

    ############################################# task extraction ########################

        query_list, labels, userid, time, t_samples, label,  session,_= helper.get_test_data(args.test_data, Query_emb_dict1)

        #t_v1,t_v2,label,session, samples = WIKI.data_loader (w2v,[1], list(zip(query_list, query_list)), labels, session )

        #label = sum(label, [])
        helper.end_to_end(query_list, labels, userid, time, tr, t_samples, label,  session, w2v_fastext, w2v1)

        #helper.end_to_end_FAST(query_list, labels, session, tr,w2v, js, rs)
        print("best_upsampling_lstm_alone_FAST_valid_ayrı_fasttext")

if __name__ == '__main__':
    main()
