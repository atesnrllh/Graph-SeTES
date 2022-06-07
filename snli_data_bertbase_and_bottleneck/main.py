import util, data, helper, train
import os, numpy, torch, sys
from model import Siamese
from torch import optim
from train import Train
import random
import math
import pandas as pd
random.seed(1)
import copy
import numpy as np


def main():
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cross_validation = True
    args = util.get_args()

    #batch_size = [ 16, 32, 64, 128, 256]
    #learning_rate = [2e-6, 2e-5,2e-4,2e-3, 2e-2]
    batch_size = [16]
    learning_rate = [2e-5]
    w2v = 0#helper.loadGloveModel()
    acc_list = []
    f = open("demofile2.txt", "a")
    best_dev_acc = 0
    if cross_validation:
        for cr in range(3, 4):
            #test_samples, pairs, unique_queries, test_labels, test_time1_samples, test_time2_samples, session_test, mean,std = helper.reshaped_data(args.test_data, args.m, args.n)
            #query1_tr, query2_tr, label_tr, tr_session = helper.read_nli_data(args.train_data+str(cr)+'.csv')
            #query1_va, query2_va, label_va, va_session = helper.reshaped_combine_seperate(args.valid_data+str(cr)+'.csv')
            query1_tr, query2_tr, label_tr, query1_va, query2_va, label_va = helper.read_nli_data()

            print("----------------------------------------------------------------------------------cr :", cr)

            for lr in learning_rate:
                for bs in batch_size:
                    print("---------------------------------------------------------------------------------lr :", lr)
                    print("----------------------------------------------------------------------------------bs :",bs)
                    total_acc = 0.0
                    model = Siamese(args)
                    #param_dict = helper.count_parameters(model)
                    #print('number of trainable parameters = ', numpy.sum(list(param_dict.values())))
                    model = model.cuda()
                    tr= Train(model, bs,lr, args, w2v, len(label_tr))
                    best_dev_acc, epoch_best = tr.train_epochs(best_dev_acc, query1_tr, query2_tr, label_tr, label_tr, query1_va, query2_va, label_va, label_tr)
                    f.write(str(lr)+"  "+str(bs)+"  "+str(epoch_best)+"\n")

    ################################################ TEST #####################
    else:
        model = Siamese(args)
        model = model.cuda()

        query1_te, query2_te, label_te = helper.read_sts_test_data()

        tr= Train(model, 1, 2e-5, args, w2v, len(label_te))

        model.load_state_dict(copy.deepcopy(torch.load("nli_bert_bottleneck.pth", torch.device("cuda:0"))))
        model.eval()
        test_result = tr.validate(query1_te, query2_te, label_te, label_te)
    #     best_dev_acc, _, _, f1 = tr.validate(query1_te, query2_te, label_te, tr_session)
    #     print("test acc: ",best_dev_acc)
    # ############################################# task extraction ########################
    #     query_list, labels, userid,times = helper.get_test_data('/home/nurullah/Desktop/data_valid_gayo/2/test.csv')
    #     helper.end_to_end(query_list, labels, userid, times, w2v,0,0,tr,model)



if __name__ == '__main__':
    main()


