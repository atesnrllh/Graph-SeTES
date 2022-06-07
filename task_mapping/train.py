import helper
import torch.nn as nn
device = "cuda:0"
import torch
import time
from transformers import get_linear_schedule_with_warmup
from transformers import  AdamW
import torch.nn.functional as F
import numpy as np
import random
#for Reproducibility
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from numpy import sqrt
from numpy import argmax

'''
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True
'''
#######
import os
class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, bs,lr,config, w2v):
        self.model = model
        self.config = config


        self.best_dev_acc = 0
        self.times_no_improvement = 0
        self.stop = False
        self.train_losses = []
        self.dev_losses = []
        self.batch_size = bs
        self.lr = lr
        self.epoch = config.epochs

        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.w2v = w2v



    def train_epochs(self, best_dev_acc, t_samples,label, v_samples,v_label,w2v_fast,w2v1):

        self.best_dev_acc = best_dev_acc
        best_model = self.model
        best_list =[]
        epoch_best = 0
        for epoch in range(self.epoch):

            print('\nTRAINING : Epoch ' + str((epoch + 1)))

            st = time.time()
            self.train(t_samples,label, w2v_fast)
            print("----%.2f----" % (time.time() - st))
            self.scheduler.step()

            print('\nVALIDATING : Epoch ' + str((epoch + 1)))
            f1, dev_valid, tp_list, dev_accuracy = self.validate(v_samples,v_label,w2v_fast)

            if dev_accuracy > epoch_best:
                epoch_best = dev_accuracy

            if self.best_dev_acc < dev_accuracy:
                self.best_dev_acc = dev_accuracy
                self.best_dev_valid = dev_valid
                self.times_no_improvement = 0
                best_model = self.model
                print("model is saved")
                torch.save(self.model.state_dict(), 'siamese_best_for_test.pth')
                best_list = tp_list
            #else:
                #self.times_no_improvement += 1
                # no improvement in validation loss for last n iterations, so stop training
                #if self.times_no_improvement == self.config.early_stop:
                #    self.stop = True
                #    break
        print("best dev acc :", self.best_dev_acc)
        return self.best_dev_acc, epoch_best

    def train(self, samples_b,labels_b,w2v_fast):


        samples_b, labels_b  = helper.batchify( samples_b, labels_b, self.batch_size)
        print('number of train batches = ', len(samples_b))

        num_batches = len(samples_b)
        self.model.train()
        total_loss = 0

        valid_preds = []
        valid_labels = []
        eval_accuracy = 0.0
        nb_eval_steps = 0
        x = np.arange(len(samples_b))
        for bn in tqdm(x, position=0, leave=True):



            q1_fast, q2_fast, q1_len, q2_len =  helper.batch_to_tensor_fasttext(samples_b[bn],w2v_fast)

            if self.config.cuda:
                q1_fast = q1_fast.cuda()
                q2_fast = q2_fast.cuda()


            self.model.zero_grad()
            out,_ = self.model(q1_fast,q1_len, q2_fast, q2_len) # q1_lenght, q2_lenght)
            listl = labels_b[bn]
            li = torch.tensor(listl, dtype=torch.float).to(device)

            loss = self.loss_fn(out, li)
            total_loss += loss.item()

            preds = (out > 0.5) * 1
            #_, preds = torch.max(out, dim=1)
            preds = preds.detach().cpu().numpy()
            label_ids = labels_b[bn]  # .to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy, len_acc, valid_preds, valid_labels = helper.flat_accuracy(preds, label_ids,
                                                                                         valid_preds, valid_labels)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_norm)
            self.optimizer.step()

        valid_preds = [item for sublist in valid_preds for item in sublist]
        valid_labels = [item for sublist in valid_labels for item in sublist]
        f1, acc , tp, fp, fn, tn = helper.calculate_f1(valid_preds, valid_labels)
        print("F1 {} Acc {} Tp: {} Fp: {} Fn: {} Tn: {}".format(f1, acc, tp, fp, fn, tn))
        print("train loss :",total_loss / num_batches)

    def validate(self, samples_b,labels_b, w2v_fast):


        samples_b, labels_b  = helper.batchify( samples_b, labels_b, self.batch_size)
        total_loss = 0
        eval_accuracy = 0
        nb_eval_steps = 0

        print('number of train batches = ', len(samples_b))

        num_batches = len(samples_b)
        valid_preds = []
        valid_labels = []
        self.model.eval()
        x = np.arange(len(samples_b))
        yhat, ytest = [], []
        for bn in tqdm(x):



            q1_fast, q2_fast, q1_len, q2_len =  helper.batch_to_tensor_fasttext(samples_b[bn],w2v_fast)

            if self.config.cuda:
                q1_fast = q1_fast.cuda()
                q2_fast = q2_fast.cuda()

            with torch.no_grad():

                out,_ = self.model(q1_fast,q1_len, q2_fast, q2_len) # q1_lenght, q2_lenght)

                listl = labels_b[bn]
                li = torch.tensor(listl, dtype=torch.float).to(device)

                loss = self.loss_fn(out, li)
                total_loss += loss.item()

                ytest.append(out.detach().cpu().numpy())
                label_ids = labels_b[bn]#.to('cpu').numpy()
                yhat.append(label_ids)

        yhat =  [item for sublist in yhat  for item in sublist]
        ytest = [item for sublist in ytest for item in sublist]
        fpr, tpr, thresholds = roc_curve(yhat, ytest, pos_label=1)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        threshold = thresholds[ix]
        y_pred = (ytest > threshold) * 1

        acc = accuracy_score(yhat, y_pred)
        f1 = f1_score(yhat, y_pred)
        tn, fp, fn, tp = confusion_matrix(yhat,y_pred).ravel()
        print("F1 {} Acc {} Tp: {} Fp: {} Fn: {} Tn: {}".format(f1, acc, tp, fp, fn, tn))

        print("valid loss :", total_loss / num_batches)

        return acc, valid_preds, [tp,fp,fn,tn], f1


    def store_cache(self, Query_list, vector):
        self.CACHE_SEP  = '|\t|\t|'
        cache_file = "/media/nurullah/E/agnostic_bert/search-master1/datasets/muse_siamese/cache_muse.csv"

        with open(cache_file, mode='a', encoding='utf-8') as data_file:
          for i in range(len(Query_list)):
            embedding = [str(number) for number in vector[i].cpu().detach().numpy()]
            embedding = ','.join(embedding)
            data_file.write(self.CACHE_SEP.join([Query_list[i].strip(), embedding]) + '\n')

    def validate_oneshot(self, Query_list, w2v_fast, w2v1):



        print('number of train batches = ', len(Query_list))
        Query_list_batch, Query_list_batch2 = helper.test_batchify(list(zip(Query_list, Query_list)), Query_list, 1)

        self.model.eval()
        x = np.arange(len(Query_list_batch))
        yhat, ytest = [], []
        for bn in tqdm(x):

            q1_fast, q2_fast, q1_len, q2_len =  helper.batch_to_tensor_fasttext( Query_list_batch[bn],w2v_fast)

            if self.config.cuda:
                q1_fast = q1_fast
                q2_fast = q2_fast

            with torch.no_grad():
                empty, vector = self.model(q1_fast,q1_len, q2_fast, q2_len) #, q1_lenght, q2_lenght)
                norm = vector.norm(p=2, dim=1, keepdim=True)
                vector = vector.div(norm)
                self.store_cache(Query_list_batch2[bn], vector)

        print("overrr")
