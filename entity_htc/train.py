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
import nltk

from scipy import spatial
'''
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True
'''
#######
class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, bs,lr,config, w2v, wiki2tensor):
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
        self.wiki2tensor = wiki2tensor
        self.threshold = -100


    def train_epochs(self, best_dev_acc, t_samples,label,session, v_samples,v_label,v_session,w2v_fast,w2v1):

        self.best_dev_acc = best_dev_acc
        best_model = self.model
        best_list =[]
        epoch_best = 0
        for epoch in range(self.epoch):

            print('\nTRAINING : Epoch ' + str((epoch + 1)))

            st = time.time()
            self.train(t_samples,label,session, w2v_fast, w2v1)
            print("----%.2f----" % (time.time() - st))
            self.scheduler.step()

            print('\nVALIDATING : Epoch ' + str((epoch + 1)))
            f1, dev_valid, tp_list, dev_accuracy = self.validate(v_samples,v_label,v_session,w2v_fast, w2v1)

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

    def train(self, samples_b,labels_b,sessions_b,w2v_fast,w2v1):


        samples_b, labels_b, sessions_b  = helper.batchify( samples_b, labels_b, sessions_b, self.batch_size)
        print('number of train batches = ', len(samples_b))

        num_batches = len(samples_b)
        self.model.train()
        total_loss = 0

        valid_preds = []
        valid_labels = []
        eval_accuracy = 0.0
        nb_eval_steps = 0
        yhat, ytest = [], []
        x = np.arange(len(samples_b))
        for bn in tqdm(x, position=0, leave=True):

            q1, q2, ses, q1_lenght, q2_lenght = helper.wiki2tensor(samples_b[bn],sessions_b[bn], self.wiki2tensor)

            q1_fast, q2_fast, q1_len, q2_len =  helper.batch_to_tensor_fasttext(samples_b[bn],w2v_fast, w2v1)

            if self.config.cuda:
                q1 = q1.cuda()
                q2 = q2.cuda()# batch_size x max_len
                ses = ses.cuda()
                q1_fast = q1_fast.cuda()
                q2_fast = q2_fast.cuda()


            self.model.zero_grad()
            out,_ = self.model(q1,q2,ses, q1_lenght, q2_lenght,q1_fast,q1_len, q2_fast, q2_len) # q1_lenght, q2_lenght)
            listl = labels_b[bn]
            li = torch.tensor(listl, dtype=torch.float).to(device)

            loss = self.loss_fn(out, li)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_norm)
            self.optimizer.step()


            ytest.append(out.detach().cpu().numpy())
            label_ids = labels_b[bn]#.to('cpu').numpy()
            yhat.append(label_ids)

        yhat =  [item for sublist in yhat  for item in sublist]
        ytest = [item for sublist in ytest for item in sublist]
        fpr, tpr, thresholds = roc_curve(yhat, ytest, pos_label=1)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        self.threshold = thresholds[ix]
        print("threshold ", self.threshold)
        y_pred = (ytest > self.threshold) * 1

        acc = accuracy_score(yhat, y_pred)
        f1 = f1_score(yhat, y_pred)
        tn, fp, fn, tp = confusion_matrix(yhat,y_pred).ravel()
        print("F1 {} Acc {} Tp: {} Fp: {} Fn: {} Tn: {}".format(f1, acc, tp, fp, fn, tn))


    def validate(self, samples_b,labels_b,sessions_b, w2v_fast, w2v1):


        samples_b, labels_b, sessions_b  = helper.batchify( samples_b, labels_b, sessions_b, self.batch_size)
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
            q1, q2, ses, q1_lenght, q2_lenght = helper.wiki2tensor(samples_b[bn], sessions_b[bn], self.wiki2tensor)

            q1_fast, q2_fast, q1_len, q2_len =  helper.batch_to_tensor_fasttext(samples_b[bn],w2v_fast, w2v1)

            if self.config.cuda:
                q1 = q1.cuda()
                q2 = q2.cuda()# batch_size x max_len
                ses = ses.cuda()
                q1_fast = q1_fast.cuda()
                q2_fast = q2_fast.cuda()

            with torch.no_grad():
                out,_ = self.model(q1, q2, ses, q1_lenght, q2_lenght,q1_fast,q1_len, q2_fast, q2_len) #, q1_lenght, q2_lenght)

                listl = labels_b[bn]
                li = torch.tensor(listl, dtype=torch.float).to(device)

                loss = self.loss_fn(out, li)
                total_loss += loss.item()

                ytest.append(out.detach().cpu().numpy())
                label_ids = labels_b[bn]#.to('cpu').numpy()
                yhat.append(label_ids)

        yhat =  [item for sublist in yhat  for item in sublist]
        ytest = [item for sublist in ytest for item in sublist]


        print("threshold ", self.threshold)
        y_pred = (ytest > self.threshold) * 1

        acc = accuracy_score(yhat, y_pred)
        f1 = f1_score(yhat, y_pred)
        tn, fp, fn, tp = confusion_matrix(yhat,y_pred).ravel()
        print("F1 {} Acc {} Tp: {} Fp: {} Fn: {} Tn: {}".format(f1, acc, tp, fp, fn, tn))

        print("valid loss :", total_loss / num_batches)

        return acc, valid_preds, [tp,fp,fn,tn], f1


    def one_shot(self, q1,q2,w2v_fast,alpha):


        edist = 1- nltk.edit_distance(q1,q2) / max(len(q1),len(q2))

        fast1 = 0
        q1 = q1.split(" ")
        lenght1 = 0
        for j in range(len(q1)):
            try:
                fast1 += w2v_fast[q1[j]]
                lenght1 += 1
            except:
                continue

        fast2 = 0
        q2 = q2.split(" ")
        lenght2 = 0
        for j in range(len(q2)):
            try:
                fast2 += w2v_fast[q2[j]]
                lenght2 += 1
            except:
                continue

        cossim = 1 - spatial.distance.cosine(fast1/lenght1, fast2/lenght2)

        sim = alpha * edist + (1-alpha) * cossim

        return sim
        #return out[0][pred]

    def validate_oneshot(self, samples_b,labels_b,sessions_b, w2v_fast, w2v1):


        # print('number of train batches = ', len(samples_b))
        samples_b, labels_b, sessions_b  = helper.test_batchify( samples_b, labels_b, sessions_b, self.batch_size)

        self.model.eval()
        x = np.arange(len(samples_b))
        yhat, ytest = [], []
        for bn in x:

            q1, q2, ses, q1_lenght, q2_lenght = helper.wiki2tensor(samples_b[bn],sessions_b[bn], self.wiki2tensor)
            q1_fast, q2_fast, q1_len, q2_len =  helper.batch_to_tensor_fasttext(samples_b[bn],w2v_fast, w2v1)

            if self.config.cuda:
                q1 = q1.cuda()
                q2 = q2.cuda()
                ses = ses.cuda()
                q1_fast = q1_fast.cuda()
                q2_fast = q2_fast.cuda()

            with torch.no_grad():
                out,_ = self.model(q1, q2, ses, q1_lenght, q2_lenght,q1_fast,q1_len, q2_fast, q2_len) #, q1_lenght, q2_lenght)

                listl = labels_b[bn]
                li = torch.tensor(listl, dtype=torch.float).to(device)

                loss = self.loss_fn(out, li)

                ytest.append(out.detach().cpu().numpy())

        ytest = [item for sublist in ytest for item in sublist]
        return ytest

