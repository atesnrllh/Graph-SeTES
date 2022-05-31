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
    def one_shot(self, q1,q2, session, sam, w2v_fast, w2v1):

        #q1, q2, t1_tensor, session_id_tensor, q1_lenght, q2_lenght =helper.batch_to_tensor(w2v, [(q1,q2)], [session])

        #q1, q2, t1_tensor, ses, q1_lenght, q2_lenght = helper.batch_to_tensor(q1,q2,1, [session])
        sample = []
        sample.append([q1])
        sample.append([q2])
        q1, q2, t1_tensor, ses, q1_lenght, q2_lenght = helper.batch_to_tensor(sample,[session])

        q1_fast, q2_fast, q1_len, q2_len =  helper.batch_to_tensor_fasttext(sam,w2v_fast,w2v1)


        self.model.eval()
        #label = torch.tensor(label*1, dtype=torch.long).to(device).unsqueeze(0)
        if self.config.cuda:
            q1 = q1.cuda()
            q2 = q2.cuda()# batch_size x max_len
            ses = ses.cuda()
            q1_fast = q1_fast.cuda()
            q2_fast = q2_fast.cuda()

        with torch.no_grad():
            out,cos = self.model(q1, q2, ses, q1_lenght, q2_lenght,q1_fast,q1_len, q2_fast, q2_len) #, q1_lenght, q2_lenght)

            #loss = self.loss_fn(out, label)
            #out = (out > 0.5) * 1
            #_, pred = torch.max(out, dim=1)
            out = out.detach().cpu().numpy()
            cos = cos.detach().cpu().numpy()
        return out, cos
        #return out[0][pred]

    def validate_oneshot(self, samples_b,labels_b,sessions_b, w2v_fast, w2v1):


        print('number of train batches = ', len(samples_b))
        samples_b, labels_b, sessions_b  = helper.test_batchify( samples_b, labels_b, sessions_b, self.batch_size)

        self.model.eval()
        x = np.arange(len(samples_b))
        yhat, ytest = [], []
        for bn in tqdm(x):

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

