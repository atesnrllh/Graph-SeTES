
from torch import nn
import torch
from nn_layer import Lstm_Embedding, Encoder, Attention, Attn, brsoff_Attn
from torch.nn import functional as F
import helper
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.config = args

        Ci = 1
        Co = 60
        Ks = [1,2,3,4,5]

        self.emb_linear = nn.Linear(self.config.emsize, self.config.emsize)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, self.config.emsize)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, q):

        emb_linear1 = (self.relu(self.emb_linear(q)))
        x = emb_linear1.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x[:], 1)
        return x


class LSTM(nn.Module):

    def __init__(self, dimension=150):
        super(LSTM, self).__init__()

        self.embedding = nn.Linear(300, 300)

        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.1)
        self.dimension = dimension
        self.fc = nn.Linear(2*dimension,300)
        self.relu = nn.ReLU()

    def forward(self, text, text_len):

        #text_emb = self.relu(self.embedding(text))

        packed_input = pack_padded_sequence(text, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        x = torch.cat((out_forward, out_reverse), 1)
        #text_fea = self.drop(out_reduced)

        #x = self.drop(self.relu(self.fc(x)))
        #x = self.relu(self.fc(x))
        x = torch.squeeze(x, 1)
        #text_out = torch.sigmoid(text_fea)

        return x

class LinearLayer(nn.Module):

    def __init__(self):
        super(LinearLayer, self).__init__()

        self.relu = nn.ReLU()

        self.cosd = nn.Linear(300,300)
        self.eucd  = nn.Linear(300,300)
        self.paird = nn.Linear(300,300)
        self.time = nn.Linear(300,300)
        self.cosdf = nn.Linear(300,300)
        self.eucdf  = nn.Linear(300,300)
        self.pairdf = nn.Linear(300,300)

        self.d5 = nn.Linear(300,1)
        self.d4 = nn.Linear(300,1)
        self.d3 = nn.Linear(300,1)
        self.d543 = nn.Linear(3,3)

        self.d2 = nn.Linear(1,1)
        self.d1 = nn.Linear(1,1)
        self.d0 = nn.Linear(1,1)
        self.d210 = nn.Linear(3,3)

        self.concat = nn.Linear(1200,300)
        self.concatf = nn.Linear(1200,300)
        self.concat2 = nn.Linear(600,100)
        self.concat3 = nn.Linear(4,4)
        self.classifier = nn.Linear(100,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)


    def forward(self, pairwise, cos, euclid, time_diff, pairwisef, cosf, euclidf):

        ### d9
        time_diff =  time_diff.repeat(1,300)
        time_diff = self.time(time_diff)


        ### d8 d7 d6
        pairwise =  pairwise.repeat(1,300)
        pairwise = self.dropout(self.relu(self.paird(pairwise)))

        cos =  cos.repeat(1,300)
        cos = self.dropout(self.relu(self.cosd(cos)))

        euclid =  euclid.repeat(1,300)
        euclid = self.dropout(self.relu(self.eucd(euclid)))

        pairwisef =  pairwisef.repeat(1,300)
        pairwisef = self.dropout(self.relu(self.pairdf(pairwisef)))

        cosf =  cosf.repeat(1,300)
        cosf = self.dropout(self.relu(self.cosdf(cosf)))

        euclidf =  euclidf.repeat(1,300)
        euclidf = self.dropout(self.relu(self.eucdf(euclidf)))


        ### d5 d4 d3
        #d543 = torch.cat([self.dropout(self.relu(self.d3(sub))), self.dropout(self.relu(self.d4(sub_mul))), self.dropout(self.relu(self.d5(mul_sub)))], dim=1)
        #d543 = self.dropout(self.relu(self.d543(d543)))


        ### d2 d1 d0
        #d210 = torch.cat([self.dropout(self.relu(self.d0(sub))), self.dropout(self.relu(self.d1(sub_mul_abs))), self.dropout(self.relu(self.d2(dia_mul_sub.unsqueeze(1))))], dim=1)
        #d210 = self.dropout(self.relu(self.d210(d210)))


        conca = torch.cat([pairwise, cos, euclid, time_diff],dim=1)
        conca = self.dropout(self.relu(self.concat(conca)))

        concatf = torch.cat([pairwisef, cosf, euclidf, time_diff],dim=1)
        concatf = self.dropout(self.relu(self.concatf(concatf)))

        x = torch.cat([conca, concatf], dim=1)
        x = self.dropout(self.relu(self.concat2(x)))
        x = self.classifier(x)

        return self.sigmoid(x.squeeze(1))



class Siamese(nn.Module):

    def __init__(self, args):
        super(Siamese, self).__init__()
        self.LSTM = LSTM()
        self.LSTM2 = LSTM()
        self.CNN = CNN(args)
        self.CNN2 = CNN(args)
        self.LinearLayer = LinearLayer()

    def distance(self,x1,x2):

        self.pdist = nn.PairwiseDistance(p=2)
        pairwise = self.pdist(x1,x2).unsqueeze(1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos =  self.cos(x1, x2).unsqueeze(1)
        euclid = torch.diagonal(torch.cdist(x1, x2, p=2),0).unsqueeze(1)

        sub = torch.abs(x1 - x2)
        sub_mul = sub * sub
        mul_sub = torch.abs(x1 * x1- x2 * x2)


        x1_ = torch.diagonal(torch.matmul(x1, x1.transpose(0,1)),0)
        x2_ = torch.diagonal(torch.matmul(x2, x2.transpose(0,1)),0)
        dia_mul_sub = torch.abs(torch.sqrt(x1_) - torch.sqrt(x2_))
        sub_mul_abs = torch.abs(sub * sub)

        return pairwise, cos, euclid, sub, sub_mul, mul_sub, dia_mul_sub, sub_mul_abs


    def forward(self, q1,q2, time_diff, q1_lenght, q2_lenght, q1_fast,q1_len, q2_fast, q2_len):
        #
        # q1, q2 = self.CNN(q1), self.CNN(q2)
        # q1_fast, q2_fast = self.CNN2(q1_fast), self.CNN2(q2_fast)
        q1, q2 = self.LSTM(q1, q1_lenght), self.LSTM(q2, q2_lenght)
        q1_fast, q2_fast = self.LSTM2(q1_fast, q1_len), self.LSTM2(q2_fast, q2_len)

        pairwise, cos, euclid, sub, sub_mul, mul_sub, dia_mul_sub, sub_mul_abs = self.distance(q1, q2)
        pairwise_f, cos_f, euclid_f, sub, sub_mul, mul_sub, dia_mul_sub, sub_mul_abs = self.distance(q1_fast, q2_fast)

        x = self.LinearLayer( pairwise, cos, euclid, time_diff, pairwise_f, cos_f, euclid_f)

        cos = (cos_f + cos_f) / 2
        return x, cos


