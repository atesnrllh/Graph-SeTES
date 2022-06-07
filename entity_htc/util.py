from argparse import ArgumentParser

def get_args():

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='/media/nurullah/E/datasets/Gayo-Avello dataset/webis-smc-12/webis-smc-12.csv',
                        help='location of the data corpus')
    # parser.add_argument('--batch_size', type=int, default=256, metavar='N',
    #                     help='batch size')

    parser.add_argument('--train_data', type=str,
                        default='/home/nurullah/Desktop/data_valid_gayo_squential_split/10/train',
                        help='location of the data corpus')
    parser.add_argument('--valid_data', type=str,
                        default='/home/nurullah/Desktop/data_valid_gayo_squential_split/10/valid',
                        help='location of the data corpus')

    parser.add_argument('--test_data', type=str,
                        default='/home/nurullah/Desktop/data_valid_gayo_squential_split/10/test.csv',
                        help='location of the data corpus')

    parser.add_argument('--bidirection', default=True,
                        help='use bidirectional recurrent unit')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--char_emsize', type=int, default=98,
                        help='size of word embeddings')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper limit of epoch')
    parser.add_argument('--emtraining', action='store_true',
                        help='train embedding layer')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--nhid_emb', type=int, default=32,
                        help='number of hidden units per layer for the encoder')
    parser.add_argument('--nlayer_emb', type=int, default=1,
                        help='number of layers in the encoder')
    parser.add_argument('--nhid_enc', type=int, default=32,
                        help='number of hidden units per layer for the encoder')
    parser.add_argument('--nlayer_enc', type=int, default=1,
                        help='number of layers in the encoder')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--cuda', default="cuda:0", help='use CUDA for computation')
    parser.add_argument('--m', type=int, default=1, help='')
    parser.add_argument('--n', type=int, default=0, help='')
    parser.add_argument('--max_query_length', type=int, default=10, help='maximum length of a query')
    parser.add_argument('--tokenize', action='store_true', help='tokenize instances using word_tokenize')
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='ca_lstm/', help='path to save the final model')
    parser.add_argument('--word_vectors_file', type=str, default='glove.840B.300d.txt',
                        help='GloVe word embedding file name')
    parser.add_argument('--word_vectors_directory', type=str, default='/media/nurullah/E/datasets/dictionary/',
                        help='Path of GloVe word embeddings')
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help='gradient clipping')
    parser.add_argument('--early_stop', type=int, default=10000,
                        help='early stopping criterion')
    args = parser.parse_args()
    return args
