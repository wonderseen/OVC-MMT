'''
Our implementation of our proposed OVC refers to the original data pipeline of VAG-NMT
and modifications were done, including:
1. Preprocess all object-level features and detect translation-relevant/irrelevant objects
   in Multi30K, which needs lots of prep work, which costs around 1-2 days.
2. Preprocess all vision-weighted weights using pretrained LM checkpoints in the CPU mode,
   which costs about several hours.
3. Modificate the data pipeline to match the data flow of several OVC variants.
4. Rewrite our own OVC and its variants.
5. Each run for training a OVC model costs around 4 hours on 2 2080Ti GPUs.


## To run the model using the shell command
=> [activate the Pytorch enviroments, particularly we used Pytorch-1.2]
=> . run_ovc_training.sh

## To test the model using a better beam_size using the shell command
=> . run_ovc_evaluation.sh.

'''

import torch
from torch import optim, nn
import time
import argparse
import numpy as np
import os
from preprocessing import *
from machine_translation_vision.meteor.meteor import Meteor
from machine_translation_vision.user_loss import *
from machine_translation_vision.models import NMT_AttentionImagine_Seq2Seq_Beam_V16 as OVC
from train import *
from bleu import *


standard_exp = True # False: source-degradation, True: standard
mix = False # For mixed reimplementation
portion = 1.0 if mix else 0.0 


use_cuda = torch.cuda.is_available()
assert use_cuda, "pls use CUDA."

## Initialize the terms from argparse
PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
PARSER.add_argument('--data_path', required=True, help='path to multimodal machine translation dataset')
PARSER.add_argument('--trained_model_path', required=True, help='path to save the trained model and output')
PARSER.add_argument('--sr', type=str, required=True, help='the source language')
PARSER.add_argument('--tg', type=str, required=True, help='the target language')
# Network Structure
PARSER.add_argument('--imagine_attn', type=str, default='dot', help='attntion type for imagine_attn module. Options can be dot|mlp')
PARSER.add_argument('--activation_vse', action='store_false', help='whether using tanh after embedding layers')
PARSER.add_argument('--embedding_size', type=int, default=256, help='embedding layer size for both encoder and decoder')
PARSER.add_argument('--hidden_size', type=int, default=512, help='hidden state size for both encoder and decoder')
PARSER.add_argument('--shared_embedding_size', type=int, default=512, help='the shared space size to project decoder/encoder hidden state and image features')
PARSER.add_argument('--n_layers', type=int, default=1, help='number of stacked layer for encoder and decoder')
PARSER.add_argument('--tied_emb', action='store_false', help='whether to tie the embdding layers weights to the output layer')

# Dropout
PARSER.add_argument('--dropout_im_emb', type=float, default=0.2, help='the dropout applied to im_emb layer')
PARSER.add_argument('--dropout_txt_emb', type=float, default=0.4, help='the dropout applied to the text_emb layer')
PARSER.add_argument('--dropout_rnn_enc', type=float, default=0.0, help='the dropout applied to the rnn encoder layer')
PARSER.add_argument('--dropout_rnn_dec', type=float, default=0.0, help='the dropout applied to the rnn decoder layer')
PARSER.add_argument('--dropout_emb', type=float, default=0.3, help='the dropout applied ot the embedding layer of encoder embedidng state')
PARSER.add_argument('--dropout_ctx', type=float, default=0.5, help='the dropout applied to the context vectors of encoder')
PARSER.add_argument('--dropout_out', type=float, default=0.5, help='the dropout applied to the output layer of the decoder')

# Training Settingx
PARSER.add_argument('--batch_size', type=int, default=32, help='batch size during training')
PARSER.add_argument('--eval_batch_size', type=int, default=32, help='batch size during evaluation')
PARSER.add_argument('--learning_rate_mt', type=float, default=0.001, help='learning rate for machien translation task')
PARSER.add_argument('--learning_rate_vse', type=float, default=0., help='learning rate for VSE learning')
PARSER.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay applied to optimizer')
PARSER.add_argument('--loss_w', type=float, default=0.9, help='is using the mixed objective, this assigns the weight for mt and vse objective function separate.')
PARSER.add_argument('--p_mt', type=float, default=0.9, help='The probability to run machine translation task instead of vse task, when we train the tasks separately')
PARSER.add_argument('--beam_size', type=int, default=8, help='The beam size for beam search')
PARSER.add_argument('--n_epochs', type=int, default=100, help='maximum number of epochs to run')
PARSER.add_argument('--print_every', type=int, default=100, help='print frequency')
PARSER.add_argument('--eval_every', type=int, default=1000, help='evaluation frequency')
PARSER.add_argument('--save_every', type=int, default=10000, help='model save frequency')
PARSER.add_argument('--vse_separate', action='store_true', help='with mixed opjective functioin, do we apply different learning rate for different modules')
PARSER.add_argument('--vse_loss_type', type=str, default='pairwise', help='the type of vse loss which can be picked from pairwise|imageretrieval')
PARSER.add_argument('--teacher_force_ratio', type=float, default=0.8, help='whether to apply teacher_force_ratio during trianing')
PARSER.add_argument('--clip', type=float, default=1.0, help='gradient clip applied duing optimization')
PARSER.add_argument('--margin_size', type=float, default=0.1, help='default margin size applied to vse learning loss')
PARSER.add_argument('--patience', type=int, default=10, help='early_stop_patience')
PARSER.add_argument('--init_split', type=float, default=0.5, help='init_split_ratio to initialize the decoder')

## Get all the argument
ARGS = PARSER.parse_args()

## Helper Functions to Print Time Elapsed and Estimated Time Remaining, give the current time and progress
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


############################################################# Load the Dataset #######################################################
MAX_LENGTH = 80 # abandon any sentence that is longer than this length
data_path = ARGS.data_path
trained_model_output_path = ARGS.trained_model_path
source_language = ARGS.sr
target_language = ARGS.tg
BPE_dataset_suffix = '.norm.tok.lc'
dataset_suffix = '.norm.tok.lc'
dataset_im_suffix = '.norm.tok.lc.10000bpe_ims'

## Initilalize a Meteor Scorer
Meteor_Scorer = Meteor(target_language)

## Create the directory for the trained_model_output_path
if not os.path.isdir(trained_model_output_path):
    os.mkdir(trained_model_output_path)

## weights of Lv for each token
if source_language != 'en':
    vi_suffix = '.obj2' + target_language
else:
    vi_suffix = '.' + source_language + '2' + target_language

## experiment type
if standard_exp:
    val = 'val'
    test = 'test'
    train = 'train'
    source_presuffix = '.10000bpe.'
else: # deep mask
    val = 'val'
    test = 'val'
    train = 'train'
    back_source_language = source_language
    source_language = source_language + '.ambiguous'
    source_presuffix = '.'

## Load the training dataset
train_source    = load_data(os.path.join(data_path, f"{train}{BPE_dataset_suffix}{source_presuffix}{source_language}"))
train_target    = load_data(os.path.join(data_path, f"{train}{BPE_dataset_suffix}.10000bpe.{target_language}"))
train_target_VI = load_VI(os.path.join(data_path, 'VI', f"{train}.{target_language}{vi_suffix}"), train_target)

if mix:
    additional_number = int(len(train_source) * portion)
    additonal_train_source = load_data(os.path.join(data_path, f"{train}{BPE_dataset_suffix}{source_presuffix}{back_source_language}"))
    additonal_train_target = load_data(os.path.join(data_path, f"{train}{BPE_dataset_suffix}.10000bpe.{target_language}"))
    additonal_train_target_VI = load_VI(os.path.join(data_path, 'VI', f"{train}.{target_language}{vi_suffix}"), train_target)
    train_source = train_source[:additional_number] + additonal_train_source
    train_target = train_target[:additional_number] + additonal_train_target
    train_target_VI = train_target_VI[:additional_number] + additonal_train_target_VI
print(f"The size of Training Source and Training Target is: {len(train_source)} <=> {len(train_target)}")

## Load the validation dataset
val_source      = load_data(os.path.join(data_path, f"{val}{BPE_dataset_suffix}{source_presuffix}{source_language}"))
val_target      = load_data(os.path.join(data_path, f"{val}{BPE_dataset_suffix}.10000bpe.{target_language}"))

val_target_VI   = load_VI(os.path.join(data_path, 'VI', f"{val}.{target_language}{vi_suffix}"), val_target)
print(f"The size of Validation Source and Validation Target is: {len(val_source)} <=> {len(val_target)}")

## Load the test dataset
test_source     = load_data(os.path.join(data_path, f"{test}{BPE_dataset_suffix}{source_presuffix}{source_language}"))
test_target     = load_data(os.path.join(data_path, f"{test}{BPE_dataset_suffix}.10000bpe.{target_language}"))
test_target_VI  = load_VI(os.path.join(data_path, 'VI', f"{test}.{target_language}{vi_suffix}"), test_target)
print(f"The size of Test Source and Test Target is: {len(test_source)} <=> {len(test_target)}")

## Load the original validation dataset
val_ori_source  = load_data(os.path.join(data_path, f"{val}{dataset_suffix}.{source_language}"))
val_ori_target  = load_data(os.path.join(data_path, f"{val}{dataset_suffix}.{target_language}"))

## Load the original test dataset
test_ori_source = load_data(os.path.join(data_path, f"{test}{dataset_suffix}.{source_language}"))
test_ori_target = load_data(os.path.join(data_path, f"{test}{dataset_suffix}.{target_language}"))

## Creating List of pairs in the format of [[en_1, de_1], [en_2, de_2], ....[en_3, de_3]]
train_data = [[x.strip(), y.strip(), vi] for x, y, vi in zip(train_source, train_target, train_target_VI)]
val_data   = [[x.strip(), y.strip(), vi] for x, y, vi in zip(val_source, val_target, val_target_VI)]
test_data  = [[x.strip(), y.strip(), vi] for x, y, vi in zip(test_source, test_target, test_target_VI)]

## Creating List of pairs in the format of [[en_1, de_1], [en_2, de_2], ....[en_3, de_3]] for original data
val_ori_data  = [[x.strip(), y.strip()] for x, y in zip(val_ori_source, val_ori_target)]
test_ori_data = [[x.strip(), y.strip()] for x, y in zip(test_ori_source, test_ori_target)]


## Filter the data
train_data  = data_filter(train_data, MAX_LENGTH)
val_data    = data_filter(val_data, MAX_LENGTH)
test_data   = data_filter(test_data, MAX_LENGTH)

## Filter the original data
val_ori_data  = data_filter(val_ori_data, MAX_LENGTH)
test_ori_data = data_filter(test_ori_data, MAX_LENGTH)

print(f"The size of Training Data after filtering: {len(train_data)}")
print(f"The size of Val Data after filtering: {len(val_data)}")
print(f"The size of Test Data after filtering: {len(test_data)}")

## Load the Vocabulary File and Create Word2Id and Id2Word dictionaries for translation
vocab_source = load_data(os.path.join(data_path, f'vocab.{source_language}')) if not mix \
                 else load_data(os.path.join(data_path, f'vocab.{back_source_language}.mix')) 
vocab_target = load_data(os.path.join(data_path, f'vocab.{target_language}'))

## Construct the source_word2id, source_id2word, target_word2id, target_id2word dictionaries
s_word2id, s_id2word = construct_vocab_dic(vocab_source)
t_word2id, t_id2word = construct_vocab_dic(vocab_target)

print("The vocabulary size for soruce language: {}".format(len(s_word2id)))
print("The vocabulary size for target language: {}".format(len(t_word2id)))

## Generate Train, Val and Test Indexes pairs
train_data_index = create_data_index_VI(train_data, s_word2id, t_word2id)
val_data_index   = create_data_index_VI(val_data, s_word2id, t_word2id)
test_data_index  = create_data_index_VI(test_data, s_word2id, t_word2id)

val_y_ref  = [[d[1].split()] for d in val_ori_data]
test_y_ref = [[d[1].split()] for d in test_ori_data]

## Define val_y_ref_meteor and test_y_ref_meteor
val_y_ref_meteor = dict((key, [value[1]]) for key, value in enumerate(val_ori_data))
test_y_ref_meteor = dict((key, [value[1]]) for key, value in enumerate(test_ori_data))

## Load the Vision Features
train_im_feats  = np.load(os.path.join(data_path, train+dataset_im_suffix+'.npy'))
if mix: train_im_feats = np.concatenate((train_im_feats[:additional_number], train_im_feats), axis=0)
val_im_feats    = np.load(os.path.join(data_path, val+dataset_im_suffix+'.npy'))
test_im_feats   = np.load(os.path.join(data_path, test+dataset_im_suffix+'.npy'))

## Load the bottom-up and top-down attention vision features
# train_bta_im_feats  = np.load(os.path.join(data_path, train+dataset_im_suffix+'_bta_sort_with_original_similarity.npy'),
train_bta_im_feats  = np.load(os.path.join(data_path, train+dataset_im_suffix+'_bta_sort.npy'),
                        allow_pickle=True)
if mix: train_bta_im_feats = np.concatenate((train_bta_im_feats[:additional_number], train_bta_im_feats), axis=0)
val_bta_im_feats    = np.load(os.path.join(data_path, val+dataset_im_suffix+'_bta_sort.npy'),
                        allow_pickle=True)
test_bta_im_feats   = np.load(os.path.join(data_path, test+dataset_im_suffix+'_bta_sort.npy'),
                        allow_pickle=True)

## Verify the size of the train_im_features
print("Training Image Feature Size is: {}".format(train_im_feats.shape))
print("Validation Image Feature Size is: {}".format(val_im_feats.shape))
print("Testing Image Feature Size is: {}".format(test_im_feats.shape))
assert train_bta_im_feats.shape[0] == train_im_feats.shape[0], (train_bta_im_feats.shape, train_im_feats.shape)
assert val_bta_im_feats.shape[0]   == val_im_feats.shape[0], (val_bta_im_feats.shape, val_im_feats.shape)
assert test_bta_im_feats.shape[0]  == test_im_feats.shape[0], (test_bta_im_feats.shape, test_im_feats.shape)

############################## Define Model and Training Structure ##################################
## Network Structure
imagine_attn = ARGS.imagine_attn
activation_vse = ARGS.activation_vse
embedding_size = ARGS.embedding_size
hidden_size = ARGS.hidden_size
shared_embedding_size = ARGS.shared_embedding_size
n_layers = ARGS.n_layers
tied_emb = ARGS.tied_emb

## Dropout
dropout_im_emb = ARGS.dropout_im_emb
dropout_txt_emb = ARGS.dropout_txt_emb
dropout_rnn_enc = ARGS.dropout_rnn_enc
dropout_rnn_dec = ARGS.dropout_rnn_dec
dropout_emb = ARGS.dropout_emb
dropout_ctx = ARGS.dropout_ctx
dropout_out = ARGS.dropout_out

## Training Setting
batch_size = ARGS.batch_size
eval_batch_size = ARGS.eval_batch_size
batch_num = math.floor(len(train_data_index) / batch_size)
learning_rate = ARGS.learning_rate_mt
weight_decay = ARGS.weight_decay
loss_w= ARGS.loss_w
beam_size = ARGS.beam_size
n_epochs = ARGS.n_epochs
print_every = ARGS.print_every
eval_every = ARGS.eval_every
save_every = ARGS.save_every
vse_separate = ARGS.vse_separate
vse_loss_type = ARGS.vse_loss_type

## Define the teacher force_ratio
teacher_force_ratio = ARGS.teacher_force_ratio
clip = ARGS.clip

## Define the margin size
margin_size = ARGS.margin_size
patience = ARGS.patience

## Initialize models
input_size = len(s_word2id) + 1
output_size = len(t_word2id) + 1

## Definet eh init_split
init_split = ARGS.init_split

## Define the model
imagine_model = OVC(input_size, 
                output_size,
                train_im_feats.shape[1],
                train_bta_im_feats.shape[2],
                embedding_size, 
                embedding_size, 
                hidden_size, 
                shared_embedding_size, 
                loss_w, 
                activation_vse=activation_vse, 
                attn_model=imagine_attn, 
                dropout_ctx=dropout_ctx, 
                dropout_emb=dropout_emb, 
                dropout_out=dropout_out, 
                dropout_rnn_enc=dropout_rnn_enc, 
                dropout_rnn_dec=dropout_rnn_dec, 
                dropout_im_emb=dropout_im_emb, 
                dropout_txt_emb=dropout_txt_emb, 
                tied_emb=tied_emb,
                init_split=init_split)

if use_cuda: imagine_model.cuda()
## Use Multiple GPUs if they are available
"""
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    baseline_model = nn.DataParallel(baseline_model)
"""

## Define the loss criterion
vocab_mask = torch.ones(output_size)
vocab_mask[0] = 0
if use_cuda:
    vocab_mask = vocab_mask.cuda()

# criterion_mt = FocalLoss(reduction='none', with_logits=False)#, smooth_eps=0.1)
criterion_mt = nn.NLLLoss(weight=vocab_mask, reduce=False)
if use_cuda: criterion_mt = criterion_mt.cuda()

# criterion_vse = nn.HingeEmbeddingLoss(margin=margin_size, size_average=False)
# if vse_loss_type == "pairwise":
#     criterion_vse = PairwiseRankingLoss(margin=margin_size)
# if vse_loss_type == "imageretrieval":
#     criterion_vse = ImageRetrievalRankingLoss(margin=margin_size)
# if use_cuda: criterion_vse = criterion_vse.cuda()
criterion_vse = None
    


if not vse_separate:
    ## Define the optimizer
    # optimizer = optim.Adam(imagine_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weight_group = {
        'params': [p for n, p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' not in n],
        'weight_decay': weight_decay,
    }
    bias_group = {
        'params': [p for n, p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' in n],
    }
    param_groups = [weight_group, bias_group]
else:
    mt_weight_group = {
        'params': [p for n, p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' not in n and 'vse_imagine' not in n],
        'weight_decay': weight_decay,
    }
    mt_bias_group = {
        'params': [p for n, p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' in n and 'vse_imagine' not in n],
    }
    vse_weight_group = {
        'params': [p for n, p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' not in n and 'vse_imagine' in n],
        'weight_decay': weight_decay,
        'lr': learning_rate / 2,
    }
    vse_bias_group = {
        'params': [p for n, p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' in n and 'vse_imagine' in n],
        'lr': learning_rate / 2,
    }
    param_groups = [mt_weight_group, mt_bias_group, vse_weight_group, vse_bias_group]

## Define Optimizer
optimizer = optim.Adam(param_groups, lr=learning_rate) # Optimize the parameters

## Define a learning rate optimizer
lr_decay_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=10)

################################ Print the configuration settings #################################
print("Configurations:\n")
print("######## Network Structure #########")
print("embedding_size: {}".format(embedding_size))
print("hidden_size: {}".format(hidden_size))
print("shared_embedding_size: {}".format(shared_embedding_size))
print("n_layers: {}".format(n_layers))
print("tied_emb: {}".format(tied_emb))
print('\n')
print("####### Dropout #######")
print("dropout_im_emb: {}".format(dropout_im_emb))
print("dropout_txt_emb: {}".format(dropout_txt_emb))
print("dropout_rnn_enc: {}".format(dropout_rnn_enc))
print("dropout_rnn_dec: {}".format(dropout_rnn_dec))
print("dropout_emb: {}".format(dropout_emb))
print("dropout_ctx: {}".format(dropout_ctx))
print("dropout_out: {}".format(dropout_out))
print('\n')
print("####### Training Setting #######")
print("batch_size: {}".format(batch_size))
print("eval_batch_size: {}".format(eval_batch_size))
print("learning_rate: {}".format(learning_rate))
print("weight_decay: {}".format(weight_decay))
print("loss_w: {}".format(loss_w))
print("beam_size: {}".format(beam_size))
print("n_epochs: {}".format(n_epochs))
print("print_every: {}".format(print_every))
print("eval_every: {}".format(eval_every))
print("save_every: {}".format(save_every))
print("vse_separate: {}".format(vse_separate))
print("teacher_force_ratio: {}".format(teacher_force_ratio))
print("clip: {}".format(clip))
print("input_size: {}".format(input_size))
print("output_size: {}".format(output_size))
print("vse_margin: {}".format(margin_size))
print("vse_loss_type: {}".format(vse_loss_type))
print("init_split: {}".format(init_split))
print('\n')

########################################## Begin Training ###########################################
print_mt_loss = 0
print_vse_loss = 0
print_loss = 0

## Start Training
print("Begin Training")
iter_count = 0
best_bleu = 0
best_meteor = 0
best_loss = 10000000
early_stop = patience

start = time.time()
print(f"\n\nportion = {portion}") 
print("source_language -> target_language", source_language, target_language)
print(f"Trained_model_output_path{trained_model_output_path}\n\n")
for epoch in range(1, n_epochs + 1):
    # for batch_x, batch_y, batch_vi, batch_im, batch_bta_im, batch_x_lengths, batch_y_lengths in data_generator_tl_mtv_bta_vi_shuffle(
    for batch_x, batch_y, batch_vi, batch_im, batch_bta_im, batch_x_lengths, batch_y_lengths in data_generator_tl_mtv_bta_vi(
                                                                              train_data_index,
                                                                              train_im_feats,
                                                                              train_bta_im_feats,
                                                                              batch_size):

        ## Run the train function
        train_loss, train_loss_mt, train_loss_vse = train_imagine_beam_bta_vi(
                                                        batch_x, batch_y, batch_vi,
                                                        batch_im, batch_bta_im,
                                                        batch_x_lengths,
                                                        imagine_model,
                                                        optimizer,
                                                        criterion_mt, criterion_vse,
                                                        loss_w,
                                                        teacher_force_ratio,
                                                        clip=clip)
        
        print_loss += train_loss

        # if use_cuda:
        #     torch.cuda.empty_cache()

        ## Update translation loss and vse loss
        print_mt_loss += train_loss_mt
        print_vse_loss += train_loss_vse
        
        if iter_count == 0: 
            iter_count += 1
            continue
        
        if iter_count % print_every == 0:
            print_loss_avg = print_loss / print_every
            print_mt_loss_avg = print_mt_loss / print_every
            print_vse_loss_avg = print_vse_loss / print_every
            # Reset the print_loss, print_mt_loss and print_vse_loss
            print_loss = 0
            print_mt_loss = 0
            print_vse_loss = 0
            
            print_summary = "%s (%d %d%%) train_loss: %.4f, train_mt_loss: %.4f, train_vse_loss: %.4f" % (
                time_since(start, iter_count / n_epochs / batch_num),
                iter_count,
                iter_count / n_epochs / batch_num * 100,
                print_loss_avg,
                print_mt_loss_avg,
                print_vse_loss_avg)
            print(print_summary)
        
        if iter_count % eval_every == 0:
            ## Convert model into eval phase
            imagine_model.eval()
            val_translations = []

            ## Compute Val Loss
            ## Print the Bleu Score and loss for Dev Dataset
            val_print_loss = 0
            val_print_mt_loss = 0
            val_print_vse_loss = 0
            eval_iters = 0
            for val_x, val_y, val_vi, val_im, val_bta_im, val_x_lengths, val_y_lengths in data_generator_tl_mtv_bta_vi(
                                                                            val_data_index,
                                                                            val_im_feats,
                                                                            val_bta_im_feats,
                                                                            batch_size):
                val_loss, val_mt_loss, val_vse_loss = imagine_model(
                    val_x,
                    val_x_lengths,
                    val_y,
                    val_vi,
                    val_im,
                    val_bta_im,
                    teacher_force_ratio,
                    criterion_mt=criterion_mt,
                    criterion_vse=criterion_vse)
                val_print_loss += val_loss.item()
                val_print_mt_loss += val_mt_loss.item()
                val_print_vse_loss += val_vse_loss.item()
                eval_iters += 1
            ## Compute the Average Losses
            val_loss_mean = val_print_loss / eval_iters
            val_mt_loss_mean = val_print_mt_loss / eval_iters
            val_vse_loss_mean = val_print_vse_loss / eval_iters

            ## Check the val_mt_loss_mean
            lr_decay_scheduler.step(val_mt_loss_mean)

            ## Save the model when it reaches the best validation loss or best BLEU score
            if val_mt_loss_mean < best_loss:
                torch.save(imagine_model, os.path.join(trained_model_output_path, 'nmt_trained_imagine_model_best_loss.pt'))
                ## update the best_loss
                best_loss = val_mt_loss_mean
            print(f"dev_loss: {val_loss_mean}, dev_mt_loss: {val_mt_loss_mean}, dev_vse_loss: {val_vse_loss_mean}")


            ## Generate translation
            for val_x, val_y, val_im, val_bta_im, val_x_lengths, val_y_lengths, val_sorted_index in data_generator_bta_mtv(
                                                                          val_data_index,
                                                                          val_im_feats,
                                                                          val_bta_im_feats,
                                                                          eval_batch_size):
                with torch.no_grad():
                    val_translation, _ = imagine_model.beamsearch_decode(
                        val_x,
                        val_x_lengths,
                        val_im,
                        val_bta_im,
                        beam_size,
                        max_length=MAX_LENGTH) # Optimize to take in the Image Variables

                # Reorder val_translations and convert them back to words
                val_translation_reorder = translation_reorder_BPE(val_translation, val_sorted_index, t_id2word) 
                val_translations += val_translation_reorder
            
            ## Compute the BLEU Score
            val_bleu = compute_bleu(val_y_ref, val_translations)

            ## Compute the METEOR Score
            val_translations_meteor = dict((key, [' '.join(value)]) for key, value in enumerate(val_translations))
            val_meteor = Meteor_Scorer.compute_score(val_y_ref_meteor, val_translations_meteor)

            print(f"dev_bleu: {val_bleu[0]}, dev_meteor: {val_meteor[0]}")

            ## Randomly Pick a sentence and translate it to the target language. 
            sample_source, sample_ref, sample_output = random_sample_display(val_ori_data, val_translations)
            print("An example demo:")
            print("src: {}".format(sample_source))
            print("ref: {}".format(sample_ref))
            print("pred: {}".format(sample_output))
        
            if val_bleu[0] > best_bleu:
                torch.save(imagine_model, os.path.join(trained_model_output_path, 'nmt_trained_imagine_model_best_BLEU.pt'))
                ## update the best_bleu score
                best_bleu = val_bleu[0]
                early_stop = patience
            else:
                early_stop -= 1

            if val_meteor[0] > best_meteor:
                torch.save(imagine_model, os.path.join(trained_model_output_path, 'nmt_trained_imagine_model_best_METEOR.pt'))
                ## update the best_bleu score
                best_meteor = val_meteor[0]

            ## Print out the best loss and best BLEU so far
            print(f"Current Early_Stop Counting: {early_stop}")
            print(f"Best Loss so far is: {best_loss}")
            print(f"Best BLEU so far is: {best_bleu}")
            print(f"Best METEOR so far is: {best_meteor}")
        if iter_count % save_every == 0:
            ## Save the model every save_every iterations.
            torch.save(imagine_model, os.path.join(trained_model_output_path, f'nmt_trained_imagine_model_{iter_count}.pt'))
        
        if early_stop == 0:
            break
        
        ## Update the Iteration
        iter_count += 1
    
    if early_stop == 0:
        break

print("Training is done.")
print("Evalute the Test Result")


######################### Use the best BLEU Model to Evaluate #####################################################
## Load the Best BLEU Model
best_model = torch.load(os.path.join(trained_model_output_path, 'nmt_trained_imagine_model_best_BLEU.pt'))
if use_cuda:
    best_model.cuda()

## Convert best_model to eval phase
best_model.eval()

test_translations = []
for test_x, test_y, test_im, test_bta_im, test_x_lengths, test_y_lengths, test_sorted_index in data_generator_bta_mtv(
                                                              test_data_index,
                                                              test_im_feats,
                                                              test_bta_im_feats,
                                                              eval_batch_size):
    with torch.no_grad():
        test_translation, _ = best_model.beamsearch_decode(
            test_x,
            test_x_lengths,
            test_im,
            test_bta_im,
            beam_size,
            MAX_LENGTH)

    ## Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation, test_sorted_index, t_id2word) 
    test_translations += test_translation_reorder


## Compute the test bleu score
test_bleu = compute_bleu(test_y_ref, test_translations)

## Compute the METEOR Score
test_translations_meteor = dict((key,[' '.join(value)]) for key, value in enumerate(test_translations))
test_meteor = Meteor_Scorer.compute_score(test_y_ref_meteor, test_translations_meteor)

print(f"Test BLEU score from the best BLEU model: {test_bleu[0]}")
print(f"Test METEOR score from the best BLEU model: {test_meteor[0]}")
print("\n")
## Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(trained_model_output_path, f'test_2017_prediction_best_BLEU.{target_language}')

with open(test_prediction_path, 'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')


########################### Use the best METEOR Model to Evaluate #############################################
## Load the Best METEOR Model
best_meteor_model = torch.load(os.path.join(trained_model_output_path, 'nmt_trained_imagine_model_best_METEOR.pt'))
if use_cuda:
    best_meteor_model.cuda()

## Convert best_model to eval phase
best_meteor_model.eval()

test_translations = []
for test_x, test_y, test_im, test_bta_im, test_x_lengths, test_y_lengths, test_sorted_index in data_generator_bta_mtv(
                                                              test_data_index,
                                                              test_im_feats,
                                                              test_bta_im_feats,
                                                              eval_batch_size):
    with torch.no_grad():
        test_translation, _ = best_meteor_model.beamsearch_decode(
            test_x,
            test_x_lengths,
            test_im,
            test_bta_im,
            beam_size,
            MAX_LENGTH)

    # Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation, test_sorted_index, t_id2word) 
    test_translations += test_translation_reorder


## Compute the test bleu score
test_bleu = compute_bleu(test_y_ref, test_translations)

## Compute the METEOR Score
test_translations_meteor = dict((key, [' '.join(value)]) for key, value in enumerate(test_translations))
test_meteor = Meteor_Scorer.compute_score(test_y_ref_meteor, test_translations_meteor)

print(f"Test BLEU score from the best METEOR model: {test_bleu[0]}")
print(f"Test METEOR score from the best METEOR model: {test_meteor[0]}")

## Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(trained_model_output_path, f'test_2017_prediction_best_METEOR.{target_language}')

with open(test_prediction_path, 'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')


########################### Use the best loss Model to Evaluate #############################################
## Load the Best Model
best_loss_model = torch.load(os.path.join(trained_model_output_path, 'nmt_trained_imagine_model_best_loss.pt'))
if use_cuda:
    best_loss_model.cuda()
    
## Convert best_model to eval phase
best_loss_model.eval()
test_translations = []
for test_x, test_y, test_im, test_bta_im, test_x_lengths, test_y_lengths, test_sorted_index in data_generator_bta_mtv(
                                                              test_data_index,
                                                              test_im_feats,
                                                              test_bta_im_feats,
                                                              eval_batch_size):
    with torch.no_grad():
        test_translation, _ = best_loss_model.beamsearch_decode(
            test_x,
            test_x_lengths,
            test_im,
            test_bta_im,
            beam_size,
            MAX_LENGTH)
    # Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation, test_sorted_index, t_id2word) 
    test_translations += test_translation_reorder

## Compute the test bleu score
test_bleu = compute_bleu(test_y_ref, test_translations)

## Compute the METEOR Score
test_translations_meteor = dict((key, [' '.join(value)]) for key, value in enumerate(test_translations))
test_meteor = Meteor_Scorer.compute_score(test_y_ref_meteor, test_translations_meteor)
print(f"\nTest BLEU score from the best loss model: {test_bleu[0]}")
print(f"Test METEOR score from the best loss model: {test_meteor[0]}")

## Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(trained_model_output_path, f'test_2017_prediction_best_loss.{target_language}')
with open(test_prediction_path, 'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')
