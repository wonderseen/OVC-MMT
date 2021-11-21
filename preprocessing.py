"""
This program file contains all the functions implemented to load and preprocess 
the dataset for OVC
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
from nltk.tokenize import sent_tokenize
import re
from machine_translation_vision.samplers import BucketBatchSampler
from collections import Counter
import unicodedata
import random

## Define a couple of parameters
SOS_token = 2
EOS_token = 3
UNK_token = 1
PAD_Token = 0
use_cuda = torch.cuda.is_available()

#Load the dataset in a text file located with data_path
def load_data(data_path):
    with open(data_path,'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data


def quick_sort(lists):
    if not lists:
        return []
    assert isinstance(lists, list)

    if len(lists) == 1:
        return lists

    pivot = lists.pop()
    llist, rlist = [], []

    for x in lists:
        if x > pivot:
            rlist.append(x)
        else:
            llist.append(x)
    return quick_sort(llist) + [pivot] + quick_sort(rlist)


def heap_struct(alist):
    alist.reverse()
    for i in range(int((len(alist)-1)/2)):
        if len(alist) < 2*i+1:
            break
        if alist[i] < alist[2*i+1]:
            alist[i], alist[2*i+1] = alist[2*i+1], alist[i]
        if alist[2*i+1] < alist[2*i+2]:
            alist[2*i+2], alist[2*i+1] = alist[2*i+1], alist[2*i+2]
    return alist


def heap_topk(s, k):
    topk = quick_sort(s[:k])
    for x in s[k:]:
        if x<topk[-1]:
            topk[-1] = x
            topk = heap_struct(topk)
            topk.reverse()
    return topk


def load_VI(data_path, target_sentence):
    with open(data_path,'r') as f:
        data = f.readlines()
    VIS = [[float(v) for v in vi.split()] for vi in data]

    new_VIS = []

    try:
        for id, (VI, sent) in enumerate(zip(VIS, target_sentence)):
            pos = -1
            new_vi = []
            tokens = sent.split()

            if tokens[-1] == 't.' and tokens[-2] in ['bar@@', 'dar@@']:## noisy case of the bpe tokenizer, bart. -> bar@@ t.
                VI = VI[:-1]

            ##
            for i, vi in enumerate(VI):

                ##
                temp_pos, bpe_length = pos, 1
                while tokens[temp_pos].endswith('@@'):
                    temp_pos += 1
                    bpe_length += 1
                vi /= float(bpe_length)

                ##
                pos += 1
                new_vi.append(vi)  

                while tokens[pos].endswith('@@'):
                    pos += 1
                    new_vi.append(vi)                
                    if pos == len(tokens)-1: break
            assert pos == len(tokens)-1, (pos, len(tokens)-1)
            assert len(tokens) == len(new_vi), (len(tokens), len(new_vi))
            assert i==len(VI)-1, (i, len(VI)-1)
            
            ## top k
            # top_k = heap_topk(new_vi, 5)
            # top_k = list(Counter(top_k).keys())
            # if 0. in top_k: top_k.remove(0.)
            # if top_k is not None:
            #     new_vi = [vi if vi in top_k else 0. for vi in new_vi]

            # print(id)
            # print(new_vi)
            # print(tokens)

            new_VIS.append(new_vi)
    except BaseException:
        print(id, len(VI), VI)
        print(len(tokens), sent)
    assert len(target_sentence) == len(new_VIS), (len(target_sentence), len(new_VIS), len(VIS))
    return new_VIS


def format_data(data_x, data_y, IKEA=False):
    if not IKEA:
        data=[[x.strip(), y.strip()] for x, y in zip(data_x, data_y)]
    else:
        data=[]
        for x, y in zip(data_x, data_y):
            ##conver the paragraph into sentences
            x_s = sent_tokenize(x)
            y_s = sent_tokenize(y)
            ## Check if len of the list is the same
            if len(x_s) == len(y_s):
                data += [[x.strip(), y.strip()] for x, y in zip(x_s, y_s)]
    return data


#Construct Word2Id and Id2Word Dictionaries from a loaded vocab file
def construct_vocab_dic(vocab):
    word2id = {}
    id2word = {}
    for i,word in enumerate(vocab):
        word2id[word.strip()] = i + 1
        id2word[i + 1] = word.strip()
    return word2id,id2word


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


# Filter out the pairs that has a sentence having
def data_filter(data, max_length):
    new_data = []
    for d in data:
        if len(d[0].split()) <= max_length and len(d[1].split()) <= max_length:
            new_data.append(d)
    return new_data


missing_words = []
def indexes_from_sentence(vocab, sentence, drop_unk=False):
    words = sentence.split(' ')
    for i, word in enumerate(words):
        if word not in vocab.keys():
            if _run_strip_accents(word) in vocab.keys():
                words[i] = _run_strip_accents(word)
            else:
                if word not in missing_words:
                    missing_words.append(word)
    indexes = [vocab.get(word, UNK_token) for word in words]
    if drop_unk: indexes = [i for i in indexes if i != UNK_token]
    return indexes


def variable_from_sentence(vocab, sentence):
    indexes =  (vocab, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        var = var.cuda()
    return var


def variables_from_pair(pair,s_vocab, t_vocab):
    input_variable = variable_from_sentence(s_vocab, pair[0])
    target_variable = variable_from_sentence(t_vocab, pair[1])
    return (input_variable, target_variable)


# Create data pairs with each pair represented by corresponding wordids in each language. 
def create_data_index(pairs, source_vocab, target_vocab, drop_unk=False):
    source_indexes = [indexes_from_sentence(source_vocab, x[0], drop_unk=drop_unk) + [EOS_token] for x in pairs]
    target_indexes = [indexes_from_sentence(target_vocab, x[1], drop_unk=drop_unk) + [EOS_token] for x in pairs]
    return [[s, t] for s, t in zip(source_indexes, target_indexes)]


def create_data_index_VI(pairs, source_vocab, target_vocab, drop_unk=False):
    source_indexes = [indexes_from_sentence(source_vocab, x[0], drop_unk=drop_unk) + [EOS_token] for x in pairs]
    target_indexes = [indexes_from_sentence(target_vocab, x[1], drop_unk=drop_unk) + [EOS_token] for x in pairs]

    vis = [x[2] + [0.] for x in pairs]
    return [[s, t, vi] for s, t, vi in zip(source_indexes, target_indexes, vis)]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq_new = seq + [0 for i in range(max_length - len(seq))]
    return seq_new


def data_generator(data_pairs, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size / batch_size)
    for i in range(0, data_size, batch_size):
        if i+batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i+batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i+batch_size]]
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        # Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)

        # Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              


        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        # Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        # Generate batch_x and batch_y
        batch_x = Variable(torch.LongTensor(batch_x_pad_sorted))
        batch_y = Variable(torch.LongTensor(batch_y_pad_sorted))
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        # Yield the batch data|
        yield batch_x, \
            batch_y, \
            list(reversed(sorted(batch_x_lengths))), \
            batch_y_lengths_sorted, \
            x_reverse_sorted_index


def data_generator_tl(data_pairs, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length.  
    """
    # Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    # Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths,batch_size)

    # Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        # Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        # Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        # Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)

        # Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        # Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(torch.LongTensor(batch_y_pad_sorted))
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        # Yield the batch data|
        yield batch_x, batch_y, list(reversed(sorted(batch_x_lengths))), batch_y_lengths_sorted


def data_generator_single(batch_data_x):
    x_length = max([len(x) for x in batch_data_x])

    # Get a list of tokens
    batch_x_pad = []
    batch_x_lengths = []
    
    # Updated batch_x_lengths, batch_x_pad
    for x_tokens in batch_data_x:
        x_l = len(x_tokens)
        x_pad_seq = pad_seq(x_tokens,x_length)
        batch_x_lengths.append(x_l)
        batch_x_pad.append(x_pad_seq)

    # Reorder the lengths
    x_sorted_index = list(np.argsort(batch_x_lengths))
    x_reverse_sorted_index = list(reversed(x_sorted_index))
    batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

    batch_x = Variable(torch.LongTensor(batch_x_pad_sorted))
    if use_cuda:
        batch_x = batch_x.cuda()
    return batch_x,list(reversed(sorted(batch_x_lengths))),x_reverse_sorted_index


def data_generator_mtv(data_pairs, data_im, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size / batch_size)
    for i in range(0, data_size, batch_size):
        if i+batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i+batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i+batch_size]]
            batch_data_im = torch.from_numpy(data_im[i:i+batch_size])
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]
            batch_data_im = torch.from_numpy(data_im[i:data_size])
            
        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
        
        #Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
        
        #Yield the batch data|
        yield batch_x, batch_y, batch_im, list(reversed(sorted(batch_x_lengths))), batch_y_lengths_sorted, x_reverse_sorted_index


def data_generator_bta_mtv(data_pairs, data_im, data_bta_im, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
        data_bta_im:
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size / batch_size)
    for i in range(0, data_size, batch_size):
        if i+batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i+batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i+batch_size]]
            batch_data_im = torch.from_numpy(data_im[i:i+batch_size])
            batch_data_bta_im = torch.from_numpy(data_bta_im[i:i+batch_size])
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]
            batch_data_im = torch.from_numpy(data_im[i:data_size])
            batch_data_bta_im = torch.from_numpy(data_bta_im[i:i+batch_size])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)

        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]

        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        batch_bta_im_sorted = torch.zeros_like(batch_data_bta_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
            batch_bta_im_sorted[i] = batch_data_bta_im[x]
        
        #Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        batch_bta_im = Variable(batch_bta_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
            batch_bta_im = batch_bta_im.cuda()
        
        #Yield the batch data|
        yield batch_x, batch_y, batch_im, batch_bta_im, list(reversed(sorted(batch_x_lengths))), batch_y_lengths_sorted,x_reverse_sorted_index


def data_generator_tl_mtv_bta_vi_shuffle(data_pairs, data_im, data_bta_im, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
        data_bta_im:
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """

    dt = [(a,b,c) for a,b,c in zip(data_pairs, data_im, data_bta_im)]
    random.shuffle(dt)

    data_pairs = [a[0] for a in dt]
    data_im = np.array([a[1] for a in dt])
    data_bta_im = np.array([a[2] for a in dt])

    data_size = len(data_pairs)
    num_batches = math.floor(data_size/batch_size)
    for i in range(0,data_size,batch_size):
        if i+batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i+batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i+batch_size]]
            batch_data_vi = [d[2] for d in data_pairs[i:i+batch_size]]
            batch_data_im = torch.from_numpy(data_im[i:i+batch_size])
            batch_data_bta_im = torch.from_numpy(data_bta_im[i:i+batch_size])
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]
            batch_data_vi = [d[2] for d in data_pairs[i:data_size]]
            batch_data_im = torch.from_numpy(data_im[i:data_size])
            batch_data_bta_im = torch.from_numpy(data_bta_im[i:i+batch_size])

        # The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        # Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []
        batch_vi_pad = []

        # Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens, x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        # Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]


        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        # Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens, vi in zip(batch_data_y, batch_data_vi):
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens, y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)

            vi_pad_seq = pad_seq(vi,y_length)
            batch_vi_pad.append(vi_pad_seq)
        # Reorder the lengths
        batch_vi_pad_sorted =[batch_vi_pad[i] for i in x_reverse_sorted_index]
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        # Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        batch_bta_im_sorted = torch.zeros_like(batch_data_bta_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
            batch_bta_im_sorted[i] = batch_data_bta_im[x]
        
        # Generate batch_x and batch_y
        batch_x, batch_y, batch_vi = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(torch.LongTensor(batch_y_pad_sorted)),Variable(torch.FloatTensor(batch_vi_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        batch_bta_im = Variable(batch_bta_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_vi = batch_vi.cuda()
            batch_im = batch_im.cuda()
            batch_bta_im = batch_bta_im.cuda()

        #Yield the batch data|
        yield batch_x, batch_y, batch_vi, batch_im, batch_bta_im, list(reversed(sorted(batch_x_lengths))), batch_y_lengths_sorted


def data_generator_tl_mtv(data_pairs, data_im, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    # Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    # Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths,batch_size)

    # Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        #Get the corresponding image as well
        batch_data_im = torch.from_numpy(data_im[bidx])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)

        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
        
        #Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
        
        #Yield the batch data|
        yield batch_x, batch_y,batch_im, list(reversed(sorted(batch_x_lengths))), batch_y_lengths_sorted


def data_generator_tl_mtv_vi(data_pairs, data_im, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    #Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    #Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths, batch_size)

    #Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        batch_data_vi = [d[2] for d in [data_pairs[y] for y in bidx]]

        #Get the corresponding image as well
        batch_data_im = torch.from_numpy(data_im[bidx])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        vi_length = max([len(vi) for vi in batch_data_vi])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []
        batch_vi_pad = []
        batch_vi_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens, x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)

        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)

        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        #Pad data_vi and reorder it with respect to the x_reverse_sorted_index
        for vi in batch_data_vi:
            vi_l = len(vi)
            vi_pad_seq = pad_seq(vi,vi_length)
            batch_vi_lengths.append(vi_l)
            batch_vi_pad.append(vi_pad_seq)

        #Reorder the lengths
        batch_vi_pad_sorted =[batch_vi_pad[i] for i in x_reverse_sorted_index]
        batch_vi_lengths_sorted = [batch_vi_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]


        #Generate batch_x and batch_y
        batch_x, batch_y, batch_vi = Variable(torch.LongTensor(batch_x_pad_sorted)),\
            Variable(torch.LongTensor(batch_y_pad_sorted)),\
            Variable(torch.FloatTensor(batch_vi_pad_sorted))
        
        batch_im = Variable(batch_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_vi = batch_vi.cuda()
            batch_im = batch_im.cuda()
        
        #Yield the batch data|
        yield batch_x,batch_y,batch_vi,batch_im,list(reversed(sorted(batch_x_lengths))),batch_y_lengths_sorted


def data_generator_tl_mtv_bta_vi(data_pairs, data_im, data_bta_im, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    #Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    #Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths,batch_size)

    #Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        batch_data_vi = [d[2] for d in [data_pairs[y] for y in bidx]]

        #Get the corresponding image as well
        batch_data_im = torch.from_numpy(data_im[bidx])
        batch_data_bta_im = torch.from_numpy(data_bta_im[bidx])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        vi_length = max([len(vi) for vi in batch_data_vi])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []
        batch_vi_pad = []
        batch_vi_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        #Pad data_vi and reorder it with respect to the x_reverse_sorted_index
        for vi in batch_data_vi:
            vi_l = len(vi)
            vi_pad_seq = pad_seq(vi,vi_length)
            batch_vi_lengths.append(vi_l)
            batch_vi_pad.append(vi_pad_seq)
        #Reorder the lengths
        batch_vi_pad_sorted =[batch_vi_pad[i] for i in x_reverse_sorted_index]
        batch_vi_lengths_sorted = [batch_vi_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        batch_im_bta_sorted = torch.zeros_like(batch_data_bta_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
            batch_im_bta_sorted[i] = batch_data_bta_im[x]

        #Generate batch_x and batch_y
        batch_x, batch_y, batch_vi = Variable(torch.LongTensor(batch_x_pad_sorted)),\
            Variable(torch.LongTensor(batch_y_pad_sorted)),\
            Variable(torch.FloatTensor(batch_vi_pad_sorted))
        
        batch_im = Variable(batch_im_sorted.float())
        batch_bta_im = Variable(batch_im_bta_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_vi = batch_vi.cuda()
            batch_im = batch_im.cuda()
            batch_bta_im = batch_bta_im.cuda()

        #Yield the batch data|
        yield batch_x,\
            batch_y,\
            batch_vi,\
            batch_im,\
            batch_bta_im,\
            list(reversed(sorted(batch_x_lengths))),\
            batch_y_lengths_sorted


def data_generator_tl_mtv_imretrieval(data_pairs, data_im, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    #Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    #Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths, batch_size)

    #Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        #print(bidx)
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]
        #Get the corresponding image as well
        batch_data_im = torch.from_numpy(data_im[bidx])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
        
        #Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
        index_retrieval = [bidx[x] for x in x_reverse_sorted_index]
        #print(index_retrieval)

        #Yield the batch data|
        yield batch_x, batch_y, batch_im, list(reversed(sorted(batch_x_lengths))), index_retrieval


def data_generator_tl_mtv_imretrieval_bta(data_pairs, data_im, data_bta_im, batch_size):
    """
    This is an implementation of generating batches such that the target sentences always have 
    the same length. We borrow the bucket sampler from nmtpytorch to generate the corresponding index,
    such that we can have this corresponding data_pair.
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index. 
        batch_size: The size of the batch
        data_im: The numpy matrix which contains the image features. Size: (N,I), N is the number of samples and I is the image feature size
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length. 
    """
    #Get the lengths of the target language
    tl_lengths = [len(x[1]) for x in data_pairs]

    #Initialize the index sampler
    data_sampler = BucketBatchSampler(tl_lengths,batch_size)

    #Iterate through the index sampler
    for bidx in data_sampler.__iter__():
        batch_data_x = [d[0] for d in [data_pairs[y] for y in bidx]]
        batch_data_y = [d[1] for d in [data_pairs[y] for y in bidx]]

        #Get the corresponding image as well
        batch_data_im = torch.from_numpy(data_im[bidx])
        batch_data_bta_im = torch.from_numpy(data_bta_im[bidx])

        #The lengths for data and labels to be padded to 
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])
        
        #Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        #Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens,x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)

        #Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]              
        #print(x_reverse_sorted_index)

        #Pad data_y and reorder it with respect to the x_reverse_sorted_index
        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens,y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        #Reorder the lengths
        batch_y_pad_sorted =[batch_y_pad[i] for i in x_reverse_sorted_index]
        batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index] 

        
        #Reorder the image numpy matrix with respect to the x_reverse_sorted_index
        batch_im_sorted = torch.zeros_like(batch_data_im)
        batch_bta_im_sorted = torch.zeros_like(batch_data_bta_im)
        for i,x in enumerate(x_reverse_sorted_index):
            batch_im_sorted[i] = batch_data_im[x]
            batch_bta_im_sorted[i] = batch_data_bta_im[x]
        
        #Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(torch.LongTensor(batch_y_pad_sorted))
        batch_im = Variable(batch_im_sorted.float())
        batch_bta_im = Variable(batch_bta_im_sorted.float())
        
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_im = batch_im.cuda()
            batch_bta_im = batch_bta_im.cuda()
        index_retrieval = [bidx[x] for x in x_reverse_sorted_index]

        #Yield the batch data|
        yield batch_x, batch_y, batch_im, batch_bta_im,list(reversed(sorted(batch_x_lengths))), index_retrieval


def translation_reorder(translation, length_sorted_index, id2word):
    #Reorder translation
    original_translation = [None] * len(translation)
    for i,t in zip(length_sorted_index, translation):
        original_translation[i] = [id2word.get(x, '<unk>') for x in t]
    return original_translation


def translation_reorder_BPE(translation, length_sorted_index, id2word):
    #Reorder translation
    original_translation = [None] * len(translation)
    for i,t in zip(length_sorted_index, translation):
        BPE_translation_tokens = [id2word.get(x,'<unk>') for x in t]
        #Processing the original translation such that
        BPE_translation = ' '.join(BPE_translation_tokens)
        #Search and Replace patterns

        ori_translation = re.sub(r'@@ ',"",BPE_translation)
        #Tokenlize the ori_translation and keep it in the orginal_translation list
        original_translation[i] = ori_translation.split()

    return original_translation


def translation_reorder_ATTN(attns, length_sorted_index):
    #Reorder attention
    original_attn = np.zeros(attns.shape)
    for i,attn in zip(length_sorted_index, attns):
        original_attn[i] = attn
    return original_attn
