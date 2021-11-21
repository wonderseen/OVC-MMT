'''
This Program is developed to run pretrained model on a test dataset for OVC.
- export CUDA_VISIBLE_DEVICES=0 python OVC_test.py
'''
import torch
import os
from preprocessing import *
from train import *
from bleu import *
from machine_translation_vision.meteor.meteor import Meteor

use_cuda = torch.cuda.is_available()
assert use_cuda, "PLS use CUDA."

MAX_LENGTH = 80
eval_batch_size = 16
split = 'test'
beam_size = 12

###################### Data Relevant Area #########################
source_language = 'en'
TARGET_LANGUAGE = 'DE' # FR or DE 
target_language = TARGET_LANGUAGE.lower()
data_path  = "data/AmbiguousCOCO/"               # Define the Directory of the Data Path
data_path  = f"data/Multi30K_{TARGET_LANGUAGE}/" # Define the Directory of the Test Data Path
vocab_path = f"data/Multi30K_{TARGET_LANGUAGE}/" # Define the Directory of the vocabulary file
models_order = ['1','2','3']

BPE_dataset_suffix  = ".norm.tok.lc.10000bpe"
dataset_suffix      = ".norm.tok.lc"
dataset_im_suffix   = ".norm.tok.lc.10000bpe_ims"

## Initilalize a Meteor Scorer
Meteor_Scorer = Meteor(target_language)

## Load the test dataset
test_source = load_data(os.path.join(data_path, f"{split}{BPE_dataset_suffix}.{source_language}"))
test_target = load_data(os.path.join(data_path, f"{split}{BPE_dataset_suffix}.{target_language}"))
print(f"The size of Test Source and Test Target is: {len(test_source)} <=> {len(test_target)}")

## Load the original test dataset
test_ori_source = load_data(os.path.join(data_path, f"{split}{dataset_suffix}.{source_language}"))
test_ori_target = load_data(os.path.join(data_path, f"{split}{dataset_suffix}.{target_language}"))

## Create the paired test_data
test_data       = [[x.strip(), y.strip()] for x, y in zip(test_source, test_target)]
test_ori_data   = [[x.strip(), y.strip()] for x, y in zip(test_ori_source, test_ori_target)]

print(f"The size of Test Data after filtering: {len(test_data)}")

## Load the Vocabulary File and Create Word2Id and Id2Word dictionaries for translation
vocab_source = load_data(os.path.join(vocab_path, f"vocab.{source_language}"))
vocab_target = load_data(os.path.join(vocab_path, f"vocab.{target_language}"))

## Construct the source_word2id, source_id2word, target_word2id, target_id2word dictionaries
s_word2id, s_id2word = construct_vocab_dic(vocab_source)
t_word2id, t_id2word = construct_vocab_dic(vocab_target)

print(f"The vocabulary size for soruce language: {len(s_word2id)}")
print(f"The vocabulary size for target language: {len(t_word2id)}")

## Generate Train, Val and Test Indexes pairs
test_data_index = create_data_index(test_data, s_word2id, t_word2id, drop_unk=False)
test_y_ref = [[d[1].split()] for d in test_ori_data]
test_y_ref_meteor = dict((key, [value[1]]) for key, value in enumerate(test_ori_data))

## We first test oracle performance based on the available vocabulary.
oracle_y_pred = []
for ws in test_target:
    new_line = []
    for w in ws.split():
        if w in vocab_target:
            new_line.append(w)
        else:
            new_line.append('none')
    new_line = re.sub(r'@@ ', "", ' '.join(new_line)).split()
    oracle_y_pred.append(new_line)
test_bleu = compute_bleu(test_y_ref, oracle_y_pred)
print(f"oracle bleu: {test_bleu[0]}")


oracle_translations_meteor = dict((key, [' '.join(value)]) for key, value in enumerate(oracle_y_pred))
oracle_meteor = Meteor_Scorer.compute_score(test_y_ref_meteor, oracle_translations_meteor)
print(f"oracle meteor: {oracle_meteor[0]}")

## Load the vision features
test_im_feats     = np.load(os.path.join(data_path, f"{split}{dataset_im_suffix}.npy"))
test_bta_im_feats = np.load(os.path.join(data_path, f"{split}{dataset_im_suffix}_bta_sort.npy"))



max_test_bleu = 0.
max_test_meteor = 0.
for d in models_order:
    ################################################################
    model_path_suffix_suffix = f"saves/Multi30K_OVC_Lm_Lv_{target_language}_{d}"
    model_path_suffix = f"{model_path_suffix_suffix}/nmt_trained_imagine_model_best_" 
    output_path = f"{model_path_suffix_suffix}/prediction" # Directory to save the translation results from a trained model
    metrics = ['METEOR', 'BLEU']
    model_paths = [f"{model_path_suffix}{metric}.pt" for metric in metrics]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for model_path, metric in zip(model_paths, metrics):
        print(f"\n---------- {model_path} -----------")
        ## Load the model
        best_model = torch.load(model_path)
        best_model.vse_imagine.dropout_im_emb = 0.
        best_model.dropout_im_emb = 0.

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


        ## Save the translation prediction to the trained_model_path
        test_prediction_path = os.path.join(output_path, f"test_multimodal_model_prediction.{target_language}.{metric}")
        with open(test_prediction_path, 'w') as f:
            for x in test_translations:
                f.write(' '.join(x) + '\n')


        ## Compute the METEOR Score
        test_translations_meteor = dict((key, [' '.join(value)]) for key, value in enumerate(test_translations))
        test_meteor = Meteor_Scorer.compute_score(test_y_ref_meteor, test_translations_meteor)
        print(f"test meteor of the best {metric} model: {test_meteor[0] * 100.}")
        max_test_meteor = max(max_test_meteor, test_meteor[0] * 100.)


        ## Compute the BLEU Score (orginal implementation of VAG_NMT)
        test_bleu = compute_bleu(test_y_ref, test_translations)
        print(f"test bleu of the best {metric} model: {test_bleu[0] * 100.}")
        max_test_bleu = max(max_test_bleu, test_bleu[0] * 100.)

        ## Compute the BLEU Score
        import subprocess
        status, bleuinfo = subprocess.getstatusoutput('perl scripts/multi-bleu.perl -lc {} < {}'.format(
            f'{data_path}{split}{dataset_suffix}.{target_language}',
            f'{output_path}/test_multimodal_model_prediction.{target_language}.{metric}'))
        bleu = re.findall(r'BLEU = (.*?),', bleuinfo)
        print(f'bleu {bleuinfo}\n')

        del best_model
        torch.cuda.empty_cache()


    print("##############################")
    print(f"max bleu score = {max_test_bleu}")
    print(f"max meteor score = {max_test_meteor}")
