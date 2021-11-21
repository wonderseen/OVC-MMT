import os
import numpy as np

ambiguous_suffix = '##'
ambiguous_types = ['people', 'scene', 'clothing', 'instruments', 'animals', 'bodyparts', 'vehicles', 'other']
ambiguous_types = [ambiguous_suffix+w for w in ambiguous_types]

msk_color = True

def color_mask(sent):
    sent = ' '+ sent.lower() + ' '
    regular_colors = ['white', 'green', 'yellow', 'black', 'blue', 'orange', 'red', 'gray', 'purple']
    for w in regular_colors:
        sent = sent.replace(' '+w+' ', ' '+ambiguous_suffix+'color'+' ')
    sent = ' '.join(sent.split())
    return sent


def punctuation(sent):
    ws = sent.split()
    for i, w in enumerate(ws):
        if w == '.newly': e
        for p in ['.', ',', '-', '&', '?', '!']:
            if p in w:
                w = w.replace(p, ' '+p+' ')
        ws[i] = w
    return ' '.join(ws)


def remove_noise_word(r):
    for w in [
        'a', 'an', ',', 'one', 'of', 'two', '2', 'three', 'eight', '8', '1', '3', '4', 'four', 'six',
        'several', 'group', 'many', 'five', 'the', 'it', 'that', 'or', 'and', 'there', 'here', 'their',
        'some', 'on', 'in', '4\'', 'last', 'who', 'us', 'her', 'his', 'first', 'what',
        ]:
        if w in r: r.remove(w)

    # for w in ['men', 'man', 'women', 'lady', 'child', 'children', 'woman', 'older man', 'people']:
    #     if w in r:
    #         r.remove(w)
    #         r.append('person')
    return r

def mask_clean(masked_sentence):
    masked_sentence = punctuation(masked_sentence)
    if msk_color:
        masked_sentence = color_mask(masked_sentence)
    return masked_sentence

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue
        sentence = sentence.lower()

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        ambiguous_words = []
        current_phrase = []
        add_to_phrase = False
        add_to_ambiguous_word = False

        # print(sentence)
        for token in sentence.split():
            if add_to_phrase:
                ##
                # mask all the tokens of entity-phrase
                # if add_to_ambiguous_word:
                #     # special cases without visual awareness
                #     if token in ['2', 'of', 'one', 'a', 'group', 'an', 'many', ',',
                #     'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'night', 'several']:
                #         ambiguous_words.append(token.replace(']',''))
                #     else:
                #         if token[-1] == ']':
                #             ambiguous_words.append(ambiguous_suffix+parts[2])


                if token[-1] == ']':
                    add_to_ambiguous_word = False
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)

                    ## only mask the last token of entity-phrase
                    if 'notvisual' in parts[2].lower() or 'other' in parts[2].lower():
                        ambiguous_words.extend(current_phrase)
                    else:
                        ambiguous_words.extend(current_phrase[:-1])
                        ambiguous_words.append(ambiguous_suffix+parts[2])

                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)
                words.append(token)


            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                    if parts[2] != 'notvisual':
                        pass
                    else:
                        add_to_ambiguous_word = True
                else:
                    ambiguous_words.append(token)
                    words.append(token)

        sentence_data = {'sentence' : ' '.join(words), 'phrases' : [], 'ambiguous_words': ' '.join(ambiguous_words)}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type,
                                             })

        annotations.append(sentence_data)

    return annotations

splits = ['train', 'val']#, 'test']

## image list
flicker30k_dir = 'flickr30k_entities/Sentences/'
for root, dirs, files in os.walk(flicker30k_dir):
    flicker30k_image_list = files
    break

def blue1(reference, candicate):
    w_r = reference.split()
    c_r = candicate.split()
    same = 0.
    for c in c_r:
       same += float(c in w_r)
    return same / float(len(c_r))


for split in splits:
    multi30k_en_file = 'data/Multi30K_DE/%s.norm.tok.lc.en' % split
    multi30k_en_file_bpe = 'data/Multi30K_DE/%s.norm.tok.lc.10000bpe.en' % split
    new_multi30k_en_file = 'data/Multi30K_DE/%s.norm.tok.lc.en.ambiguous' % split
    relative_obj_file = 'data/%s_relative_obj.en' % split
    all_relative_obj_file = 'data/%s_all_relative_obj.en' % split
    clean_flicker_file = 'data/%s_clean_flicker_entity.en' % split
    with open(multi30k_en_file, 'r') as f: multi30k_en = [
       l.strip().replace('&apos;s', '\'s').replace('&quot;', '"').replace('&amp; amp ;', '&').replace('&apos; re', '\'re').replace('&apos;', '\'') for l in f.readlines()]
    with open(multi30k_en_file_bpe, 'r') as f: multi30k_en_bpe = [l.strip() for l in f.readlines()]

    with open('data/Multi30K_DE/%s_images.txt' % split, 'r') as f:
       multi30k_image_list = [l.strip().replace('jpg', 'txt') for l in f.readlines()]
       if split == 'test': 
         multi30k_image_list = [l.split('_')[0]+'.txt' for l in multi30k_image_list]

    total_token = 0
    masked_token = 0
    with open(new_multi30k_en_file, 'w') as f:
       with open(relative_obj_file, 'w') as r_f:
           with open(all_relative_obj_file, 'w') as all_r_f:
                with open(clean_flicker_file, 'w') as clean_flicker_f:
                    for i, img in enumerate(multi30k_image_list):
                        assert img in flicker30k_image_list, (split, img)

                        annotations = get_sentence_data(os.path.join(flicker30k_dir, img))
                        clean_flicker_f.write('||'.join([sent['sentence'] for sent in annotations])+'\n')

                        similarity = [blue1(s['sentence'], multi30k_en[i]) for s in annotations]
                        sentence_idx = np.argmax(similarity)
                        s = annotations[sentence_idx]


                        # if 'notvisual' in s['ambiguous_words']: print(s, '\n', s['ambiguous_words'])
                        for w in s['ambiguous_words'].split():
                            if w.startswith(ambiguous_suffix) and w not in ambiguous_types: ambiguous_types.append(w)
                        masked_sentence = s['ambiguous_words']
                        masked_sentence = mask_clean(masked_sentence)
                        f.write(masked_sentence + '\n')


                        total_token += len(masked_sentence.split())
                        masked_token += len([s for s in masked_sentence.split() if s.startswith('##')])
                        

                        # 
                        masked_objects = '||'.join([w['phrase'] for w in s['phrases']]) + '\n'
                        r_f.write(masked_objects)


                        # 
                        roc = []
                        for s in [s]:#annotations:
                            r = ' '.join([w['phrase'] for w in s['phrases']]).split()

                            r = list(set(' '.join([w['phrase'] for w in s['phrases']]).split()))
                            r = remove_noise_word(r)

                            r = list(set(r))

                            roc.extend(r)

                        roc = ' '.join(roc) + '\n'
                        all_r_f.write(roc)

              # if blue1(s['sentence'], multi30k_en[i]) < 0.7: print(s['sentence'], multi30k_en[i])
    print(ambiguous_types)
    print('split =', split)
    print('total token numbers = ', total_token)
    print('masked token numbers =', masked_token)


## write vocab
split = 'train' 
new_multi30k_en_file = 'data/Multi30K_DE/%s.norm.tok.lc.en.ambiguous' % split
new_multi30k_en_vocab_file = 'data/Multi30K_DE/vocab.en.ambiguous'
with open(new_multi30k_en_file, 'r') as f: sent = ' '.join(f.readlines())
vocabs = ['<unk>', '<s>', '</s>'] + list(set(sent.split()))
with open(new_multi30k_en_vocab_file, 'w') as f:
    for v in vocabs:
       f.write(v+'\n')