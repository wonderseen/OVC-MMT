
languages = ['en', 'de', 'fr']
vocab_dirs = ['data/Multi30K_DE/', 'data/Multi30K_DE/', 'data/Multi30K_FR/']


for language, vocab_dir in zip(languages, vocab_dirs):
    with open('data/AmbiguousCOCO/test_2017_mscoco.lc.norm.tok.'+language, 'r') as f:
        coco = [line.strip() for line in f.readlines()]
    with open(vocab_dir + 'vocab.'+language, 'r') as f:
        en_vocab = [(line.strip(), len(line)) for i, line in enumerate(f.readlines())]

    unk = '[unk]'
    en_vocab += [(unk, -1)]
    en_vocab = dict(en_vocab)

    def get_bpe_segment(token):
        for l in range(len(token)-2 if token.endswith('@@') else len(token)):
            word2id = en_vocab.get(token[l:], -1)
            if word2id != -1:
                bpe = []
                bpe.append(token[l:])
                if l != 0:
                    bpe.extend(get_bpe_segment(token[:l]+'@@'))
                return bpe
        return [token]
        


    with open('data/AmbiguousCOCO/test.norm.tok.lc.10000bpe.'+language, 'w') as f:
        for line in coco:
            tokens = line.split()
            bpe_tokens = []
            for token in tokens:
                if token in en_vocab.keys():
                    bpe = token
                else:
                    bpe = get_bpe_segment(token)
                    bpe.reverse()
                    bpe = ' '.join(bpe)
                bpe_tokens.append(bpe)
            f.write(' '.join(bpe_tokens)+'\n')

