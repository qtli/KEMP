import numpy as np
import torch
import json
import csv
import nltk
nltk.download('punkt')

from ast import literal_eval

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.corpus import wordnet


def clean(sentence, word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4  # speak state id which will be used in word embedding stage
SYS_idx = 5  # listener state id
KG_idx = 6  # concept state id
CLS_idx = 7
SEP_idx = 8

init_index2word = {
        UNK_idx: "[UNK]",
        PAD_idx: "[PAD]",
        EOS_idx: "[EOS]",
        SOS_idx: "[SOS]",
        USR_idx: "[USR]",
        SYS_idx: "[SYS]",
        KG_idx: "[KG]",
        CLS_idx: "[CLS]",
    }

word2index = {str(v): int(k) for k, v in init_index2word.items()}
word2count = {str(v): 1 for k, v in init_index2word.items()}
index2word = init_index2word
n_words = len(word2index)  # Count default tokens

def index_word(word):
    n_words = len(word2index)  # Count default tokens
    if word not in word2index:
        word2index[word] = n_words
        word2count[word] = 1
        index2word[n_words] = word
        n_words += 1
    else:
        word2count[word] += 1

def index_words(sentence):
    for word in sentence:
        index_word(word.strip())

def read_emp_dataset():
    word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}
    train_context = np.load('EmpatheticDialogue/sys_dialog_texts.train.npy', allow_pickle=True)
    train_target = np.load('EmpatheticDialogue/sys_target_texts.train.npy', allow_pickle=True)
    train_emotion = np.load('EmpatheticDialogue/sys_emotion_texts.train.npy', allow_pickle=True)
    train_situation = np.load('EmpatheticDialogue/sys_situation_texts.train.npy', allow_pickle=True)

    dev_context = np.load('EmpatheticDialogue/sys_dialog_texts.dev.npy', allow_pickle=True)
    dev_target = np.load('EmpatheticDialogue/sys_target_texts.dev.npy', allow_pickle=True)
    dev_emotion = np.load('EmpatheticDialogue/sys_emotion_texts.dev.npy', allow_pickle=True)
    dev_situation = np.load('EmpatheticDialogue/sys_situation_texts.dev.npy', allow_pickle=True)

    test_context = np.load('EmpatheticDialogue/sys_dialog_texts.test.npy', allow_pickle=True)
    test_target = np.load('EmpatheticDialogue/sys_target_texts.test.npy', allow_pickle=True)
    test_emotion = np.load('EmpatheticDialogue/sys_emotion_texts.test.npy', allow_pickle=True)
    test_situation = np.load('EmpatheticDialogue/sys_situation_texts.test.npy', allow_pickle=True)

    data_train = {'context': [], 'target': [], 'emotion': [], 'situation': []}
    data_dev = {'context': [], 'target': [], 'emotion': [], 'situation': []}
    data_test = {'context': [], 'target': [], 'emotion': [], 'situation': []}



    for context in train_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            index_words(u)
        data_train['context'].append(u_list)
    for target in train_target:
        target = clean(target, word_pairs)
        data_train['target'].append(target)
        index_words(target)
    for situation in train_situation:
        situation = clean(situation, word_pairs)
        data_train['situation'].append(situation)
        index_words(situation)
    for emotion in train_emotion:
        data_train['emotion'].append(emotion)
    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion']) == len(
        data_train['situation'])

    for context in dev_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            index_words(u)
        data_dev['context'].append(u_list)
    for target in dev_target:
        target = clean(target, word_pairs)
        data_dev['target'].append(target)
        index_words(target)
    for situation in dev_situation:
        situation = clean(situation, word_pairs)
        data_dev['situation'].append(situation)
        index_words(situation)
    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)
    assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion']) == len(data_dev['situation'])

    for context in test_context:
        u_list = []
        for u in context:
            u = clean(u, word_pairs)
            u_list.append(u)
            index_words(u)
        data_test['context'].append(u_list)
    for target in test_target:
        target = clean(target, word_pairs)
        data_test['target'].append(target)
        index_words(target)
    for situation in test_situation:
        situation = clean(situation, word_pairs)
        data_test['situation'].append(situation)
        index_words(situation)
    for emotion in test_emotion:
        data_test['emotion'].append(emotion)
    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(
        data_test['situation'])

    vocab = [word2index, word2count, index2word, len(word2index)]
    print('n_words: ', len(word2index))
    return data_train, data_dev, data_test, vocab


def gen_embeddings(n_words, word2index, emb_dim=300, emb_file='glove.6B.300d.txt'):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(n_words, emb_dim) * 0.01
    print('Embeddings: %d x %d' % (n_words, emb_dim))
    if emb_file is not None:
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        for line in open(emb_file).readlines():
            sp = line.split()
            if(len(sp) == emb_dim + 1):
                if sp[0] in word2index:
                    pre_trained += 1
                    embeddings[word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / n_words))
    return embeddings


def emotion_intensity(NRC, word):
    '''
    Function to calculate emotion intensity (Eq. 1 in our paper)
    :param NRC: NRC_VAD vectors
    :param word: query word
    :return:
    '''
    v, a, d = NRC[word]
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468


def get_concept_dict():
    '''
    Retrieve concepts from ConceptNet using the EmpatheticDialogue tokens as queries
    :return:
    '''
    with open('EmpatheticDialogue/dataset_preproc.json', "r") as f:
        [data_tra, data_val, data_tst, vocab] = json.load(f)
        word2index, word2count, index2word, n_words = vocab

    embeddings = gen_embeddings(n_words, word2index)

    VAD = json.load(open("VAD.json", "r", encoding="utf-8"))  # NRC_VAD
    CN = csv.reader(open("assertions.csv", "r", encoding="utf-8"))  # ConceptNet raw file

    concept_dict = {}
    concept_file = open("ConceptNet.json", "w", encoding="utf-8")

    relation_dict = {}
    rd = open("relation.json", "w", encoding="utf-8")

    for i, row in enumerate(CN):
        if i%1000000 == 0:
            print("Processed {} rows".format(i))
        items = "".join(row).split("\t")
        c1_lang = items[2].split("/")[2]
        c2_lang = items[2].split("/")[2]
        if c1_lang == "en" and c2_lang == "en":
            if len(items) != 5:
                print("concept error!")
            relation = items[1].split("/")[2]
            c1 = items[2].split("/")[3]
            c2 = items[3].split("/")[3]
            c1 = wnl.lemmatize(c1)
            c2 = wnl.lemmatize(c2)
            weight = literal_eval("{" + row[-1].strip())["weight"]

            if weight < 1.0:  # filter tuples where confidence score is smaller than 1.0
                continue
            if c1 in word2index and c2 in word2index and c1 != c2 and c1.isalpha() and c2.isalpha():
                if relation not in word2index:
                    if relation in relation_dict:
                        relation_dict[relation] += 1
                    else:
                        relation_dict[relation] = 0
                c1_vector = torch.Tensor(embeddings[word2index[c1]])
                c2_vector = torch.Tensor(embeddings[word2index[c2]])
                c1_c2_sim = torch.cosine_similarity(c1_vector, c2_vector, dim=0).item()

                v1, a1, d1 = VAD[c1] if c1 in VAD else [0.5, 0.0, 0.5]
                v2, a2, d2 = VAD[c2] if c2 in VAD else [0.5, 0.0, 0.5]
                emotion_gap = 1-(abs(v1-v2) + abs(a1-a2))/2
                # <c1 relation c2>
                if c2 not in stop_words:
                    c2_vad = emotion_intensity(VAD, c2) if c2 in VAD else 0.0
                    # score = c2_vad + c1_c2_sim + (weight - 1) / (10.0 - 1.0) + emotion_gap
                    score = c2_vad + emotion_gap
                    if c1 in concept_dict:
                        concept_dict[c1][c2] = [relation, c2_vad, c1_c2_sim, weight, emotion_gap, score]
                    else:
                        concept_dict[c1] = {}
                        concept_dict[c1][c2] = [relation, c2_vad, c1_c2_sim, weight, emotion_gap, score]
                # reverse relation  <c2 relation c1>
                if c1 not in stop_words:
                    c1_vad = emotion_intensity(VAD, c1) if c1 in VAD else 0.0
                    # score = c1_vad + c1_c2_sim + (weight - 1) / (10.0 - 1.0) + emotion_gap
                    score = c1_vad + emotion_gap
                    if c2 in concept_dict:
                        concept_dict[c2][c1] = [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]
                    else:
                        concept_dict[c2] = {}
                        concept_dict[c2][c1] = [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]

    print("concept num: ", len(concept_dict))
    json.dump(concept_dict, concept_file)

    relation_dict = sorted(relation_dict.items(), key=lambda x: x[1], reverse=True)
    json.dump(relation_dict, rd)


def rank_concept_dict():
    concept_dict = json.load(open("ConceptNet.json", "r", encoding="utf-8"))
    rank_concept_file = open('ConceptNet_VAD_dict.json', 'w', encoding='utf-8')

    rank_concept = {}
    for i in concept_dict:
        # [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]   relation, weight, score
        rank_concept[i] = dict(sorted(concept_dict[i].items(), key=lambda x: x[1][5], reverse=True))  # 根据vad由大到小排序
        rank_concept[i] = [[l, concept_dict[i][l][0], concept_dict[i][l][1], concept_dict[i][l][2], concept_dict[i][l][3], concept_dict[i][l][4], concept_dict[i][l][5]] for l in concept_dict[i]]
    json.dump(rank_concept, rank_concept_file, indent=4)



REMOVE_RELATIONS = ["Antonym", "ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom", "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo", "SymbolOf", "FormOf", "AtLocation", "DerivedFrom", "SymbolOf", "CreatedBy", "Synonym", "MadeOf"]

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def wordCate(word_pos):
    w_p = get_wordnet_pos(word_pos[1])
    if w_p == wordnet.NOUN or w_p == wordnet.ADV or w_p == wordnet.ADJ or w_p == wordnet.VERB:
        return True
    else:
        return False

def read_our_dataset(concept_num=3, total_concept_num=10,):
    with open('EmpatheticDialogue/dataset_preproc.json', "r") as f:
        [data_tra, data_val, data_tst, vocab] = json.load(f)
        word2index, word2count, index2word, n_words = vocab

    VAD = json.load(open("VAD.json", "r", encoding="utf-8"))
    concept = json.load(open("ConceptNet_VAD_dict.json", "r", encoding="utf-8"))

    data_tra['concepts'], data_val['concepts'], data_tst['concepts'] = [], [], []
    data_tra['sample_concepts'], data_val['sample_concepts'], data_tst['sample_concepts'] = [], [], []
    data_tra['vads'], data_val['vads'], data_tst['vads'] = [], [], []  # each sentence's vad vectors
    data_tra['vad'], data_val['vad'], data_tst['vad'] = [], [], []  # each word's emotion intensity
    data_tra['target_vad'], data_val['target_vad'], data_tst['target_vad'] = [], [], []  # each target word's emotion intensity
    data_tra['target_vads'], data_val['target_vads'], data_tst['target_vads'] = [], [], []  # each target word's vad vectors

    # train
    train_contexts = data_tra['context']
    for i, sample in enumerate(train_contexts):
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []  # concepts of each sample
        total_concepts = []
        total_concepts_tid = []
        for j, sentence in enumerate(sample):
            words_pos = nltk.pos_tag(sentence)

            vads.append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
            vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

            sentence_concepts = [
                concept[word] if word in word2index and word not in stop_words and word in concept and wordCate(words_pos[wi]) else []
                for wi, word in enumerate(sentence)]

            sentence_concept_words = []  # for each sentence
            sentence_concept_vads = []
            sentence_concept_vad = []

            for cti, uc in enumerate(sentence_concepts):  # filter concepts of each token, complete their VAD value, select top total_concept_num.
                concept_words = []  # for each token
                concept_vads = []
                concept_vad = []
                if uc != []:  # this token has concepts
                    for c in uc:  # iterate the concept lists [c,r,w] of each token
                        if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:   # remove concpets that are stopwords or not in the dict
                            if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                concept_words.append(c[0])
                                concept_vads.append(VAD[c[0]])
                                concept_vad.append(emotion_intensity(VAD, c[0]))
                                total_concepts.append(c[0])  # all concepts of a sentence
                                total_concepts_tid.append([j,cti])  # the token that each concept belongs to

                    # concept_words = concept_words[:5]
                    # concept_vads = concept_vads[:5]
                    # concept_vad = concept_vad[:5]
                    concept_words = concept_words[:concept_num]
                    concept_vads = concept_vads[:concept_num]
                    concept_vad = concept_vad[:concept_num]

                sentence_concept_words.append(concept_words)
                sentence_concept_vads.append(concept_vads)
                sentence_concept_vad.append(concept_vad)

            sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]
            concepts.append(sentence_concepts)
        data_tra['concepts'].append(concepts)
        data_tra['sample_concepts'].append([total_concepts, total_concepts_tid])
        data_tra['vads'].append(vads)
        data_tra['vad'].append(vad)

    train_targets = data_tra['target']
    for i, target in enumerate(train_targets):
        # each item is the VAD info list of each target token
        data_tra['target_vads'].append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
        data_tra['target_vad'].append([emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
    print("trainset finish.")

    # valid
    valid_contexts = data_val['context']
    for i, sample in enumerate(valid_contexts):
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []
        total_concepts = []
        total_concepts_tid = []

        for j, sentence in enumerate(sample):
            words_pos = nltk.pos_tag(sentence)

            vads.append(
                [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
            vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

            sentence_concepts = [
                concept[word] if word in word2index and word not in stop_words and word in concept and wordCate(
                    words_pos[wi]) else []
                for wi, word in enumerate(sentence)]

            sentence_concept_words = []  # for each sentence
            sentence_concept_vads = []
            sentence_concept_vad = []

            for cti, uc in enumerate(sentence_concepts):
                concept_words = []  # for each token
                concept_vads = []
                concept_vad = []
                if uc != []:
                    for c in uc:
                        if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                            if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                concept_words.append(c[0])
                                concept_vads.append(VAD[c[0]])
                                concept_vad.append(emotion_intensity(VAD, c[0]))

                                total_concepts.append(c[0])
                                total_concepts_tid.append([j,cti])

                    concept_words = concept_words[:concept_num]
                    concept_vads = concept_vads[:concept_num]
                    concept_vad = concept_vad[:concept_num]

                sentence_concept_words.append(concept_words)
                sentence_concept_vads.append(concept_vads)
                sentence_concept_vad.append(concept_vad)

            sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]
            concepts.append(sentence_concepts)

        data_val['concepts'].append(concepts)
        data_tra['sample_concepts'].append([total_concepts, total_concepts_tid])
        data_val['vads'].append(vads)
        data_val['vad'].append(vad)

    valid_targets = data_val['target']
    for i, target in enumerate(valid_targets):
        data_val['target_vads'].append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
        data_val['target_vad'].append([emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
    print('validset finish.')

    # test
    test_contexts = data_tst['context']
    for i, sample in enumerate(test_contexts):
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []
        total_concepts = []
        total_concepts_tid = []
        for j, sentence in enumerate(sample):
            words_pos = nltk.pos_tag(sentence)

            vads.append(
                [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
            vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

            sentence_concepts = [
                concept[
                    word] if word in word2index and word not in stop_words and word in concept and wordCate(
                    words_pos[wi]) else []
                for wi, word in enumerate(sentence)]

            sentence_concept_words = []  # for each sentence
            sentence_concept_vads = []
            sentence_concept_vad = []

            for cti, uc in enumerate(sentence_concepts):
                concept_words = []  # for each token
                concept_vads = []
                concept_vad = []
                if uc != []:
                    for c in uc:
                        if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                            if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                concept_words.append(c[0])
                                concept_vads.append(VAD[c[0]])
                                concept_vad.append(emotion_intensity(VAD, c[0]))

                                total_concepts.append(c[0])
                                total_concepts_tid.append([j,cti])

                    concept_words = concept_words[:concept_num]
                    concept_vads = concept_vads[:concept_num]
                    concept_vad = concept_vad[:concept_num]

                sentence_concept_words.append(concept_words)
                sentence_concept_vads.append(concept_vads)
                sentence_concept_vad.append(concept_vad)

            sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]
            concepts.append(sentence_concepts)

        data_tst['concepts'].append(concepts)
        data_tra['sample_concepts'].append([total_concepts, total_concepts_tid])
        data_tst['vads'].append(vads)
        data_tst['vad'].append(vad)

    test_targets = data_tst['target']
    for i, target in enumerate(test_targets):
        data_tst['target_vads'].append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
        data_tst['target_vad'].append([emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
    print('testset finish.')

    return data_tra, data_val, data_tst, vocab


if __name__ == '__main__':
    # step 1
    data_tra, data_val, data_tst, vocab = read_emp_dataset()  # vocab = [word2index, word2count, index2word, n_words]
    with open('EmpatheticDialogue/dataset_preproc.json', "w") as f:
        json.dump([data_tra, data_val, data_tst, vocab], f)
        print("Saved EmpatheticDialogue/dataset_preproc.json file.")

    # step 2
    get_concept_dict()

    # step 3
    rank_concept_dict()

    # step 4
    data_tra, data_val, data_tst, vocab = read_our_dataset()  # vocab = [word2index, word2count, index2word, n_words]
    with open('kemp_dataset_preproc.json', "w") as f:
        json.dump([data_tra, data_val, data_tst, vocab], f)
        print("Saved kemp_dataset_preproc.json file.")