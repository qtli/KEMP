import logging
import os
import torch
import torch.utils.data as data
from collections import defaultdict
import json


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, word2index, args):
        """Reads source and target sequences from txt files."""
        self.word2index = word2index
        self.data = data
        self.args = args
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15,
            'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31}
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]

        inputs = self.preprocess([self.data["context"][index],
                                  self.data["vads"][index],
                                  self.data["vad"][index],
                                  self.data["concepts"][index]])
        item["context"], item["context_ext"], item["context_mask"], item["vads"], item["vad"], \
        item["concept_text"], item["concept"], item["concept_ext"], item["concept_vads"], item["concept_vad"], \
        item["oovs"]= inputs

        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["target_ext"] = self.target_oovs(item["target_text"], item["oovs"])
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"],
                                                                     self.emo_map)  # one-hot and scalor label
        item["emotion_widx"] = self.word2index[item["emotion_text"]]

        return item

    def __len__(self):
        return len(self.data["target"])

    def target_oovs(self, target, oovs):
        ids = []
        for w in target:
            if w not in self.word2index:
                if w in oovs:
                    ids.append(len(self.word2index) + oovs.index(w))
                else:
                    ids.append(self.args.UNK_idx)
            else:
                ids.append(self.word2index[w])
        ids.append(self.args.EOS_idx)
        return torch.LongTensor(ids)

    def process_oov(self, context, concept):  #
        ids = []
        oovs = []
        for si, sentence in enumerate(context):
            for w in sentence:
                if w in self.word2index:
                    i = self.word2index[w]
                    ids.append(i)
                else:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(self.word2index) + oov_num)

        for sentence_concept in concept:
            for token_concept in sentence_concept:
                for c in token_concept:
                    if c not in oovs and c not in self.word2index:
                        oovs.append(c)
        return ids, oovs

    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if anw:
            sequence = [self.word2index[word] if word in self.word2index else self.args.UNK_idx for word in arr] + [self.args.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            context = arr[0]
            context_vads = arr[1]
            context_vad = arr[2]
            concept = [arr[3][l][0] for l in range(len(arr[3]))]
            concept_vads = [arr[3][l][1] for l in range(len(arr[3]))]
            concept_vad = [arr[3][l][2] for l in range(len(arr[3]))]

            X_dial = [self.args.CLS_idx]
            X_dial_ext = [self.args.CLS_idx]
            X_mask = [self.args.CLS_idx]  # for dialogue state
            X_vads = [[0.5, 0.0, 0.5]]
            X_vad = [0.0]

            X_concept_text = defaultdict(list)
            X_concept = [[]]  # 初始值是cls token
            X_concept_ext = [[]]
            X_concept_vads = [[0.5, 0.0, 0.5]]
            X_concept_vad = [0.0]
            assert len(context) == len(concept)

            X_ext, X_oovs = self.process_oov(context, concept)
            X_dial_ext += X_ext

            for i, sentence in enumerate(context):
                X_dial += [self.word2index[word] if word in self.word2index else self.args.UNK_idx for word in sentence]
                spk = self.word2index["[USR]"] if i % 2 == 0 else self.word2index["[SYS]"]
                X_mask += [spk for _ in range(len(sentence))]
                X_vads += context_vads[i]
                X_vad += context_vad[i]

                for j, token_conlist in enumerate(concept[i]):
                    if token_conlist == []:
                        X_concept.append([])
                        X_concept_ext.append([])
                        X_concept_vads.append([0.5, 0.0, 0.5])  # ??
                        X_concept_vad.append(0.0)
                    else:
                        X_concept_text[sentence[j]] += token_conlist[:self.args.concept_num]
                        X_concept.append([self.word2index[con_word] if con_word in self.word2index else self.args.UNK_idx for con_word in token_conlist[:self.args.concept_num]])

                        con_ext = []
                        for con_word in token_conlist[:self.args.concept_num]:
                            if con_word in self.word2index:
                                con_ext.append(self.word2index[con_word])
                            else:
                                if con_word in X_oovs:
                                    con_ext.append(X_oovs.index(con_word) + len(self.word2index))
                                else:
                                    con_ext.append(self.args.UNK_idx)
                        X_concept_ext.append(con_ext)
                        X_concept_vads.append(concept_vads[i][j][:self.args.concept_num])
                        X_concept_vad.append(concept_vad[i][j][:self.args.concept_num])

                        assert len([self.word2index[con_word] if con_word in self.word2index else self.args.UNK_idx for con_word in token_conlist[:self.args.concept_num]]) == len(concept_vads[i][j][:self.args.concept_num]) == len(concept_vad[i][j][:self.args.concept_num])
            assert len(X_dial) == len(X_mask) == len(X_concept) == len(X_concept_vad) == len(X_concept_vads)

            return X_dial, X_dial_ext, X_mask, X_vads, X_vad, \
                   X_concept_text, X_concept, X_concept_ext, X_concept_vads, X_concept_vad, \
                   X_oovs

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]  # one


    def collate_fn(self, batch_data):
        def merge(sequences):  # len(sequences) = bsz
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.ones(len(sequences), max(lengths)).long() ## padding index 1 1=True, in mask means padding.
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
            return padded_seqs, lengths

        def merge_concept(samples, samples_ext, samples_vads, samples_vad):
            concept_lengths = []  # 每个sample的concepts数目
            token_concept_lengths = []  # 每个sample的每个token的concepts数目
            concepts_list = []
            concepts_ext_list = []
            concepts_vads_list = []
            concepts_vad_list = []

            for i, sample in enumerate(samples):
                length = 0  # 记录当前样本总共有多少个concept，
                sample_concepts = []
                sample_concepts_ext = []
                token_length = []
                vads = []
                vad = []

                for c, token in enumerate(sample):
                    if token == []:  # 这个token没有concept
                        token_length.append(0)
                        continue
                    length += len(token)
                    token_length.append(len(token))
                    sample_concepts += token
                    sample_concepts_ext += samples_ext[i][c]
                    vads += samples_vads[i][c]
                    vad += samples_vad[i][c]

                if length > self.args.total_concept_num:
                    value, rank = torch.topk(torch.LongTensor(vad), k=self.args.total_concept_num)

                    new_length = 1
                    new_sample_concepts = [self.args.SEP_idx]  # for each sample
                    new_sample_concepts_ext = [self.args.SEP_idx]
                    new_token_length = []
                    new_vads = [[0.5,0.0,0.5]]
                    new_vad = [0.0]

                    cur_idx = 0
                    for ti, token in enumerate(sample):
                        if token == []:
                            new_token_length.append(0)
                            continue
                        top_length = 0
                        for ci, con in enumerate(token):
                            point_idx = cur_idx + ci
                            if point_idx in rank:
                                top_length += 1
                                new_length += 1
                                new_sample_concepts.append(con)
                                new_sample_concepts_ext.append(samples_ext[i][ti][ci])
                                new_vads.append(samples_vads[i][ti][ci])
                                new_vad.append(samples_vad[i][ti][ci])
                                assert len(samples_vads[i][ti][ci]) == 3

                        new_token_length.append(top_length)
                        cur_idx += len(token)

                    new_length += 1  # for sep token
                    new_sample_concepts = [self.args.SEP_idx] + new_sample_concepts
                    new_sample_concepts_ext = [self.args.SEP_idx] + new_sample_concepts_ext
                    new_vads = [[0.5,0.0,0.5]] + new_vads
                    new_vad = [0.0] + new_vad

                    concept_lengths.append(new_length)  # the number of concepts including SEP
                    token_concept_lengths.append(new_token_length)  # the number of tokens which have concepts
                    concepts_list.append(new_sample_concepts)
                    concepts_ext_list.append(new_sample_concepts_ext)
                    concepts_vads_list.append(new_vads)
                    concepts_vad_list.append(new_vad)
                    assert len(new_sample_concepts) == len(new_vads) == len(new_vad) == len(new_sample_concepts_ext), "The number of concept tokens, vads [*,*,*], and vad * should be the same."
                    assert len(new_token_length) == len(token_length)
                else:
                    length += 1
                    sample_concepts = [self.args.SEP_idx] + sample_concepts
                    sample_concepts_ext = [self.args.SEP_idx] + sample_concepts_ext
                    vads = [[0.5,0.0,0.5]] + vads
                    vad = [0.0] + vad

                    concept_lengths.append(length)
                    token_concept_lengths.append(token_length)
                    concepts_list.append(sample_concepts)
                    concepts_ext_list.append(sample_concepts_ext)
                    concepts_vads_list.append(vads)
                    concepts_vad_list.append(vad)

            if max(concept_lengths) != 0:
                padded_concepts = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len); add 1 for root
                padded_concepts_ext = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len)
                padded_concepts_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(samples), max(concept_lengths), 1) ## padding index 1 (bsz, max_concept_len)
                padded_concepts_vad = torch.FloatTensor([[0.0]]).repeat(len(samples), max(concept_lengths))  ## padding index 1 (bsz, max_concept_len)
                padded_mask = torch.ones(len(samples), max(concept_lengths)).long()  # concept(dialogue) state

                for j, concepts in enumerate(concepts_list):
                    end = concept_lengths[j]
                    if end == 0:
                        continue
                    padded_concepts[j, :end] = torch.LongTensor(concepts[:end])
                    padded_concepts_ext[j, :end] = torch.LongTensor(concepts_ext_list[j][:end])
                    padded_concepts_vads[j, :end, :] = torch.FloatTensor(concepts_vads_list[j][:end])
                    padded_concepts_vad[j, :end] = torch.FloatTensor(concepts_vad_list[j][:end])
                    padded_mask[j, :end] = self.args.KG_idx  # for DIALOGUE STATE

                return padded_concepts, padded_concepts_ext, concept_lengths, padded_mask, token_concept_lengths, padded_concepts_vads, padded_concepts_vad
            else:  # there is no concept in this mini-batch
                return torch.Tensor([]), torch.LongTensor([]), torch.LongTensor([]), torch.BoolTensor([]), torch.LongTensor([]), torch.Tensor([]), torch.Tensor([])

        def merge_vad(vads_sequences, vad_sequences):  # for context
            lengths = [len(seq) for seq in vad_sequences]
            padding_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(vads_sequences), max(lengths), 1)
            padding_vad = torch.FloatTensor([[0.5]]).repeat(len(vads_sequences), max(lengths))

            for i, vads in enumerate(vads_sequences):
                end = lengths[i]  # the length of context
                padding_vads[i, :end, :] = torch.FloatTensor(vads[:end])
                padding_vad[i, :end] = torch.FloatTensor(vad_sequences[i][:end])
            return padding_vads, padding_vad  # (bsz, max_context_len, 3); (bsz, max_context_len)

        def adj_mask(context, context_lengths, concepts, token_concept_lengths):
            '''

            :param self:
            :param context: (bsz, max_context_len)
            :param context_lengths: [] len=bsz
            :param concepts: (bsz, max_concept_len)
            :param token_concept_lengths: [] len=bsz;
            :return:
            '''
            bsz, max_context_len = context.size()
            max_concept_len = concepts.size(1)  # include sep token
            adjacency_size = max_context_len + max_concept_len
            adjacency = torch.ones(bsz, max_context_len, adjacency_size)   ## todo padding index 1, 1=True

            for i in range(bsz):
                # ROOT -> TOKEN
                adjacency[i, 0, :context_lengths[i]] = 0
                adjacency[i, :context_lengths[i], 0] = 0

                con_idx = max_context_len+1       # add 1 because of sep token
                for j in range(context_lengths[i]):
                    adjacency[i, j, j - 1] = 0 # TOEKN_j -> TOKEN_j-1

                    token_concepts_length = token_concept_lengths[i][j]
                    if token_concepts_length == 0:
                        continue
                    else:
                        adjacency[i, j, con_idx:con_idx+token_concepts_length] = 0
                        adjacency[i, 0, con_idx:con_idx+token_concepts_length] = 0
                        con_idx += token_concepts_length
            return adjacency

        batch_data.sort(key=lambda x: len(x["context"]), reverse=True)
        item_info = {}
        for key in batch_data[0].keys():
            item_info[key] = [d[key] for d in batch_data]

        assert len(item_info['context']) == len(item_info['vad'])

        ## dialogue context
        context_batch, context_lengths = merge(item_info['context'])
        context_ext_batch, _ = merge(item_info['context_ext'])
        mask_context, _ = merge(item_info['context_mask'])  # for dialogue state!

        ## dialogue context vad
        context_vads_batch, context_vad_batch = merge_vad(item_info['vads'], item_info['vad'])  # (bsz, max_context_len, 3); (bsz, max_context_len)

        assert context_batch.size(1) == context_vad_batch.size(1)


        ## concepts, vads, vad
        concept_inputs = merge_concept(item_info['concept'],
                                       item_info['concept_ext'],
                                       item_info["concept_vads"],
                                       item_info["concept_vad"])  # (bsz, max_concept_len)
        concept_batch, concept_ext_batch, concept_lengths, mask_concept, token_concept_lengths, concepts_vads_batch, concepts_vad_batch = concept_inputs

        ## adja_mask (bsz, max_context_len, max_context_len+max_concept_len)
        if concept_batch.size()[0] != 0:
            adjacency_mask_batch = adj_mask(context_batch, context_lengths, concept_batch, token_concept_lengths)
        else:
            adjacency_mask_batch = torch.Tensor([])

        ## target response
        target_batch, target_lengths = merge(item_info['target'])
        target_ext_batch, _ = merge(item_info['target_ext'])

        d = {}
        ##input
        d["context_batch"] = context_batch.to(self.args.device)  # (bsz, max_context_len)
        d["context_ext_batch"] = context_ext_batch.to(self.args.device)  # (bsz, max_context_len)
        d["context_lengths"] = torch.LongTensor(context_lengths).to(self.args.device)  # (bsz, )
        d["mask_context"] = mask_context.to(self.args.device)
        d["context_vads"] = context_vads_batch.to(self.args.device)   # (bsz, max_context_len, 3)
        d["context_vad"] = context_vad_batch.to(self.args.device)  # (bsz, max_context_len)

        ##concept
        d["concept_batch"] = concept_batch.to(self.args.device)  # (bsz, max_concept_len)
        d["concept_ext_batch"] = concept_ext_batch.to(self.args.device)  # (bsz, max_concept_len)
        d["concept_lengths"] = torch.LongTensor(concept_lengths).to(self.args.device)  # (bsz)
        d["mask_concept"] = mask_concept.to(self.args.device)  # (bsz, max_concept_len)
        d["concept_vads_batch"] = concepts_vads_batch.to(self.args.device)  # (bsz, max_concept_len, 3)
        d["concept_vad_batch"] = concepts_vad_batch.to(self.args.device)   # (bsz, max_concept_len)
        d["adjacency_mask_batch"] = adjacency_mask_batch.bool().to(self.args.device)

        ##output
        d["target_batch"] = target_batch.to(self.args.device)  # (bsz, max_target_len)
        d["target_ext_batch"] = target_ext_batch.to(self.args.device)
        d["target_lengths"] = torch.LongTensor(target_lengths).to(self.args.device)  # (bsz,)

        ##program
        d["target_emotion"] = torch.LongTensor(item_info['emotion']).to(self.args.device)
        d["emotion_label"] = torch.LongTensor(item_info['emotion_label']).to(self.args.device)  # (bsz,)
        d["emotion_widx"] = torch.LongTensor(item_info['emotion_widx']).to(self.args.device)
        assert d["emotion_widx"].size() == d["emotion_label"].size()

        ##text
        d["context_txt"] = item_info['context_text']
        d["target_txt"] = item_info['target_text']
        d["emotion_txt"] = item_info['emotion_text']
        d["concept_txt"] = item_info['concept_text']
        d["oovs"] = item_info["oovs"]

        return d


def write_config(args):
    if not args.test:
        if not os.path.exists(os.path.join(args.save_path, 'result', args.model)):
            os.makedirs(os.path.join(args.save_path, 'result', args.model))
        with open(os.path.join(args.save_path, 'result', args.model, 'config.txt'),'w') as the_file:
            for k, v in args.__dict__.items():
                if "False" in str(v):
                    pass
                elif "True" in str(v):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k,v))


def flatten(t):
    return [item for sublist in t for item in sublist]


def load_dataset(args):
    print('file: ', args.dataset)
    if os.path.exists(args.dataset):
        print("LOADING empathetic_dialogue")
        with open(args.dataset, 'r') as f:
            [data_tra, data_val, data_tst, vocab] = json.load(f)
    else:
        print("data file not exists !!")

    for i in range(3):
        # print('[situation]:', ' '.join(data_tra['situation'][i]))
        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[concept of context]:')
        for si, sc in enumerate(data_tra['concepts'][i]):
            print('concept of sentence {} : {}'.format(si, flatten(sc[0])))
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")

    print("train length: ", len(data_tra['situation']))
    print("valid length: ", len(data_val['situation']))
    print("test length: ", len(data_tst['situation']))
    return data_tra, data_val, data_tst, vocab


def prepare_data_seq(args, batch_size=32):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset(args)  # read data
    word2index, word2count, index2word, n_words = vocab

    logging.info("Vocab  {} ".format(n_words))

    dataset_train = Dataset(pairs_tra, word2index, args)  # data, word2index, args
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=dataset_train.collate_fn)

    dataset_valid = Dataset(pairs_val, word2index, args)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=dataset_valid.collate_fn)

    dataset_test = Dataset(pairs_tst, word2index, args)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=dataset_test.collate_fn)
    write_config(args)
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)