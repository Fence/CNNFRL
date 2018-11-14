import os
import ipdb
import json
import random
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from copy import deepcopy
from utils import timeit, load_pkl, save_pkl



class DataProcessor(object):
    """docstring for DataProcessor"""
    def __init__(self, args):
        self.domain = args.domain
        self.word2vec = args.word2vec
        self.word_dim = args.word_dim
        self.num_words = args.num_words
        self.load_indices = args.load_indices
        self.sub_sampling = args.sub_sampling
        self.use_ngrams = args.use_ngrams
        self.num_grams = args.num_grams

        # for grid-world domain
        self.agent_name = args.agent_name
        self.hist_len = args.hist_len   # 8
        self.image_dim = args.image_dim # 32
        self.state_beta_dim = args.state_dim # 3
        self.image_padding = args.image_padding
        self.max_train_doms = args.max_train_doms       # 6400
        self.start_valid_dom = args.start_valid_dom     # 6400
        self.start_test_dom = args.start_test_dom       # 7200

        self.border_start = self.image_padding + 1  # >= 1 
        self.border_end = self.image_dim + self.image_padding - 2  # <= dim + pad - 2
        self.padded_state_shape = (self.image_dim + self.image_padding*2, self.image_dim + self.image_padding*2)
        self.state_alpha_dim = self.state_beta_dim + self.image_padding * 2
        self.pos_bias = np.array([self.image_padding, self.image_padding], dtype=np.uint8)
        self.move = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        self.move2action = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        self.action2move = {0: [0, -1], 1: [0, 1], 2: [-1, 0], 3: [1, 0]}


    def load_data(self):
        #ipdb.set_trace()
        data = sio.loadmat('data/gridworld_o_%d.mat' % self.image_dim)
        self.images = data['all_images']
        self.states_xy = data['all_states_xy_by_domain']

        train_data = {'alpha': {'states': [], 'actions': []}, 'beta': {'states': [], 'actions': []}, 'full': []}
        valid_data = {'alpha': {'states': [], 'actions': []}, 'beta': {'states': [], 'actions': []}, 'full': []}
        test_data = {'alpha': {'states': [], 'actions': []}, 'beta': {'states': [], 'actions': []}, 'full': []}

        self.split_data(0, self.max_train_doms, train_data)
        self.split_data(self.start_valid_dom, self.start_test_dom, valid_data)
        self.split_data(self.start_test_dom, len(self.images), test_data)

        print('train: {}\t valid: {}\t test: {}'.format(
                                                        len(train_data['beta']['states']), 
                                                        len(valid_data['beta']['states']), 
                                                        len(test_data['beta']['states'])))
        return train_data, valid_data, test_data


    def split_data(self, start, end, data):
        pd = self.image_padding
        for i in tqdm(xrange(start, end)):
            state = np.zeros(self.padded_state_shape, dtype=np.uint8)
            state[pd: -pd, pd: -pd] = self.images[i, 0]
            paths = self.states_xy[i, 0]

            for p in xrange(len(paths)):
                actions_alpha, states_alpha, actions_beta, states_beta, actions_full = self.parse_path(paths[p, 0], state)
                data['alpha']['states'].extend(states_alpha)
                data['alpha']['actions'].extend(actions_alpha)
                data['beta']['states'].extend(states_beta)
                data['beta']['actions'].extend(actions_beta)
                data['full'].extend(actions_full)


    def parse_path(self, path, state):
        path += self.pos_bias
        pd = self.image_padding
        states_alpha, states_beta = [], []
        actions_alpha, actions_beta, actions_full = [], [], []

        s_alpha = np.zeros([self.hist_len, self.state_alpha_dim, self.state_alpha_dim], dtype=np.uint8)
        s_beta = np.zeros([self.hist_len, self.state_beta_dim, self.state_beta_dim], dtype=np.uint8)

        l = len(path)
        try:
            for i in xrange((l + 1) / 2):
                a_alpha = self.move2action[tuple(path[i + 1] - path[i])] 
                actions_alpha.append(a_alpha)
                a_beta = self.move2action[tuple(path[l - i - 1] - path[l - i - 2])] 
                actions_beta.append(a_beta)
                a_full = a_alpha * 4 + a_beta # action combination 
                actions_full.append(a_full)

                a_xy = path[i]
                b_xy = path[l - i - 1]

                s_alpha[: -1] = s_alpha[1: ]
                s_alpha[-1] = state[a_xy[0]-1-pd: a_xy[0]+2+pd, a_xy[1]-1-pd: a_xy[1]+2+pd]
                states_alpha.append(np.transpose(s_alpha, (1, 2, 0))) # channel last

                s_beta[: -1] = s_beta[1: ]
                s_beta[-1] = state[b_xy[0]-1: b_xy[0]+2, b_xy[1]-1: b_xy[1]+2]
                states_beta.append(np.transpose(s_beta, (1, 2, 0))) # channel last
        except:
            # some position may be wrong when generating the path with the source matlab code
            return [], [], [], [], []

        return actions_alpha, states_alpha, actions_beta, states_beta, actions_full 


    def is_valid_pos(self, xy):
        # not in the border
        return not (xy[0] > self.border_end or xy[0] < self.border_start or xy[1] > self.border_end or xy[1] < self.border_start)          
        

    @timeit
    def read_eas_texts(self):
        # build the part of speech dictionary
        self.pos_dict = {'UNK': 0}
        for d in ['cooking', 'win2k', 'wikihow']:
            sent_data = json.load(open('data/%s_dependency.json' % d))
            for sents in sent_data:
                for sent in sents:
                    for word, pos in sent:
                        if pos not in self.pos_dict:
                            self.pos_dict[pos] = len(self.pos_dict)

        print('len(pos_dict): %d' % len(self.pos_dict))

        text_data = load_pkl('data/%s_labeled_text_data.pkl' % self.domain)
        sent_data = json.load(open('data/%s_dependency.json' % self.domain))
        if self.domain == 'wikihow': # for wikihow data
            text_data = text_data[:150]
            sent_data = sent_data[:150]
        
        filename = 'data/%s_indices.pkl' % self.domain
        if os.path.exists(filename) and self.load_indices:
            print('Loading indices file from %s' % filename)
            train_indices, valid_indices, test_indices = load_pkl(filename)
        else:
            num_texts = len(text_data)
            piece = int(num_texts / 10)
            indices = range(num_texts)
            random.shuffle(indices)
            train_indices = indices[: piece * 8]
            valid_indices = indices[piece * 8: piece * 9]
            test_indices = indices[piece * 9: ]
            print('Saving indices file to %s' % filename)
            save_pkl([train_indices, valid_indices, test_indices], filename)

        train_data = [[], []]
        valid_data = []
        test_data = []
        text_lengths = []
        if self.use_ngrams:
            context_window = 2 * self.num_grams - 1
            preset_distance = np.arange(context_window, dtype=np.int32)
            preset_distance = np.abs(preset_distance - self.num_grams + 1)
        #ipdb.set_trace()
        print('Preparing training data ...')
        for i in xrange(len(text_data)):
            eas_text = {}
            eas_text['tokens'] = text_data[i]['words']
            self.create_matrix(eas_text, sent_data[i])

            essential_actions = []
            optional_actions = []
            exclusive_actions = []
            for acts in text_data[i]['acts']:
                if acts['act_type'] == 1:
                    essential_actions.append(acts['act_idx'])
                elif acts['act_type'] == 2:
                    optional_actions.append(acts['act_idx'])
                elif acts['act_type'] == 3:
                    if acts['act_idx'] not in exclusive_actions:
                        exclusive_actions.append(acts['act_idx'])
                        exclusive_actions.extend(acts['related_acts'])

            #ipdb.set_trace()
            count_sample = 0
            positive_samples = []
            negative_samples = []
            if self.use_ngrams:
                num_words = len(eas_text['tokens'])
            else:
                num_words = min(len(eas_text['tokens']), self.num_words)
            eas_text['length'] = num_words
            for ind in xrange(num_words):
                text = deepcopy(eas_text) # deepcopy the dict and then modify the elements
                if self.use_ngrams:
                    tmp_ind = ind + self.num_grams - 1
                    text['pos'] = eas_text['all_pos'][tmp_ind]
                    text['distance'] = preset_distance
                    text['all_pos'] = eas_text['all_pos'][ind: ind + context_window]
                    text['text_matrix'] = eas_text['text_matrix'][ind: ind + context_window]
                    text['length'] = context_window
                    text['word_vec'] = eas_text['text_matrix'][tmp_ind]
                else:
                    text['pos'] = eas_text['all_pos'][ind]
                    distance = np.arange(self.num_words, dtype=np.int32)
                    distance = np.abs(distance - ind)
                    text['distance'] = distance
                    text['word_vec'] = eas_text['text_matrix'][ind]

                if ind in essential_actions:
                    count_sample += 1
                    text['label'] = 1 #one_hot(2, 1)
                    positive_samples.append(text)
                
                elif ind in optional_actions or ind in exclusive_actions:
                    count_sample += 1
                    text['label'] = 1 #one_hot(2, 1)
                    positive_samples.append(text)
                    # count_sample += 1
                    # eas_text['label'] = 0 #one_hot(2, 0)
                    # negative_samples.append(deepcopy(eas_text))
                
                else:
                    text['label'] = 0 #one_hot(2, 0)
                    count_sample += 1
                    negative_samples.append(text)
            text_lengths.append(count_sample)

            if i in train_indices:
                train_data[0].extend(positive_samples)
                train_data[1].extend(negative_samples)
            elif i in valid_indices:
                valid_data.extend(positive_samples)
                valid_data.extend(negative_samples)
            else:
                test_data.extend(positive_samples)
                test_data.extend(negative_samples)

        if self.sub_sampling:
            train_data = self.sub_sample_data(train_data)
        else:
            train_data = self.shuffle_data(train_data[0] + train_data[1])

        print('total: {}\t train: {}\t valid: {}\t test: {}'.format(
            sum(text_lengths), len(train_data), len(valid_data), len(test_data)))
        train_data = self.unpack_data(train_data)
        valid_data = self.unpack_data(valid_data)
        test_data = self.unpack_data(test_data)

        return train_data, valid_data, test_data


    def create_matrix(self, text, sents):
        text['all_pos'] = []
        for sent in sents:
            for word, pos in sent:
                text['all_pos'].append(self.pos_dict[pos])
        
        text_matrix = []
        for w in text['tokens']:
            if w in self.word2vec.vocab:
                text_matrix.append(self.word2vec[w])
            else:
                text_matrix.append(np.zeros(self.word_dim))

        text_matrix = np.array(text_matrix)
        if self.use_ngrams:
            word_vector_padding = np.zeros([self.num_grams - 1, self.word_dim])
            text_matrix = np.concatenate([word_vector_padding, text_matrix, word_vector_padding])
            pos_padding = [0] * (self.num_grams - 1)
            text['all_pos'] = pos_padding + text['all_pos'] + pos_padding
            text['text_matrix'] = text_matrix
        else:
            pad_len = self.num_words - len(text_matrix)
            if pad_len > 0:
                text['all_pos'].extend([0] * pad_len)
                text_matrix = np.concatenate((text_matrix, np.zeros([pad_len, self.word_dim])))
            else:
                text['all_pos'] = text['all_pos'][: self.num_words]
                text_matrix = text_matrix[: self.num_words]
            text['text_matrix'] = text_matrix


    def sub_sample_data(self, data):
        #ipdb.set_trace()
        positive_samples, negative_samples = data
        num_samples = min(len(positive_samples), len(negative_samples))
        positive_indices = random.sample(range(len(positive_samples)), num_samples)
        negative_indices = random.sample(range(len(negative_samples)), num_samples)
        positive_samples = [positive_samples[i] for i in positive_indices]
        negative_samples = [negative_samples[i] for i in negative_indices]
        positive_samples.extend(negative_samples)

        return self.shuffle_data(positive_samples)

        

    def shuffle_data(self, data):
        indices = range(len(data))
        random.shuffle(indices)
        all_samples = [data[i] for i in indices]

        return all_samples



    def unpack_data(self, data):
        masks = []
        labels = []
        distances = []
        pos_index = []
        pos_indices = []
        text_matrix = []
        word_vec = []

        for d in data:
            pos_index.append(d['pos'])
            pos_indices.append(d['all_pos'])
            labels.append(d['label'])
            mask = np.zeros([len(d['all_pos']), 1], dtype=np.float32)
            mask[: d['length'], 0] = 1
            masks.append(mask)
            distances.append(d['distance'])
            text_matrix.append(d['text_matrix'])
            word_vec.append(d['word_vec'])

        pos_index = np.array(pos_index, dtype=np.int32)
        pos_index.shape = (len(pos_index), 1)
        #pos_indices = np.array(pos_indices, dtype=np.int32)
        #distances = np.array(distances, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        #masks = np.array(masks, dtype=np.int32)
        #word_vec = np.array(word_vec, dtype=np.float32)

        return pos_index, pos_indices, text_matrix, word_vec, distances, labels, masks

    
    def one_hot(self, dim, label):
        label_vector = np.zeros(dim, dtype=np.uint8)
        label_vector[label] = 1
        
        return label_vector


