#coding:utf-8
import os, time, ipdb
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import timeit, load_pkl, save_pkl, str2bool
from data_processor import DataProcessor
from gensim.models import KeyedVectors



class BasicCNN(object):
    """docstring for BasicCNN"""
    def __init__(self, sess, args):
        self.sess = sess
        self.dis_dim = args.dis_dim
        self.pos_dim = args.pos_dim
        self.num_pos = args.num_pos
        self.word_dim = args.word_dim
        self.num_words = args.num_words
        self.num_actions = args.num_actions
        self.num_filters = args.num_filters
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.use_cnn = args.use_cnn

        if args.use_ngrams:
            self.num_words = 2 * args.num_grams - 1 
        
        self.train_mode = args.train_mode
        self.agent_name = args.agent_name
        if args.train_mode == 'eas':
            self.init_distance_emb = np.zeros([self.num_words, self.dis_dim], dtype=np.float32)
            for i in xrange(self.num_words):
                self.init_distance_emb[i] = i
            #self.pre_trained_pos_embedding = load_pkl('data/pos_emb.pkl')
            self.build_eas_model()
            #self.embedding.assign(self.pre_trained_pos_embedding)
        else:
            self.agent_name = args.agent_name
            self.hist_len = args.hist_len
            self.beta_dim = args.state_dim 
            self.alpha_dim = args.state_dim + args.image_padding * 2
            self.build_grid_model()
        

    def conv2d(self, x, output_dim, kernel_size, stride, initializer, 
                activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):
        with tf.variable_scope(name):
            # data_format = 'NHWC'
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

            w = tf.get_variable('w', kernel_size, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding)

            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(conv, b)

            if activation_fn != None:
                out = activation_fn(out)

            return out, w, b


    def max_pooling(self, x, kernel_size, stride, padding='VALID', name='max_pool'):
        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [1, kernel_size[0], kernel_size[1], 1]
            return tf.nn.max_pool(x, kernel_size, stride, padding)


    def linear(self, x, output_dim, activation_fn=None, name='linear'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [x.get_shape()[-1], output_dim], tf.float32,
                initializer=tf.truncated_normal_initializer(0.0, 0.1))
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(tf.matmul(x, w), b)

            if activation_fn != None:
                out = activation_fn(out)

            return out, w, b


    def build_grid_model(self):
        layers = []
        self.weights = {}
        kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        self.states_alpha = tf.placeholder(tf.float32, [None, self.alpha_dim, self.alpha_dim, self.hist_len], 'states_alpha')
        self.states_beta = tf.placeholder(tf.float32, [None, self.beta_dim, self.beta_dim, self.hist_len], 'states_beta')

        if self.agent_name == 'alpha':
            l1, self.weights['conv_w'], self.weights['conv_b'] = self.conv2d(self.states_alpha,
                self.num_filters, (3, 3), (1, 1), kernel_initializer, padding='SAME', name='conv2d')
            l1_shape = l1.get_shape().as_list()
            l1_flat = tf.reshape(l1, [-1, reduce(lambda x, y: x * y, l1_shape[1:])])
            layers.extend([l1, l1_flat])
        
        elif self.agent_name == 'beta':
            l1, self.weights['conv_w'], self.weights['conv_b'] = self.conv2d(self.states_beta,
                self.num_filters, (3, 3), (1, 1), kernel_initializer, padding='SAME', name='conv2d')
            l1_shape = l1.get_shape().as_list()
            l1_flat = tf.reshape(l1, [-1, reduce(lambda x, y: x * y, l1_shape[1:])])
            layers.extend([l1, l1_flat])
        
        else:
            l_alpha, self.weights['conv_alpha_w'], self.weights['conv_alpha_b'] = self.conv2d(self.states_alpha,
                self.num_filters, (3, 3), (1, 1), kernel_initializer, padding='SAME', name='conv2d_alpha')
            l_alpha_shape = l_alpha.get_shape().as_list()
            l_alpha_flat = tf.reshape(l_alpha, [-1, reduce(lambda x, y: x * y, l_alpha_shape[1:])])

            l_beta, self.weights['conv_beta_w'], self.weights['conv_beta_b'] = self.conv2d(self.states_beta,
                self.num_filters, (3, 3), (1, 1), kernel_initializer, padding='SAME', name='conv2d_beta')
            l_beta_shape = l_beta.get_shape().as_list()
            l_beta_flat = tf.reshape(l_beta, [-1, reduce(lambda x, y: x * y, l_beta_shape[1:])])
            l1_flat = tf.concat([l_alpha_flat, l_beta_flat], axis=-1)
            layers.extend([l_alpha, l_alpha_flat, l_beta, l_beta_flat, l1_flat])

        l2, self.weights['l2_w'], self.weights['l2_b'] = self.linear(l1_flat, 256, tf.nn.relu, name='dense')
        self.logits, self.weights['q_w'], self.weights['q_b'] = self.linear(l2, self.num_actions, name='output')
        # print the shape of each layer
        layers.extend([l2, self.logits])
        for layer in layers:
            print(layer.shape)

        self.targets = tf.placeholder(tf.int32, [None], 'labels')
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits))
        self.trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()



    def build_eas_model(self):
        #ipdb.set_trace()
        self.embedding = tf.get_variable('pos_emb', [self.num_pos, self.pos_dim], tf.float32)
        #self.embedding = tf.placeholder(tf.float32, [self.num_pos, self.pos_dim], 'embedding')
        self.word_emb = tf.placeholder(tf.float32, [None, self.num_words, self.word_dim], 'word_emb')
        self.current_word = tf.placeholder(tf.float32, [None, self.word_dim], 'current_word')
        self.word_vec = tf.expand_dims(self.current_word, axis=1)

        self.masks = tf.placeholder(tf.float32, [None, self.num_words, 1], 'mask')
        self.dis_emb = tf.placeholder(tf.float32, [self.num_words, self.dis_dim], 'dis_emb')
        self.dis_inds = tf.placeholder(tf.int32, [None, self.num_words], 'dis_ind')
        self.dis_matrix = tf.nn.embedding_lookup(self.dis_emb, self.dis_inds)

        self.pos_index = tf.placeholder(tf.int32, [None, 1], 'index')
        self.pos_indices = tf.placeholder(tf.int32, [None, self.num_words], 'indices')
        self.pos_vector = tf.nn.embedding_lookup(self.embedding, self.pos_index)
        self.pos_matrix = tf.nn.embedding_lookup(self.embedding, self.pos_indices)
        #self.pos_matrix = tf.multiply(self.masks, self.pre_pos_matrix)
        
        # [batch_size, num_words, 1] <== [batch_size, num_words, pos_dim] * [batch_size, pos_dim, 1]
        #self.attention = tf.matmul(self.pos_matrix, self.pos_vector, transpose_b=True)
        #self.attention = tf.nn.softmax(tf.matmul(self.pos_matrix, self.pos_vector, transpose_b=True))
        # [batch_size, num_words, pos_dim] <== [batch_size, num_words, 1] .* [batch_size, num_words, pos_dim]
        #self.pos_input = tf.expand_dims(tf.multiply(self.attention, self.pos_matrix), -1)
        #self.pos_input = tf.reduce_sum(tf.multiply(self.attention, self.pos_matrix), axis=1)
        self.pre_pos_input = tf.concat([self.pos_matrix, self.dis_matrix], axis=-1)
        self.pos_input = tf.multiply(self.masks, self.pre_pos_input)

        w = {}
        layers = []
        #ipdb.set_trace()
        if self.use_cnn:
            if self.agent_name == 'alpha':
                self.input = tf.expand_dims(self.pos_input, -1)
            elif self.agent_name == 'beta':
                self.input = tf.expand_dims(self.word_emb, -1)
            else:
                self.pre_input = tf.multiply(self.masks, tf.concat([self.word_emb, self.pre_pos_input], axis=-1))
                self.input = tf.expand_dims(self.pre_input, -1)

            ngrams = []
            kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
            for n in [2, 3, 4, 5]: # bigram to five-gram
                conv, w['conv%d_w'%n], w['conv%d_b'] = self.conv2d( self.input, 
                                                                    self.num_filters, 
                                                                    (n, self.input.shape[2]),
                                                                    (1, 1),
                                                                    kernel_initializer,
                                                                    name='conv%d'%n)
                pool = self.max_pooling(conv, (self.num_words - n + 1, 1), (1, 1), name='max_pool%d'%n)
                ngrams.append(pool)
            
            l1 = tf.concat(ngrams, axis=3)
            l1_shape = l1.get_shape().as_list()
            l2 = tf.reshape(l1, [-1, reduce(lambda x, y: x * y, l1_shape[1:])])
            layers.extend([self.input, conv, pool])
        else:
            if self.agent_name == 'alpha':
                self.attention = tf.nn.softmax(tf.matmul(self.pos_matrix, self.pos_vector, transpose_b=True))
                self.input = tf.reduce_sum(tf.multiply(self.attention, self.pre_pos_input), axis=1)
            elif self.agent_name == 'beta':
                self.attention = tf.nn.softmax(tf.matmul(self.word_emb, self.word_vec, transpose_b=True))
                self.input = tf.reduce_sum(tf.multiply(self.attention, self.word_emb), axis=1)
            else:
                self.pre_input = tf.concat([self.word_emb, self.pos_matrix], axis=-1)
                self.all_input = tf.concat([self.word_emb, self.pos_matrix, self.dis_matrix], axis=-1)
                self.word_and_pos = tf.concat([self.word_vec, self.pos_vector], axis=-1)
                self.attention = tf.nn.softmax(tf.matmul(self.pre_input, self.word_and_pos, transpose_b=True))
                self.input = tf.reduce_sum(tf.multiply(self.attention, self.all_input), axis=-1)
            l2 = self.input
            layers.append(self.attention)
        #l2 = self.pos_input
        l3, w['dense_w'], w['dense_b'] = self.linear(l2, l2.shape[-1], activation_fn=tf.nn.relu, name='dense')
        self.logits, w['out_w'], w['out_b'] = self.linear(l3, self.num_actions, name='output')
        for layer in layers + [l2, l3, self.logits]:
            print(layer.shape)

        self.weights = w
        self.targets = tf.placeholder(tf.int32, [None], 'labels')
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits))
        self.trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()


    #@timeit
    def train(self, data):
        #ipdb.set_trace()
        losses = 0.0
        y_pred = []

        if self.train_mode == 'eas':
            pos_index, pos_indices, text_matrix, word_vec, dis_inds, labels, masks = data
            num_batchs = len(labels) // self.batch_size
            for i in xrange(num_batchs):
                batch_index = pos_index[i * self.batch_size: (i + 1) * self.batch_size]
                batch_indices = pos_indices[i * self.batch_size: (i + 1) * self.batch_size]
                batch_dis_inds = dis_inds[i * self.batch_size: (i + 1) * self.batch_size]
                batch_labels = labels[i * self.batch_size: (i + 1) * self.batch_size]
                batch_masks = masks[i * self.batch_size: (i + 1) * self.batch_size]
                batch_text_matrix = text_matrix[i * self.batch_size: (i + 1) * self.batch_size]
                bathc_word_vec = word_vec[i * self.batch_size: (i + 1) * self.batch_size]

                _, loss, z = self.sess.run([self.trainer, self.loss, self.logits],
                                           {self.pos_index: batch_index,
                                            self.pos_indices: batch_indices,
                                            self.targets: batch_labels,
                                            self.masks: batch_masks,
                                            self.dis_inds: batch_dis_inds,
                                            self.dis_emb: self.init_distance_emb,
                                            self.word_emb: batch_text_matrix,
                                            self.current_word: bathc_word_vec,
                                            #self.embedding: self.pre_trained_pos_embedding
                                            }) 
                pred = np.argmax(z, axis=1)
                losses += loss
                y_pred.extend(list(pred))
            
            f1 = compute_f1(y_pred, labels)
            losses /= num_batchs
            return losses, f1 

        else:
            #ipdb.set_trace()
            states_alpha = data['alpha']['states']
            states_beta = data['beta']['states']
            actions_alpha = data['full'] # data['alpha']['actions'] # use combined actions (16 actions)
            actions_beta = data['full'] # data['beta']['actions'] # use combined actions (16 actions)
            actions_full = data['full']
            num_batchs = len(states_alpha) // self.batch_size
            if self.agent_name == 'alpha':
                actions = actions_alpha
                for i in xrange(num_batchs):
                    batch_states_a = states_alpha[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_actions_a = actions_alpha[i * self.batch_size: (i + 1) * self.batch_size]
                    
                    _, loss, z = self.sess.run([self.trainer, self.loss, self.logits],
                                               {self.states_alpha: np.array(batch_states_a, dtype=np.float32),
                                                self.targets: np.array(batch_actions_a, dtype=np.int32)
                                               })
                    pred = np.argmax(z, axis=1)
                    losses += loss
                    y_pred.extend(list(pred))

            elif self.agent_name == 'beta':
                actions = actions_beta
                for i in xrange(num_batchs):
                    batch_states_b = states_beta[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_actions_b = actions_beta[i * self.batch_size: (i + 1) * self.batch_size]
                    
                    _, loss, z = self.sess.run([self.trainer, self.loss, self.logits],
                                               {self.states_beta: np.array(batch_states_b, dtype=np.float32),
                                                self.targets: np.array(batch_actions_b, dtype=np.int32)
                                               })
                    pred = np.argmax(z, axis=1)
                    losses += loss
                    y_pred.extend(list(pred))

            else:
                actions = actions_full
                for i in xrange(num_batchs):
                    batch_states_a = states_alpha[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_states_b = states_beta[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_actions_f = actions_full[i * self.batch_size: (i + 1) * self.batch_size]
                    
                    _, loss, z = self.sess.run([self.trainer, self.loss, self.logits],
                                               {self.states_alpha: np.array(batch_states_a, dtype=np.float32),
                                                self.states_beta: np.array(batch_states_b, dtype=np.float32),
                                                self.targets: np.array(batch_actions_f, dtype=np.int32)
                                               })
                    pred = np.argmax(z, axis=1)
                    losses += loss
                    y_pred.extend(list(pred))
                
            acc = compute_accuracy(y_pred, actions)
            losses /= num_batchs
            return losses, acc



    def test(self, data):
        y_pred = []
        if self.train_mode == 'eas':
            pos_index, pos_indices, text_matrix, word_vec, dis_inds, labels, masks = data
            num_batchs = len(labels) // self.batch_size
            for i in xrange(num_batchs):
                batch_index = pos_index[i * self.batch_size: (i + 1) * self.batch_size]
                batch_indices = pos_indices[i * self.batch_size: (i + 1) * self.batch_size]
                batch_dis_inds = dis_inds[i * self.batch_size: (i + 1) * self.batch_size]
                batch_labels = labels[i * self.batch_size: (i + 1) * self.batch_size]
                batch_masks = masks[i * self.batch_size: (i + 1) * self.batch_size]
                batch_text_matrix = text_matrix[i * self.batch_size: (i + 1) * self.batch_size]
                bathc_word_vec = word_vec[i * self.batch_size: (i + 1) * self.batch_size]

                z = self.logits.eval({  self.pos_index: batch_index,
                                        self.pos_indices: batch_indices,
                                        self.masks: batch_masks,
                                        self.dis_inds: batch_dis_inds,
                                        self.dis_emb: self.init_distance_emb,
                                        self.word_emb: batch_text_matrix,
                                        self.current_word: bathc_word_vec,
                                        #self.embedding: self.pre_trained_pos_embedding
                                        }) 
                pred = np.argmax(z, axis=1)
                y_pred.extend(list(pred))
            
            f1 = compute_f1(y_pred, labels)
            return f1 

        else:
            states_alpha = data['alpha']['states']
            states_beta = data['beta']['states']
            actions_alpha = data['full'] # data['alpha']['actions'] # use combined actions (16 actions)
            actions_beta = data['full'] # data['beta']['actions'] # use combined actions (16 actions)
            actions_full = data['full']
            num_batchs = len(states_alpha) // self.batch_size
            if self.agent_name == 'alpha':
                actions = actions_alpha
                for i in xrange(num_batchs):
                    batch_states_a = states_alpha[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_actions_a = actions_alpha[i * self.batch_size: (i + 1) * self.batch_size]
                    
                    z = self.logits.eval({self.states_alpha: np.array(batch_states_a, dtype=np.float32)})
                    pred = np.argmax(z, axis=1)
                    y_pred.extend(list(pred))

            elif self.agent_name == 'beta':
                actions = actions_beta
                for i in xrange(num_batchs):
                    batch_states_b = states_beta[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_actions_b = actions_beta[i * self.batch_size: (i + 1) * self.batch_size]
                    
                    z = self.logits.eval({self.states_beta: np.array(batch_states_b, dtype=np.float32)})
                    pred = np.argmax(z, axis=1)
                    y_pred.extend(list(pred))

            else:
                actions = actions_full
                for i in xrange(num_batchs):
                    batch_states_a = states_alpha[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_states_b = states_beta[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_actions_f = actions_full[i * self.batch_size: (i + 1) * self.batch_size]
                    
                    z = self.logits.eval({  self.states_alpha: np.array(batch_states_a, dtype=np.float32),
                                            self.states_beta: np.array(batch_states_b, dtype=np.float32),
                                            })
                    pred = np.argmax(z, axis=1)
                    y_pred.extend(list(pred))
                
            acc = compute_accuracy(y_pred, actions)
            return acc


    def predict(self, s_a, s_b):
        if self.agent_name == 'alpha':
            z = self.logits.eval({self.states_alpha: s_a})
        elif self.agent_name == 'beta':
            z = self.logits.eval({self.states_beta: s_b})
        else:
            z = self.logits.eval({self.states_alpha: s_a, self.states_beta: s_b})

        return np.argmax(z, axis=1)[0]



    def save_weights(self, weight_dir):
        print('Saving weights to %s ...' % weight_dir)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        for name in self.weights:
            save_pkl(self.weights[name].eval(), os.path.join(weight_dir, "%s.pkl" % name))
        print('Success!')


    def load_weights(self, weight_dir):
        print('Loading weights from %s ...' % weight_dir)
        for name in self.weights:
            self.weights[name].eval(load_pkl(os.path.join(weight_dir, "%s.pkl" % name)))
        print('Success!')




def compute_success_rate(net, dpsr, data_flag):
    if data_flag == 'valid': # for validation
        start, end = dpsr.start_valid_dom, dpsr.start_test_dom
    else: # for testing
        start, end = dpsr.start_test_dom, len(dpsr.images)

    pd = dpsr.image_padding
    count_exception = 0
    total = success = 0.0
    min_steps, real_steps = [], []
    #ipdb.set_trace()
    for i in tqdm(xrange(start, end)):
        state = np.zeros(dpsr.padded_state_shape, dtype=np.uint8)
        state[pd: -pd, pd: -pd] = dpsr.images[i, 0]
        paths = dpsr.states_xy[i, 0]

        valid_falg = False
        for p in xrange(len(paths)):
            try:
                path = paths[p, 0]
                path += dpsr.pos_bias
                min_step = len(path) / 2
                if min_step <= 1:
                    continue

                s_alpha = np.zeros([1, dpsr.hist_len, dpsr.state_alpha_dim, dpsr.state_alpha_dim], dtype=np.float32)
                s_beta = np.zeros([1, dpsr.hist_len, dpsr.state_beta_dim, dpsr.state_beta_dim], dtype=np.float32)
                a_xy, b_xy = path[0], path[-1]
                if dpsr.is_valid_pos(a_xy) and dpsr.is_valid_pos(b_xy):
                    valid_falg = True
                    break
            except:
                count_exception += 1
                
        if valid_falg:
            for j in xrange(args.max_steps):
                s_alpha[0, : -1] = s_alpha[0, 1: ]
                s_alpha[0, -1] = state[a_xy[0]-1-pd: a_xy[0]+2+pd, a_xy[1]-1-pd: a_xy[1]+2+pd]
                s_beta[0, : -1] = s_beta[0, 1: ]
                s_beta[0, -1] = state[b_xy[0]-1: b_xy[0]+2, b_xy[1]-1: b_xy[1]+2]

                action = net.predict(np.transpose(s_alpha, (0, 2, 3, 1)), np.transpose(s_beta, (0, 2, 3, 1)))
                act_a, act_b = divmod(action, 4)
                move_a = dpsr.action2move[act_a]
                move_b = dpsr.action2move[act_b]
                new_a_xy = a_xy + move_a
                new_b_xy = b_xy + move_b
                if dpsr.is_valid_pos(a_xy) and state[new_a_xy[0], new_a_xy[1]] != 0:
                    a_xy = new_a_xy
                if dpsr.is_valid_pos(b_xy) and state[new_b_xy[0], new_b_xy[1]] != 0:
                    b_xy = new_b_xy

                if abs(sum(a_xy - b_xy)) <= 1:
                    success += 1
                    break
            total += 1
            min_steps.append(min_step)
            real_steps.append(j)

    min_steps = sum(min_steps)
    real_steps = sum(real_steps)
    succ_rate = success / total if total > 0 else 0.0
    traj_diff = (real_steps - min_steps) * 1.0 / min_steps if min_steps > 0 else 0.0

    return succ_rate, traj_diff, count_exception



def compute_accuracy(y_pred, y_truth):
    right = 0.0
    for i in xrange(len(y_pred)):
        if y_pred[i] == y_truth[i]:
            right += 1.0
    return right / len(y_pred) if right > 0 else 0.0


def compute_f1(y_pred, y_truth):
    rec = pre = f1 = 0.0
    tagged = right = 0
    total = len(y_pred)
    for i in xrange(total):
        if y_pred[i] == 1:
            tagged += 1
            if y_pred[i] == y_truth[i]:
                right += 1.0
    if total > 0:
        rec = right / float(total)
    if tagged > 0:
        pre = right / float(tagged)
    if rec + pre > 0:
        f1 = 2 * pre * rec / (pre + rec)
    return f1



def args_init():
    parser = argparse.ArgumentParser()
    # global arguments
    parser.add_argument("--result_dir",     type=str,       default='distance',     help='')
    parser.add_argument("--train_mode",     type=str,       default='eas',      help='')
    parser.add_argument("--load_indices",   type=str2bool,  default=True,       help='')
    parser.add_argument("--save_model",     type=str2bool,  default=True,       help='')
    parser.add_argument("--epochs",         type=int,       default=200,        help='')
    parser.add_argument("--min_epochs",     type=int,       default=20,         help='')
    parser.add_argument("--early_stop",     type=int,       default=10,         help='')
    
    # preset arguments for eas domain
    parser.add_argument("--domain",         type=str,       default='win2k',    help='')
    parser.add_argument("--use_cnn",        type=str2bool,  default=True,      help='')
    parser.add_argument("--sub_sampling",   type=str2bool,  default=True,       help='')
    parser.add_argument("--use_ngrams",     type=str2bool,  default=True,       help='')
    parser.add_argument("--num_grams",      type=int,       default=5,          help='')
    parser.add_argument("--model_dim",      type=int,       default=50,         help='')
    parser.add_argument("--num_words",      type=int,       default=500,        help='')
    parser.add_argument("--num_actions",    type=int,       default=2,          help='')
    parser.add_argument("--num_filters",    type=int,       default=32,         help='')
    parser.add_argument("--batch_size",     type=int,       default=256,        help='')
    parser.add_argument("--learning_rate",  type=float,     default=0.001,      help='')
    parser.add_argument("--gpu_fraction",   type=float,     default=0.2,        help='')
    
    # preset arguments for grid-world domain
    parser.add_argument("--agent_name",     type=str,       default='full',     help='')
    parser.add_argument("--metric",         type=str,       default='succ',     help='')
    parser.add_argument("--autolen",        type=str2bool,  default=True,       help='')
    parser.add_argument("--image_dim",      type=int,       default=8,          help='')
    parser.add_argument("--state_dim",      type=int,       default=3,          help='')
    parser.add_argument("--hist_len",       type=int,       default=8,          help='')
    parser.add_argument("--image_padding",  type=int,       default=1,          help='')
    parser.add_argument("--max_train_doms", type=int,       default=6400,       help='')
    parser.add_argument("--start_valid_dom",type=int,       default=6400,       help='')
    parser.add_argument("--start_test_dom", type=int,       default=7200,       help='')
    
    args = parser.parse_args()
    return args


@timeit
def train_eas(args):
    args.word2vec = KeyedVectors.load_word2vec_format('data/mymodel-new-5-%d'%args.model_dim, binary=True)
    args.result_dir = 'results/%s_%s_%s_%s.txt' % (args.train_mode, args.domain, args.agent_name, args.result_dir)
    args.weight_dir = 'weights/%s_%s_%s.h5' % (args.train_mode, args.domain, args.agent_name)
    args.word_dim = args.dis_dim = args.pos_dim = args.model_dim

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        data = DataProcessor(args)
        train_data, valid_data, test_data = data.read_eas_texts()
        args.num_pos = len(data.pos_dict) + 1
        net = BasicCNN(sess, args)

        best_result = {'train_f1': 0.0, 'valid_f1': 0.0, 'test_f1': 0.0, 'epoch': -1}
        #ipdb.set_trace()
        with open(args.result_dir, 'w') as outfile:
            print('\n Arguments:')
            outfile.write('\n Arguments:\n')
            for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                print('{}: {}'.format(k, v))
                outfile.write('{}: {}\n'.format(k, v))
            print('\n')
            outfile.write('\n')

            print('Start training ...')
            for epoch in xrange(args.epochs):
                loss, f1 = net.train(train_data)
                valid_f1 = net.test(valid_data)
                
                print('epoch: {}\t loss: {:.6f}\t train_f1: {:.6f}'.format(epoch + 1, loss, f1))
                outfile.write('epoch: {}\t loss: {:.6f}\t train_f1: {:.6f}\n'.format(epoch + 1, loss, f1))

                if valid_f1 > best_result['valid_f1']:
                    test_f1 = net.test(test_data)
                    print('\t\t valid_f1: {:.6f}\t test_f1: {:.6f}'.format(valid_f1, test_f1))
                    outfile.write('\t\t valid_f1: {:.6f}\t test_f1: {:.6f}\n'.format(valid_f1, test_f1))

                    best_result['train_f1'] = f1
                    best_result['valid_f1'] = valid_f1
                    best_result['test_f1'] = test_f1
                    best_result['epoch'] = epoch
                    if args.save_model:
                        net.save_weights(args.weight_dir)

                if epoch >= args.min_epochs and epoch - best_result['epoch'] >= args.early_stop:
                    print('\n-----Early stopping, no improvement after %d epochs-----\n' % args.early_stop)
                    break
            
            print('\n Best result: \n {}\n'.format(best_result))  
            outfile.write('\n Best result: \n {}\n'.format(best_result))



@timeit
def train_grid(args):
    args.num_pos = 37 # the size of part-of-speech dictionary
    args.num_actions = 16 # 4
    args.result_dir = 'results/%s_%s_%d_%s_%s.txt' % (args.train_mode, args.agent_name, args.image_dim, args.metric, args.result_dir)
    args.weight_dir = 'weights/%s_%s_%d_%s.h5' % (args.train_mode, args.agent_name, args.image_dim, args.metric)
    preset_max_steps = {8: 38, 16: 86, 32: 178, 64: 246}
    args.max_steps = preset_max_steps[args.image_dim]
    if args.autolen:
        lens = {8: 2, 16: 4, 32: 8, 64: 16} # length of history
        args.hist_len = lens[args.image_dim]
    # if args.agent_name == 'full':
    #     args.num_actions = 16

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        data = DataProcessor(args)
        train_data, valid_data, test_data = data.load_data()
        net = BasicCNN(sess, args)

        if args.metric == 'acc':
            best_result = {'train_acc': 0.0, 'valid_acc': 0.0, 'test_acc': 0.0, 'epoch': -1}
        else:
            best_result = {'valid_succ': 0.0, 'valid_diff': 0.0, 'test_succ': 0.0, 'test_diff': 0.0, 'epoch': -1}
        #ipdb.set_trace()
        with open(args.result_dir, 'w') as outfile:
            print('\n Arguments:')
            outfile.write('\n Arguments:\n')
            for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                print('{}: {}'.format(k, v))
                outfile.write('{}: {}\n'.format(k, v))
            print('\n')
            outfile.write('\n')

            print('Start training ...')
            if args.metric == 'acc':
                for epoch in xrange(args.epochs):
                    loss, acc = net.train(train_data)
                    valid_acc = net.test(valid_data)
                    test_acc = net.test(test_data)
                    
                    print('epoch: {}\t loss: {:.6f}\t train_acc: {:.6f}\t valid_acc: {:.6f}\t test_acc:{:.6f}'.format(
                            epoch + 1, loss, acc, valid_acc, test_acc))
                    outfile.write('epoch: {}\t loss: {:.6f}\t train_acc: {:.6f}\t valid_acc: {:.6f}\t test_acc:{:.6f}\n'.format(
                            epoch + 1, loss, acc, valid_acc, test_acc))

                    if valid_acc > best_result['valid_acc']:
                        best_result['train_acc'] = acc
                        best_result['valid_acc'] = valid_acc
                        best_result['test_acc'] = test_acc
                        best_result['epoch'] = epoch
                        if args.save_model:
                            net.save_weights(args.weight_dir)

                    if epoch >= args.min_epochs and epoch - best_result['epoch'] >= args.early_stop:
                        print('\n-----Early stopping, no improvement after %d epochs-----\n' % args.early_stop)
                        break
            else:
                for epoch in xrange(args.epochs):
                    loss, acc = net.train(train_data)
                    valid_succ, valid_diff, _ = compute_success_rate(net, data, 'valid')
                    print('epoch: {}\t loss: {:.6f}\t train_acc: {:.6f}'.format(epoch + 1, loss, acc))
                    outfile.write('epoch: {}\t loss: {:.6f}\t train_acc: {:.6f}\n'.format(epoch + 1, loss, acc))

                    if valid_succ > best_result['valid_succ']:
                        test_succ, test_diff, _ = compute_success_rate(net, data, 'test')
                        print('valid_succ: {:.6f}\t valid_diff: {:.6f}\t test_succ: {:.6f}\t test_diff: {:.6f}\n'.format(
                                        valid_succ, valid_diff, test_succ, test_diff))
                        outfile.write('valid_succ: {:.6f}\t valid_diff: {:.6f}\t test_succ: {:.6f}\t test_diff: {:.6f}\n\n'.format(
                                        valid_succ, valid_diff, test_succ, test_diff))
                        best_result['valid_succ'] = valid_succ
                        best_result['valid_diff'] = valid_diff
                        best_result['test_succ'] = test_succ
                        best_result['test_diff'] = test_diff
                        best_result['epoch'] = epoch
                        if args.save_model:
                            net.save_weights(args.weight_dir)


                    if epoch >= args.min_epochs and epoch - best_result['epoch'] >= args.early_stop:
                        print('\n-----Early stopping, no improvement after %d epochs-----\n' % args.early_stop)
                        break

            print('\n Best result: \n {}\n'.format(best_result))  
            outfile.write('\n Best result: \n {}\n'.format(best_result))



if __name__ == '__main__':
    args = args_init()
    if args.train_mode == 'eas':
        train_eas(args)
    else:
        train_grid(args)
