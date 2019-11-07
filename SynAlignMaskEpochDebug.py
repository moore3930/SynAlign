from __future__ import absolute_import, division, print_function, unicode_literals
from models import Model
from helper import *
from sklearn.model_selection import train_test_split
from web.embedding import Embedding
from web.evaluate import evaluate_on_all

import tensorflow as tf
import numpy as np
import os
import io
import time

# model
class SynAlign(Model):

    def create_tokenizer(self):
        # creating tokenizer
        self.source_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

        lines = io.open(self.path_to_file, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines]
        inp_text, target_text = zip(*word_pairs)
        self.source_tokenizer.fit_on_texts(inp_text)
        self.target_tokenizer.fit_on_texts(target_text)

        print("size of source tokenizer is {}".format(len(self.source_tokenizer.word_index)))
        print("size of target tokenizer is {}".format(len(self.target_tokenizer.word_index)))

    def batch_process(self, lines):
        # line = line.strip().lower()
        line = [l.strip().lower().split(b'\t') for l in lines]
        try:
            source_text, target_text = zip(*line)
        except:
            print(lines)

        source_text = [line.decode('utf-8') for line in source_text]
        target_text = [line.decode('utf-8') for line in target_text]
        source_text = [preprocess_sentence(line) for line in source_text]
        target_text = [preprocess_sentence(line) for line in target_text]
        # print(source_text)
        # print(target_text)

        source_ids = self.source_tokenizer.texts_to_sequences(source_text)
        target_ids = self.target_tokenizer.texts_to_sequences(target_text)
        # print(source_ids)
        # print(target_ids)

        source_ids = tf.keras.preprocessing.sequence.pad_sequences(source_ids, padding='post')
        target_ids = tf.keras.preprocessing.sequence.pad_sequences(target_ids, padding='post')

        # mask
        source_mask = source_ids > 0
        target_mask = target_ids > 0

        # pos id
        source_max_len = source_mask.sum(1)
        target_max_len = target_mask.sum(1)

        source_pos_ids = np.array([[i for i in range(1, source_ids.shape[1] + 1)]])
        source_pos_ids = np.tile(source_pos_ids, [source_ids.shape[0], 1])
        try:
            source_pos_ids = ((source_pos_ids * 2 * np.pi) / source_max_len[:, np.newaxis]).astype(np.int32)     # [?, n]
        except:
            print(source_max_len)
            print(source_pos_ids)
        source_pos_ids = np.array([[[np.cos(d), np.sin(d)] for d in line] for line in source_pos_ids], dtype=np.float32)      # [?, n, 2]

        target_pos_ids = np.array([[i for i in range(1, target_ids.shape[1] + 1)]])
        target_pos_ids = np.tile(target_pos_ids, [target_ids.shape[0], 1])
        target_pos_ids = ((target_pos_ids * 2 * np.pi) / target_max_len[:, np.newaxis]).astype(np.int32)     # [?, m]
        target_pos_ids = np.array([[[np.cos(d), np.sin(d)] for d in line] for line in target_pos_ids], dtype=np.float32)      # [?, m, 2]

        return source_ids, target_ids, source_pos_ids, target_pos_ids, source_mask, target_mask

    def get_batch(self, path, batch_size):
        dataset = tf.data.TextLineDataset([path])
        dataset = dataset.batch(batch_size)
        iter = dataset.make_initializable_iterator()
        batch = iter.get_next()
        source_sent, target_sent, source_pos_ids, target_pos_ids, source_mask, target_mask = \
            tf.py_func(self.batch_process, [batch], [tf.int32, tf.int32, tf.float32, tf.float32, tf.bool, tf.bool])
        return source_sent, target_sent, source_pos_ids, target_pos_ids, source_mask, target_mask, iter

    def load_data(self):
        self.vocab_source_size = len(self.source_tokenizer.word_index) + 1
        self.vocab_target_size = len(self.target_tokenizer.word_index) + 1
        print(list(self.source_tokenizer.word_index.items())[:100])
        print(list(self.target_tokenizer.word_index.items())[:100])

        self.source_id2word = {v: k for k, v in self.source_tokenizer.word_index.items()}
        self.target_id2word = {v: k for k, v in self.target_tokenizer.word_index.items()}
        print(list(self.source_id2word.items())[:100])
        print(list(self.target_id2word.items())[:100])

        self.vocab_source_freq = [self.source_tokenizer.word_counts[self.source_id2word[_id]]
                                  for _id in range(1, self.vocab_source_size)]
        self.vocab_source_freq.insert(0, 0)
        self.vocab_target_freq = [self.target_tokenizer.word_counts[self.target_id2word[_id]]
                                  for _id in range(1, self.vocab_target_size)]
        self.vocab_target_freq.insert(0, 0)

    def init_embedding(self):
        # when target embeddings for initialization is assigned
        if self.p.embed_loc:
            print("Start loading embeddings ... ")
            # source part
            source_emb_init_path = self.p.embed_loc + '/source_emb'
            source_embed_init = getEmbeddings(source_emb_init_path, [self.source_id2word[i] for i in range(1, len(self.source_id2word) + 1)], self.p.embed_dim)
            self.source_emb_table = tf.get_variable(
                'source_emb',
                initializer=source_embed_init)
            # target part
            target_emb_init_path = self.p.embed_loc + '/target_emb'
            target_embed_init = getEmbeddings(target_emb_init_path, [self.target_id2word[i] for i in range(1, len(self.target_id2word) + 1)], self.p.embed_dim)
            self.target_emb_table = tf.get_variable(
                'target_emb',
                initializer=target_embed_init)
            print('Embeddings loading done ! ')
        else:
            print("Init embedding ... ")
            # self.source_emb_table = tf.get_variable(name='inp_emb', shape=[self.vocab_source_size, 128],
            #                                    initializer=tf.random_normal_initializer(mean=0, stddev=1))
            # self.target_emb_table = tf.get_variable(name='tar_emb', shape=[self.vocab_target_size, 128],
            #                                    initializer=tf.random_normal_initializer(mean=0, stddev=1))
            self.source_emb_table = tf.get_variable(name='inp_emb', shape=[self.vocab_source_size, self.p.embed_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            self.target_emb_table = tf.get_variable(name='tar_emb', shape=[self.vocab_target_size, self.p.embed_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            print("Embedding init done !")

        # init pos embedding
        self.source_pos_emb_W = tf.get_variable(name='source_W', shape=[2, self.p.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer())
        self.target_pos_emb_W = tf.get_variable(name='target_W', shape=[2, self.p.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer())
        # self.source_pos_emb_table = tf.get_variable(name='source_pos_emb', shape=[500, int(self.p.embed_dim / 10)],
        #                                             initializer=tf.contrib.layers.xavier_initializer())
        # self.target_pos_emb_table = tf.get_variable(name='target_pos_emb', shape=[500, int(self.p.embed_dim / 10)],
        #                                             initializer=tf.contrib.layers.xavier_initializer())
        print("Position Embedding init done !")

    def add_model(self, source_sent, target_sent, source_pos_ids, target_pos_ids, source_mask, target_mask):
        """
        Creates the Computational Graph

        Parameters
        ----------

        Returns
        -------
        nn_out:		Logits for each bag in the batch
        """

        # merge word embedding & position embedding
        source_sent_embed = tf.nn.embedding_lookup(self.source_emb_table, source_sent)  # [?, n, 128]
        source_pos_embed = tf.matmul(source_pos_ids, tf.tile(tf.expand_dims(self.source_pos_emb_W, 0), [tf.shape(source_pos_ids)[0], 1, 1]))   # [?, n, 128]
        source_sent_embed = source_sent_embed + source_pos_embed

        target_sent_embed = tf.nn.embedding_lookup(self.target_emb_table, target_sent)  # [?, m, 128]
        target_pos_embed = tf.matmul(target_pos_ids, tf.tile(tf.expand_dims(self.target_pos_emb_W, 0), [tf.shape(target_pos_ids)[0], 1, 1]))   # [?, n, 128]
        target_sent_embed = target_sent_embed + target_pos_embed

        # mask
        source_mask_tile = tf.tile(tf.expand_dims(source_mask, 2), [1, 1, tf.shape(target_mask)[1]])    # [?, n, m]
        target_mask_tile = tf.tile(tf.expand_dims(target_mask, 1), [1, tf.shape(source_mask)[1], 1])    # [?, n, m]
        mask = tf.logical_and(source_mask_tile, target_mask_tile)    # [?, n, m]

        # get attention context vectors
        ta_score = tf.matmul(target_sent_embed, source_sent_embed, transpose_b=True)    # [?, m, n]
        ta_score = tf.where(tf.transpose(mask, perm=[0, 2, 1]), ta_score, tf.ones(tf.shape(ta_score), dtype=tf.float32) * -999)    # [?, m, n]
        ta_soft_score = tf.nn.softmax(ta_score)     # [?, m, n]
        source_att_embed = tf.matmul(ta_soft_score, source_sent_embed)  # [?, m, 128]

        at_score = tf.transpose(ta_score, perm=[0, 2, 1])   # [?, n, m]
        at_score = tf.where(mask, at_score, tf.ones(tf.shape(at_score), dtype=tf.float32) * -999)    # [?, n, m]
        at_soft_score = tf.nn.softmax(at_score)     # [?, n, m]
        target_att_embed = tf.matmul(at_soft_score, target_sent_embed)  # [?, n, 128]

        return source_sent_embed, source_att_embed, target_sent_embed, target_att_embed

    def build_eval_graph(self):
        eval_source_sent, eval_target_sent, source_pos_ids, target_pos_ids, eval_source_mask, eval_target_mask, self.eval_iter =\
            self.get_batch(self.eval_path_to_file, self.p.batch_size)

        # merge word embedding & position embedding
        source_sent_embed = tf.nn.embedding_lookup(self.source_emb_table, eval_source_sent)  # [?, n, 128]
        source_pos_embed = tf.matmul(source_pos_ids, tf.tile(tf.expand_dims(self.source_pos_emb_W, 0), [tf.shape(source_pos_ids)[0], 1, 1]))   # [?, n, 128]
        source_sent_embed = source_sent_embed + source_pos_embed

        target_sent_embed = tf.nn.embedding_lookup(self.target_emb_table, eval_target_sent)  # [?, m, 128]
        target_pos_embed = tf.matmul(target_pos_ids, tf.tile(tf.expand_dims(self.target_pos_emb_W, 0), [tf.shape(target_pos_ids)[0], 1, 1]))   # [?, n, 128]
        target_sent_embed = target_sent_embed + target_pos_embed

        source_mask_tile = tf.tile(tf.expand_dims(eval_source_mask, 2), [1, 1, tf.shape(eval_target_mask)[1]])    # [?, n, m]
        target_mask_tile = tf.tile(tf.expand_dims(eval_target_mask, 1), [1, tf.shape(eval_source_mask)[1], 1])    # [?, n, m]
        mask = tf.logical_and(source_mask_tile, target_mask_tile)    # [?, n, m]

        ta_score = tf.matmul(target_sent_embed, source_sent_embed, transpose_b=True)    # [?, m, n]
        ta_score = tf.where(tf.transpose(mask, perm=[0, 2, 1]), ta_score, tf.ones(tf.shape(ta_score), dtype=tf.float32) * -999)    # [?, m, n]
        ts_align_score = tf.nn.softmax(ta_score)     # [?, m, n]
        self.ts_align = tf.argmax(ts_align_score, 2, output_type=tf.int32) + 1
        self.ts_align = tf.where(eval_target_mask, self.ts_align, tf.zeros(tf.shape(self.ts_align), dtype=tf.int32))    # [?, m]

        at_score = tf.transpose(ta_score, perm=[0, 2, 1])   # [?, n, m]
        at_score = tf.where(mask, at_score, tf.ones(tf.shape(at_score), dtype=tf.float32) * -999)    # [?, n, m]
        st_align_score = tf.nn.softmax(at_score)     # [?, n, m]
        self.st_align = tf.arg_max(st_align_score, 2, output_type=tf.int32) + 1
        self.st_align = tf.where(eval_source_mask, self.st_align, tf.zeros(tf.shape(self.st_align), dtype=tf.int32))    # [?, n]

        # for debug
        self.eval_source_sent = eval_source_sent
        self.eval_target_sent = eval_target_sent

    def add_loss_op(self):
        """
        Computes the loss for learning embeddings

        Parameters
        ----------
        nn_out:		Logits for each bag in the batch

        Returns
        -------
        loss:		Computes loss
        """

        # train graph
        source_sent, target_sent, source_pos_ids, target_pos_ids, source_mask, target_mask, self.train_iter =\
            self.get_batch(self.path_to_file, self.p.batch_size)
        source_sent_embed, source_att_embed, target_sent_embed, target_att_embed =\
            self.add_model(source_sent, target_sent, source_pos_ids, target_pos_ids, source_mask, target_mask)

        target_words = tf.reshape(target_sent, [-1, 1])    # [? * m]
        source_words = tf.reshape(source_sent, [-1, 1])    # [? * n]

        target_neg_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.cast(target_words, tf.int64),
            num_true=1,
            num_sampled=self.p.num_neg * self.p.batch_size,
            unique=True,
            distortion=0.75,
            range_max=self.vocab_target_size,
            unigrams=self.vocab_target_freq
        )

        source_neg_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.cast(source_words, tf.int64),
            num_true=1,
            num_sampled=self.p.num_neg * self.p.batch_size,
            unique=True,
            distortion=0.75,
            range_max=self.vocab_source_size,
            unigrams=self.vocab_source_freq
        )
        target_neg_ids = tf.cast(target_neg_ids, dtype=tf.int32)    # [? * neg_num]
        target_neg_ids = tf.tile(tf.expand_dims(tf.reshape(target_neg_ids, [self.p.batch_size, self.p.num_neg]), 2), [1, 1, tf.shape(target_sent)[1]]) # [?, num_neg, t_len]
        target_neg_embed = tf.nn.embedding_lookup(self.target_emb_table, target_neg_ids)    # [?, num_neg, t_len, 128]
        source_neg_ids = tf.cast(source_neg_ids, dtype=tf.int32)
        source_neg_ids = tf.tile(tf.expand_dims(tf.reshape(source_neg_ids, [self.p.batch_size, self.p.num_neg]), 2), [1, 1, tf.shape(source_sent)[1]]) # [?, num_neg, s_len]
        source_neg_embed = tf.nn.embedding_lookup(self.source_emb_table, source_neg_ids)    # [?, num_neg, s_len, 128]

        source_embed = tf.concat([tf.expand_dims(source_sent_embed, 1), source_neg_embed], 1)   # [?, num_neg+1, s_len, 128]
        target_embed = tf.concat([tf.expand_dims(target_sent_embed, 1), target_neg_embed], 1)   # [?, num_neg+1, t_len, 128]

        # logits
        source_logits = tf.reduce_sum(tf.multiply(source_embed, tf.tile(tf.expand_dims(target_att_embed, 1), [1, self.p.num_neg+1, 1, 1])), 3)  # [?, num_neg+1, s_len]
        target_logits = tf.reduce_sum(tf.multiply(target_embed, tf.tile(tf.expand_dims(source_att_embed, 1), [1, self.p.num_neg+1, 1, 1])), 3)  # [?, num_neg+1, t_len]

        # labels
        source_pos_labels = tf.expand_dims(tf.ones(tf.shape(source_sent), dtype=tf.float32), 1)    # [?, 1, s_len]
        source_neg_labels = tf.zeros(tf.shape(source_neg_ids), dtype=tf.float32)    # [?, num_neg, s_len]
        source_labels = tf.concat([source_pos_labels, source_neg_labels], axis=1)   # [?, num_neg+1, s_len]
        target_pos_labels = tf.expand_dims(tf.ones(tf.shape(target_sent), dtype=tf.float32), 1)    # [?, 1, t_len]
        target_neg_labels = tf.zeros(tf.shape(target_neg_ids), dtype=tf.float32)    # [?, num_neg, t_len]
        target_labels = tf.concat([target_pos_labels, target_neg_labels], axis=1)   # [?, num_neg+1, t_len]

        # loss
        source_loss = tf.nn.weighted_cross_entropy_with_logits(targets=source_labels, logits=source_logits, pos_weight=1.0, name='source_loss')   # [?, num_neg+1, s_len]
        source_mask_tile = tf.tile(tf.expand_dims(source_mask, 1), [1, tf.shape(source_loss)[1], 1])
        source_loss = tf.where(source_mask_tile, source_loss, tf.zeros(tf.shape(source_loss)))   # get valid loss
        target_loss = tf.nn.weighted_cross_entropy_with_logits(targets=target_labels, logits=target_logits, pos_weight=1.0, name='target_loss')   # [?, num_neg+1, t_len]
        target_mask_tile = tf.tile(tf.expand_dims(target_mask, 1), [1, tf.shape(target_loss)[1], 1])
        target_loss = tf.where(target_mask_tile, target_loss, tf.zeros(tf.shape(target_loss)))   # get valid loss

        loss = tf.reduce_mean(tf.reduce_sum(source_loss, 2)) + tf.reduce_mean(tf.reduce_sum(target_loss, 2))

        # if self.regularizer is not None:
        #     loss += tf.contrib.layers.apply_regularization(
        #         self.regularizer, tf.get_collection(
        #             tf.GraphKeys.REGULARIZATION_LOSSES))

        return loss

    def add_optimizer(self, loss, isAdam=True):
        """
        Add optimizer for training variables

        Parameters
        ----------
        loss:		Computed loss

        Returns
        -------
        train_op:	Training optimizer
        """
        with tf.name_scope('Optimizer'):
            if isAdam:
                optimizer = tf.train.AdamOptimizer(self.p.lr)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
            train_op = optimizer.minimize(loss)

        return train_op

    def checkpoint(self, epoch, sess):

        """
        Computes intrinsic scores for embeddings and dumps the embeddings embeddings

        Parameters
        ----------
        epoch:		Current epoch number
        sess:		Tensorflow session object

        Returns
        -------
        """
        source_emb, target_emb = sess.run([self.source_emb_table, self.target_emb_table])
        source_voc2vec = {wrd: source_emb[wid] for wrd, wid in self.source_tokenizer.word_index.items()}

        self.logger.info("Saving embedding matrix")
        f_s = open('{}/{}-source-{}'.format(self.p.emb_dir, self.p.name, epoch), 'w')
        for wrd, emb in source_voc2vec.items():
            f_s.write('{} {}\n'.format(wrd, ' '.join([str(round(v, 6)) for v in emb.tolist()])))
        f_s.flush()
        f_s.close()

        target_voc2vec = {wrd: target_emb[wid] for wrd, wid in self.target_tokenizer.word_index.items()}
        f_t = open('{}/{}-target-{}'.format(self.p.emb_dir, self.p.name, epoch), 'w')
        for wrd, emb in target_voc2vec.items():
            f_t.write('{} {}\n'.format(wrd, ' '.join([str(round(v, 6)) for v in emb.tolist()])))
        f_t.flush()
        f_t.close()
        self.logger.info("Embedding saving done ! ")

    def get_alignment(self, epoch, sess):

        cnt = 0
        step = 0
        sess.run(self.eval_iter.initializer)
        fs_out = open('{}/{}-st-alginment-{}'.format(self.p.emb_dir, self.p.name, epoch), 'w')
        ft_out = open('{}/{}-ts-alginment-{}'.format(self.p.emb_dir, self.p.name, epoch), 'w')
        fs_wa_out = open('data/en-fr-eval-wa.txt', 'w')

        while 1:
            step = step + 1
            # st_align, ts_align, s_sent, t_sent = \
            #     sess.run([self.st_align, self.ts_align, self.eval_source_sent, self.eval_target_sent])
            try:
                st_align, ts_align, s_sent, t_sent =\
                    sess.run([self.st_align, self.ts_align, self.eval_source_sent, self.eval_target_sent])
            except:
                print('{} Alignments Writing Done ! '.format(cnt))
                break

            # source sent alignment
            for i in range(s_sent.shape[0]):
                s_sent_tmp = []
                s_align_tmp = []
                for j in range(s_sent.shape[1]):
                    if s_sent[i][j] > 0:
                        s_sent_tmp.append(self.source_id2word[s_sent[i][j]])
                    if st_align[i][j] > 0:
                        s_align_tmp.append(str(st_align[i][j]))
                s_sent_out = " ".join(s_sent_tmp)
                s_align_out = " ".join(s_align_tmp)
                fs_out.write(s_sent_out + '\n')
                fs_out.write(s_align_out + '\n')

            # target sent alignment
            for i in range(t_sent.shape[0]):
                t_sent_tmp = []
                t_align_tmp = []
                for j in range(t_sent.shape[1]):
                    if t_sent[i][j] > 0:
                        t_sent_tmp.append(self.target_id2word[t_sent[i][j]])
                    if ts_align[i][j] > 0:
                        t_align_tmp.append(str(ts_align[i][j]))
                t_sent_out = " ".join(t_sent_tmp)
                t_align_out = " ".join(t_align_tmp)
                ft_out.write(t_sent_out + '\n')
                ft_out.write(t_align_out + '\n')

            # s -> t word alignment
            sent_num = 0
            for i in range(st_align.shape[0]):
                sent_num += 1
                for j in range(st_align.shape[1]):
                    if st_align[i][j] > 0:
                        fs_wa_out.write('num-' + str(sent_num) + ' ' + str(j+1) + ' -> ' + str(st_align[i][j]))
            fs_wa_out.flush()
            fs_wa_out.close()

            cnt += self.p.batch_size
            if step % 10 == 0:
                self.logger.info('Write Sents: {}'.format(cnt))

        fs_out.flush()
        fs_out.close()
        ft_out.flush()
        ft_out.close()
        print('Write Alignment Done ! ')

        return

    def eval_on_word_embedding(self, sess):
        source_emb, target_emb = sess.run([self.source_emb_table, self.target_emb_table])

        source_voc2vec = {wrd: source_emb[wid] for wrd, wid in self.source_tokenizer.word_index.items()}
        source_embedding = Embedding.from_dict(source_voc2vec)
        results = evaluate_on_all(source_embedding)
        results = {key: round(val[0], 4) for key, val in results.items()}
        curr_int = np.mean(list(results.values()))
        self.logger.info('Current Source Word2vec Score: {}'.format(curr_int))

        # # Only EN corpus could be evaluated.
        # target_voc2vec = {wrd: target_emb[wid] for wrd, wid in self.target_tokenizer.word_index.items()}
        # target_embedding = Embedding.from_dict(target_voc2vec)
        # results = evaluate_on_all(target_embedding)
        # results = {key: round(val[0], 4) for key, val in results.items()}
        # curr_int = np.mean(list(results.values()))
        # self.logger.info('Current Target Word2vec Score: {}'.format(curr_int))

        return

    def run_epoch(self, sess, epoch, shuffle=True):
        """
        Runs one epoch of training

        Parameters
        ----------
        sess:		Tensorflow session object
        epoch:		Epoch number
        shuffle:	Shuffle data while before creates batches

        Returns
        -------
        loss:		Loss over the corpus
        """
        losses = []
        cnt = 0
        step = 0
        st = time.time()
        sess.run(self.train_iter.initializer)

        while 1:
            step = step + 1
            # loss, _ = sess.run([self.loss, self.train_op])
            try:
                loss, _ = sess.run([self.loss, self.train_op])
            except:
                break
            losses.append(loss)

            cnt += self.p.batch_size
            if step % 10 == 0:
                self.logger.info(
                    'E:{} (Sents: {}/{} [{}]): Train Loss \t{:.5}\t{}\t{:.5}'.format(
                        epoch,
                        cnt,
                        10000,
                        round(cnt / 10000 * 100, 1),
                        np.mean(losses),
                        self.p.name,
                        self.best_int_avg))
            en = time.time()
            if (en - st) >= (3600):
                self.logger.info("One more hour is over")
                self.checkpoint(epoch, sess)
                st = time.time()

        return np.mean(losses)

    def fit(self, sess):
        """
        Trains the model and finally evaluates on test

        Parameters
        ----------
        sess:		Tensorflow session object

        Returns
        -------
        """
        self.saver = tf.train.Saver()
        save_dir = 'checkpoints/' + self.p.name + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir, 'best_int_avg')

        self.best_int_avg = 0.0

        if self.p.restore:
            self.saver.restore(sess, self.save_path)

        for epoch in range(self.p.max_epochs):
            self.logger.info('Epoch: {}'.format(epoch))
            train_loss = self.run_epoch(sess, epoch)

            self.eval_on_word_embedding(sess)
            self.checkpoint(epoch, sess)
            self.get_alignment(epoch, sess)

            self.logger.info(
                '[Epoch {}]: Training Loss: {:.5}, Best Loss: {:.5}\n'.format(
                    epoch, train_loss, self.best_int_avg))

    def __init__(self, params):

        # data file
        self.path_to_file = "./data/en-fr-sample.txt"
        self.eval_path_to_file = "./data/en-fr-eval.txt"

        # create tokenizer
        self.create_tokenizer()

        self.p = params

        if not os.path.isdir(self.p.log_dir):
            os.system('mkdir {}'.format(self.p.log_dir))
        if not os.path.isdir(self.p.emb_dir):
            os.system('mkdir {}'.format(self.p.emb_dir))

        self.logger = get_logger(
            self.p.name,
            self.p.log_dir,
            self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))
        self.p.batch_size = self.p.batch_size

        if self.p.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(
                scale=self.p.l2)

        self.load_data()
        self.init_embedding()
        self.build_eval_graph()

        self.loss = self.add_loss_op()

        if self.p.opt == 'adam':
            self.train_op = self.add_optimizer(self.loss)
        else:
            self.train_op = self.add_optimizer(self.loss, isAdam=False)

        self.merged_summ = tf.summary.merge_all()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SynAlign')

    parser.add_argument('-gpu', dest="gpu", default='0', help='GPU to use')
    parser.add_argument(
        '-name',
        dest="name",
        default='test_run',
        help='Name of the run')
    parser.add_argument(
        '-embed',
        dest="embed_loc",
        default=None,
        help='Embedding for initialization')
    parser.add_argument(
        '-embed_dim',
        dest="embed_dim",
        default=128,
        type=int,
        help='Embedding Dimension')
    parser.add_argument(
        '-total',
        dest="total_sents",
        default=56974869,
        type=int,
        help='Total number of sentences in file')
    parser.add_argument(
        '-lr',
        dest="lr",
        default=0.001,
        type=float,
        help='Learning rate')
    parser.add_argument(
        '-batch',
        dest="batch_size",
        default=4,
        type=int,
        help='Batch size')
    parser.add_argument(
        '-epoch',
        dest="max_epochs",
        default=50,
        type=int,
        help='Max epochs')
    parser.add_argument(
        '-l2',
        dest="l2",
        default=0.00001,
        type=float,
        help='L2 regularization')
    parser.add_argument(
        '-seed',
        dest="seed",
        default=1234,
        type=int,
        help='Seed for randomization')
    parser.add_argument(
        '-sample',
        dest="sample",
        default=1e-4,
        type=float,
        help='Subsampling parameter')
    parser.add_argument(
        '-neg',
        dest="num_neg",
        default=10,
        type=int,
        help='Number of negative samples')
    parser.add_argument(
        '-side_int',
        dest="side_int",
        default=10000,
        type=int,
        help='Number of negative samples')
    parser.add_argument(
        '-gcn_layer',
        dest="gcn_layer",
        default=1,
        type=int,
        help='Number of layers in GCN over dependency tree')
    parser.add_argument(
        '-drop',
        dest="dropout",
        default=1.0,
        type=float,
        help='Dropout for full connected layer (Keep probability')
    parser.add_argument('-opt', dest="opt", default='adam',
                        help='Optimizer to use for training')
    parser.add_argument(
        '-dump',
        dest="onlyDump",
        action='store_true',
        help='Dump context and embed matrix')
    parser.add_argument(
        '-context',
        dest="context",
        action='store_true',
        help='Include sequential context edges (default: False)')
    parser.add_argument(
        '-restore',
        dest="restore",
        action='store_true',
        help='Restore from the previous best saved model')
    parser.add_argument(
        '-embdir',
        dest="emb_dir",
        default='./embeddings/',
        help='Directory for storing learned embeddings')
    parser.add_argument(
        '-logdir',
        dest="log_dir",
        default='./log/',
        help='Log directory')
    parser.add_argument(
        '-config',
        dest="config_dir",
        default='./config/',
        help='Config directory')

    # Added these two arguments to enable others to personalize the training set. Otherwise, the programme may suffer from memory overflow easily.
    # It is suggested that the -maxlen be set no larger than 100.
    parser.add_argument(
        '-maxsentlen',
        dest="max_sent_len",
        default=50,
        type=int,
        help='Max length of the sentences in data.txt (default: 40)')
    parser.add_argument(
        '-maxdeplen',
        dest="max_dep_len",
        default=800,
        type=int,
        help='Max length of the dependency relations in data.txt (default: 800)')

    args = parser.parse_args()

    if not args.restore:
        args.name = args.name + '_' + \
            time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # set_gpu(args.gpu)

    model = SynAlign(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)

    print('Model Trained Successfully!!')
