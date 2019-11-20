import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, queue, re
import gzip, queue, threading, scipy.sparse as sp
import logging, logging.config, itertools, pathlib
import unicodedata

from pprint 	 import pprint
from threading   import Thread
from collections import defaultdict as ddict

np.set_printoptions(precision=4)


def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------    
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def debug_nn(res_list, feed_dict):
	"""
	Function for debugging Tensorflow model      

	Parameters
	----------
	res_list:       List of tensors/variables to view
	feed_dict:	Feed dict required for getting values
	
	Returns
	-------
	Returns the list of values of given tensors/variables after execution

	"""
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	return res


def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger


def getChunks(inp_list, chunk_size):
	"""
	Splits inp_list into lists of size chunk_size

	Parameters
	----------
	inp_list:       List to be splittted
	chunk_size:     Size of each chunk required
	
	Returns
	-------
	chunks of the inp_list each of size chunk_size, last one can be smaller (leftout data)
	"""
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]


def read_mappings(fname):
	"""
	A helper function for reading an object to identifier mapping

	Parameters
	----------
	fname:		Name of the file containing mapping

	Returns
	-------
	mapping:	Dictionary object containing mapping information
	"""
	mapping = {}
	for line in open(fname):
		vals = line.strip().split('\t')
		if len(vals) < 2: continue
		mapping[vals[0]] = vals[1]
	return mapping


def getEmbeddings(embed_loc, wrd_list, embed_dims):
	"""
	Gives embedding for each word in wrd_list

	Parameters
	----------
	model:		Word2vec model
	wrd_list:	List of words for which embedding is required
	embed_dims:	Dimension of the embedding

	Returns
	-------
	embed_matrix:	(len(wrd_list) x embed_dims) matrix containing embedding for each word in wrd_list in the same order
	"""
	embed_list = []

	wrd2embed = {}
	for line in open(embed_loc, encoding='utf-8', errors='ignore'):
		data = line.strip().split(' ')

		# wrd, embed = data[0], data[1:]

		# Some words may be separated by space (telephone numbers, for example).
		# It's more robust to load data as follows.
		embed = data[-1*embed_dims:]
		wrd = ' '.join(data[: -1*embed_dims])

		embed = list(map(float, embed))
		wrd2embed[wrd] = embed

	for wrd in wrd_list:
		if wrd in wrd2embed:
			embed_list.append(wrd2embed[wrd])
		else: 	
			print('Word not in embeddings dump {}'.format(wrd))
			embed_list.append(np.random.randn(embed_dims))

	# add embedding for reserved 0
	embed_list.insert(0, np.random.randn(embed_dims))
	return np.array(embed_list, dtype=np.float32)


def unicode_to_ascii(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
	# w = unicode_to_ascii(w.lower().strip())
	# w = re.sub(r"([?.!,¿])", r" \1 ", w)
	# w = re.sub(r'[" "]+', " ", w)
	#
	# # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
	# w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
	#
	# w = w.rstrip().strip()
	#
	# # adding a start and an end token to the sentence
	# # so that the model know when to start and stop predicting.
	# # w = '<start> ' + w + ' <end>'
	return w


def max_length(tensor):
    return max(len(t) for t in tensor)


def get_wa_score(labeled_file, eval_file):
	s_set = set()
	p_set = set()
	eval_set = set()
	for line in open(labeled_file):
		line_array = line.strip().split(' ')
		if line_array[4] == 'S':
			s_set.add(' '.join(line_array[:4]))
		p_set.add(' '.join(line_array[:4]))
	for line in open(eval_file):
		eval_set.add(line.strip())
	P = len(eval_set.intersection(p_set)) / len(eval_set)
	R = len(eval_set.intersection(s_set)) / len(s_set)
	F1 = (2 * P * R) / (P + R)
	AER = 1 - (len(eval_set.intersection(s_set)) + len(eval_set.intersection(p_set))) / (len(eval_set) + len(s_set))
	return P, R, F1, AER


def get_max_grow_diag_final_alignment(st_lst, ts_lst, shift_num):

    def _get_max_score_alignment(map, transpose=False):
        align_set = set()
        score_map = np.array(map)
        if len(score_map.shape) != 2:
            print('the shape must be 2 !')
            raise Exception
        max_align = score_map.argmax(axis=1)
        if transpose == False:
            for i in range(len(max_align)):
                align_set.add((i, max_align[i]))
        else:
            for i in range(len(max_align)):
                align_set.add((max_align[i], i))
        return align_set

    def _get_align_set(align_map, shift):

        align_set = set()
        for i in range(align_map.shape[0]):
            for j in range(align_map.shape[1]):
                if align_map[i][j] == 2:
                    align_set.add('num-' + str(shift) + ' ' + str(i+1) + ' -> ' + str(j+1))
        return align_set

    def _grow_diag(st_score, ts_score, align_map):
        neighbouring = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        st_score = np.array(st_score)   # [n, m]
        ts_score = np.array(ts_score).transpose((1, 0)) # [n, m]
        rows = align_map.shape[0]
        cols = align_map.shape[1]

        s_aligned = set()
        t_aligned = set()

        while 1:
            # init s_aligned and t_aligned
            for i in range(rows):
                for j in range(cols):
                    if align_map[i][j] == 2:
                        s_aligned.add(i)
                        t_aligned.add(j)

            # get candidates
            candidate_set = set()
            for i in range(rows):
                for j in range(cols):
                    if align_map[i][j] == 2:
                        for dirs in neighbouring:
                            x = i + dirs[0]
                            y = j + dirs[1]
                            if 0 <= x < rows and 0 <= y < cols and (x not in s_aligned or y not in t_aligned) and\
                                    align_map[x][y] == 1:
                                candidate_set.add((x, y))

            # get new alignment if exists
            is_end = True
            max_x = -1
            max_y = -1
            max_score = 0
            for candidate in candidate_set:
                x = candidate[0]
                y = candidate[1]
                if st_score[x][y] > max_score or ts_score[x][y] > max_score:
                    max_score = st_score[x][y] if st_score[x][y] > ts_score[x][y] else ts_score[x][y]
                    max_x = x
                    max_y = y
            if max_x >= 0 and max_y >= 0:
                align_map[max_x][max_y] = 2
                is_end = False

            if is_end:
                break
        return align_map

    def _final_process(st_score, ts_score, align_map):

        st_score = np.array(st_score)
        ts_score = np.array(ts_score).transpose()

        if st_score.shape != align_map.shape:
            print('The shapes of input1 and input3 is not consistent !')
            raise Exception

        if ts_score.shape != align_map.shape:
            print('The shapes of input2 and input3 is not consistent !')
            raise Exception

        # get final alignments
        while 1:
            # init
            s_aligned = set()
            t_aligned = set()
            for i in range(align_map.shape[0]):
                for j in range(align_map.shape[1]):
                    if align_map[i][j] == 2:
                        s_aligned.add(i)
                        t_aligned.add(j)

            # get condidates
            condidate_set = set()
            for i in range(align_map.shape[0]):
                for j in range(align_map.shape[1]):
                    if align_map[i][j] == 1 and (i not in s_aligned or j not in t_aligned) and\
                            (st_score[i][j] > 0 or ts_score[i][j] > 0):
                        condidate_set.add((i, j))

            if len(condidate_set) == 0:
                break

            # get one max alignment
            max_x = -1
            max_y = -1
            max_score = 0
            for tup in condidate_set:
                x = tup[0]
                y = tup[1]
                tmp_score = st_score[x][y] if st_score[x][y] > ts_score[x][y] else ts_score[x][y]
                if tmp_score > max_score:
                    max_x = x
                    max_y = y
                    max_score = tmp_score
            if max_x >= 0 and max_y >= 0:
                align_map[max_x][max_y] = 2

        return align_map

    st_score = np.array(st_lst)
    ts_score = np.array(ts_lst)
    if st_score.shape != ts_score.transpose((0, 2, 1)).shape:
        print('the shape of two input matrix must be consistent')
        raise Exception

    alignment_set = set()
    for i in range(st_score.shape[0]):
        # init align map, 2: sure 1: candidate 0: not align
        align_map = np.zeros(st_score[i].shape, dtype=np.int32)
        cur_align_set = set()

        # step1
        st_align_set = _get_max_score_alignment(st_score[i], False)
        ts_align_set = _get_max_score_alignment(ts_score[i], True)
        cur_align_set.update(st_align_set.intersection(ts_align_set))
        for tup in cur_align_set:
            align_map[tup[0]][tup[1]] = 2
        for tup in st_align_set:
            if align_map[tup[0]][tup[1]] == 0:
                align_map[tup[0]][tup[1]] = 1
        for tup in ts_align_set:
            if align_map[tup[0]][tup[1]] == 0:
                align_map[tup[0]][tup[1]] = 1
        # step2
        align_map = _grow_diag(st_score[i], ts_score[i], align_map)
        # step3
        align_map = _final_process(st_score[i], ts_score[i], align_map)
        # update alignment_set
        cur_align_set = _get_align_set(align_map, shift_num + i + 1)
        alignment_set.update(cur_align_set)

    return alignment_set


def get_max_grow_diag_alignment(st_lst, ts_lst, shift_num):
    # st_lst: [?, n, m]
    # ts_lst: [?, m, n]

    def _get_max_score_alignment(map, transpose=False):
        align_set = set()
        score_map = np.array(map)
        if len(score_map.shape) != 2:
            print('the shape must be 2 !')
            raise Exception
        max_align = score_map.argmax(axis=1)
        if transpose == False:
            for i in range(len(max_align)):
                align_set.add((i, max_align[i]))
        else:
            for i in range(len(max_align)):
                align_set.add((max_align[i], i))
        return align_set

    def _get_align_set(align_map, shift):

        align_set = set()
        for i in range(align_map.shape[0]):
            for j in range(align_map.shape[1]):
                if align_map[i][j] == 2:
                    align_set.add('num-' + str(shift) + ' ' + str(i+1) + ' -> ' + str(j+1))
        return align_set

    def _grow_diag(st_score, ts_score, align_map):
        neighbouring = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        st_score = np.array(st_score)
        ts_score = np.array(ts_score).transpose((1, 0))
        rows = align_map.shape[0]
        cols = align_map.shape[1]

        s_aligned = set()
        t_aligned = set()

        while 1:
            # init s_aligned and t_aligned
            for i in range(rows):
                for j in range(cols):
                    if align_map[i][j] == 2:
                        s_aligned.add(i)
                        t_aligned.add(j)

            # get candidates
            candidate_set = set()
            for i in range(rows):
                for j in range(cols):
                    if align_map[i][j] == 2:
                        for dirs in neighbouring:
                            x = i + dirs[0]
                            y = j + dirs[1]
                            if 0 <= x < rows and 0 <= y < cols and (x not in s_aligned or y not in t_aligned) and\
                                    align_map[x][y] == 1:
                                candidate_set.add((x, y))

            # get new alignment if exists
            is_end = True
            max_x = -1
            max_y = -1
            max_score = 0
            for candidate in candidate_set:
                x = candidate[0]
                y = candidate[1]
                if st_score[x][y] > max_score or ts_score[x][y] > max_score:
                    max_score = st_score[x][y] if st_score[x][y] > ts_score[x][y] else ts_score[x][y]
                    max_x = x
                    max_y = y
            if max_x >= 0 and max_y >= 0:
                align_map[max_x][max_y] = 2
                is_end = False

            if is_end:
                break
        return align_map

    def _final_process(st_score, ts_score, align_map):

        st_score = np.array(st_score)
        ts_score = np.array(ts_score).transpose()

        if st_score.shape != align_map.shape:
            print('The shapes of input1 and input3 is not consistent !')
            raise Exception

        if ts_score.shape != align_map.shape:
            print('The shapes of input2 and input3 is not consistent !')
            raise Exception

        # get final alignments
        while 1:
            # init
            s_aligned = set()
            t_aligned = set()
            for i in range(align_map.shape[0]):
                for j in range(align_map.shape[1]):
                    if align_map[i][j] == 2:
                        s_aligned.add(i)
                        t_aligned.add(j)

            # get condidates
            condidate_set = set()
            for i in range(align_map.shape[0]):
                for j in range(align_map.shape[1]):
                    if align_map[i][j] == 1 and (i not in s_aligned or j not in t_aligned) and\
                            (st_score[i][j] > 0 or ts_score[i][j] > 0):
                        condidate_set.add((i, j))

            if len(condidate_set) == 0:
                break

            # get one max alignment
            max_x = -1
            max_y = -1
            max_score = 0
            for tup in condidate_set:
                x = tup[0]
                y = tup[1]
                tmp_score = st_score[x][y] if st_score[x][y] > ts_score[x][y] else ts_score[x][y]
                if tmp_score > max_score:
                    max_x = x
                    max_y = y
                    max_score = tmp_score
            if max_x >= 0 and max_y >= 0:
                align_map[max_x][max_y] = 2

        return align_map

    st_score = np.array(st_lst)
    ts_score = np.array(ts_lst)
    if st_score.shape != ts_score.transpose((0, 2, 1)).shape:
        print('the shape of two input matrix must be consistent')
        raise Exception

    alignment_set = set()
    for i in range(st_score.shape[0]):
        # init align map, 2: sure 1: candidate 0: not align
        align_map = np.zeros(st_score[i].shape, dtype=np.int32)
        cur_align_set = set()

        # step1
        st_align_set = _get_max_score_alignment(st_score[i], False)
        ts_align_set = _get_max_score_alignment(ts_score[i], True)
        cur_align_set.update(st_align_set.intersection(ts_align_set))
        for tup in cur_align_set:
            align_map[tup[0]][tup[1]] = 2
        for tup in st_align_set:
            if align_map[tup[0]][tup[1]] == 0:
                align_map[tup[0]][tup[1]] = 1
        for tup in ts_align_set:
            if align_map[tup[0]][tup[1]] == 0:
                align_map[tup[0]][tup[1]] = 1

        # step2
        align_map = _grow_diag(st_score[i], ts_score[i], align_map)

        # # step3
        # align_map = _final_process(st_score[i], ts_score[i], align_map)

        # update alignment_set
        cur_align_set = _get_align_set(align_map, shift_num + i + 1)
        alignment_set.update(cur_align_set)

    return alignment_set


st_score = np.random.rand(1, 10, 10)
ts_score = np.random.rand(1, 10, 10)
_add = np.expand_dims(np.eye(10) * 0.3, 0)
st_score = st_score + _add
ts_score = ts_score + _add

import time
s_time = time.time()
alignment_set = get_max_grow_diag_final_alignment(st_score, ts_score, 1)
print(time.time() - s_time)

