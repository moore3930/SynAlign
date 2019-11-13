import sys
from collections import *

def main():
    input_file = sys.argv[1]
    de2id_file = open("de2id.txt", 'w')
    id2freq_file = open("id2freq.txt", 'w')
    voc2id_file = open("voc2id.txt", 'w')
    word2freq_file = open("word2freq.txt", 'w')

    word_cnt = Counter()
    dep_cnt = Counter()
    dep_cnt.update({"ROOT": 1})
    voc2id_dict = {}

    for line in open(input_file, 'r'):
        line_array = line.strip().split('\t')
        if len(line_array) == 7:
            word_cnt.update({line_array[1]: 1})
            dep_cnt.update({line_array[6]: 1})

    # voc2id
    index = 0
    for item in list(word_cnt):
        voc2id_file.write("{}\t{}\n".format(item, index))
        voc2id_dict[item] = index
        index = index + 1

    voc2id_file.flush()
    voc2id_file.close()

    # word2freq
    for item in word_cnt.items():
        word2freq_file.write("{}\t{}\n".format(item[0], item[1]))
        id2freq_file.write("{}\t{}\n".format(voc2id_dict[item[0]], item[1]))

    word2freq_file.flush()
    word2freq_file.close()
    id2freq_file.flush()
    id2freq_file.close()

    # de2id
    index = 0
    for item in list(dep_cnt):
        de2id_file.write("{}\t{}\n".format(item, index))
        index = index + 1

    de2id_file.flush()
    de2id_file.close()

if __name__ == '__main__':
    main()
