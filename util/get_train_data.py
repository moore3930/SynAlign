import sys
from collections import *

def main():
    input_file = sys.argv[1]
    de2id_file = open(sys.argv[2], 'r')

    de2id = {}

    for line in de2id_file:
        line_array = line.strip().split('\t')
        de2id[line_array[0]] = int(line_array[1])

    word_list = []
    dep_list = []

    fout = open("{}.out".format(sys.argv[1]), 'w')
    for line in open(input_file):
        line_array = line.strip().split('\t')
        if len(line_array) < 2:
            word_cnt = len(word_list)
            dep_cnt = len(dep_list)
            word_list_str = " ".join(word_list)
            dep_list_str = " ".join(dep_list)
            fout.write("{} {} {} {}\n".format(word_list_str, word_cnt, dep_cnt, dep_list_str))
            word_list = []
            dep_list = []
        else:
            word_list.append(str(line_array[1]))
            if line_array[6] != 'ROOT':
                dep_list.append("{}|{}|{}".format(int(line_array[5])-1, int(line_array[0])-1, de2id[line_array[6]]))

    fout.flush()
    fout.close()

if __name__ == '__main__':
    main()
