import sys
from collections import *

def main():
    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]

    # init de2id
    de2id = {}
    idx = 0
    cnt = 0
    for line in open(input_file_1, 'r'):
        line_array = line.strip().split('\t')
        if len(line_array) != 7:
            cnt += 1
            continue
        if line_array[6] not in de2id:
            idx += 1
            de2id[line_array[6]] = idx
    print('{} source sentences'.format(cnt))

    cnt = 0
    for line in open(input_file_2, 'r'):
        line_array = line.strip().split('\t')
        if len(line_array) != 7:
            cnt += 1
            continue
        if line_array[6] not in de2id:
            idx += 1
            de2id[line_array[6]] = idx
    print('{} target sentences'.format(cnt))
    print(de2id)

    word_list = []
    dep_list = []
    source_lst = []
    target_lst = []

    # get out put list
    for line in open(input_file_1):
        line_array = line.strip().split('\t')
        if len(line_array) < 2:
            word_cnt = len(word_list)
            dep_cnt = len(dep_list)
            word_list_str = " ".join(word_list)
            dep_list_str = " ".join(dep_list)
            source_lst.append("{}\t{} {} {}".format(word_list_str, word_cnt, dep_cnt, dep_list_str))
            word_list = []
            dep_list = []
        else:
            word_list.append(str(line_array[1]))
            if line_array[6] != 'ROOT':
                dep_list.append("{}|{}|{}".format(int(line_array[5])-1, int(line_array[0])-1, de2id[line_array[6]]))

    word_list = []
    dep_list = []
    for line in open(input_file_2):
        line_array = line.strip().split('\t')
        if len(line_array) < 2:
            word_cnt = len(word_list)
            dep_cnt = len(dep_list)
            word_list_str = " ".join(word_list)
            dep_list_str = " ".join(dep_list)
            target_lst.append("{}\t{} {} {}".format(word_list_str, word_cnt, dep_cnt, dep_list_str))
            word_list = []
            dep_list = []
        else:
            word_list.append(str(line_array[1]))
            if line_array[6] != 'ROOT':
                dep_list.append("{}|{}|{}".format(int(line_array[5])-1, int(line_array[0])-1, de2id[line_array[6]]))

    # fout
    fout = open(input_file_1 + '.out', 'w')
    for source, target in zip(source_lst, target_lst):
        fout.write(source + '\t' + target + '\n')

    fout.flush()
    fout.close()

if __name__ == '__main__':
    main()
