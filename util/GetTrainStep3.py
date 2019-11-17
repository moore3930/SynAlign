import sys
from collections import *


def main():
    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    input_file_3 = sys.argv[3]
    input_file_4 = sys.argv[4]

    # init de2id
    class FnScope:
        de2id = {}
        pos2id = {}
        de_idx = 0
        pos_idx = 0

    def init_dict(input_file):
        cnt = 0
        for line in open(input_file, 'r'):
            line_array = line.strip().split('\t')
            if len(line_array) != 7:
                cnt += 1
                continue
            if line_array[6] not in FnScope.de2id:
                FnScope.de_idx += 1
                FnScope.de2id[line_array[6]] = FnScope.de_idx
            if line_array[3] not in FnScope.pos2id:
                FnScope.pos_idx += 1
                FnScope.pos2id[line_array[3]] = FnScope.pos_idx
        print('{} sentences'.format(cnt))

    def get_write_lst(input_file):
        word_list = []
        pos_list = []
        dep_list = []
        output_lst = []

        # get output list
        for line in open(input_file):
            line_array = line.strip().split('\t')
            if len(line_array) < 2:
                word_cnt = len(word_list)
                dep_cnt = len(dep_list)
                word_list_str = " ".join(word_list)
                pos_list_str = " ".join(pos_list)
                dep_list_str = " ".join(dep_list)
                output_lst.append(
                    "{}\t{}\t{} {} {}".format(word_list_str, pos_list_str, word_cnt, dep_cnt, dep_list_str))
                word_list = []
                pos_list = []
                if len(word_list) != len(pos_list):
                    print('Wrong')
                    print(line)
                dep_list = []
            else:
                word_list.append(str(line_array[1]))
                if line_array[6] != 'ROOT':
                    dep_list.append(
                        "{}|{}|{}".format(int(line_array[5]) - 1, int(line_array[0]) - 1, FnScope.de2id[line_array[6]]))
                pos_list.append(str(FnScope.pos2id[line_array[3]]))

        return output_lst

    init_dict(input_file_1)
    init_dict(input_file_2)
    init_dict(input_file_3)
    init_dict(input_file_4)

    print(FnScope.de2id)
    print(FnScope.pos2id)
    print(FnScope.de_idx)
    print(FnScope.pos_idx)

    s_sample_lst = get_write_lst(input_file_1)
    t_sample_lst = get_write_lst(input_file_2)
    s_eval_lst = get_write_lst(input_file_3)
    t_eval_lst = get_write_lst(input_file_4)

    # fout
    fout = open(input_file_1 + '.out', 'w')
    for source, target in zip(s_sample_lst, t_sample_lst):
        fout.write(source + '\t' + target + '\n')
    fout.flush()
    fout.close()

    fout = open(input_file_3 + '.out', 'w')
    for source, target in zip(s_eval_lst, t_eval_lst):
        fout.write(source + '\t' + target + '\n')
    fout.flush()
    fout.close()

if __name__ == '__main__':
    main()
