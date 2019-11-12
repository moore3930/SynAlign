import sys
from collections import *

def repalce(lst, frq_set, repalce_word):
    for i in range(len(lst)):
        if lst[i] not in frq_set:
            lst[i] = repalce_word
    return lst

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[1] + '.fmt'
    mc = sys.argv[2]
    w_cnt = Counter()
    for line in open(input_file):
        line_array = line.strip().split(' ')
        for w in line_array:
            w_cnt.update({w: 1})
    freq_set = set([item[0] for item in w_cnt.most_common(int(mc))])
    fout = open(output_file, 'w')
    for line in open(input_file):
        lst_tuple = []
        for sent in line.strip().split('\t'):
            line_array = repalce(sent.strip().split(' '), freq_set, 'UNK')
            lst_tuple.append(line_array)
        out_lst = [' '.join(line_array) for line_array in lst_tuple]
        fout.write('\t'.join(out_lst) + '\n')
    fout.flush()
    fout.close()

if __name__ == '__main__':
    main()
