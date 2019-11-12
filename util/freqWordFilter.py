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
    s_mc = sys.argv[2]
    t_mc = sys.argv[3]
    sw_cnt = Counter()
    tw_cnt = Counter()
    for line in open(input_file):
        line_array = line.strip().split('\t')
        if len(line_array) != 2:
            print(line_array)
            print('Wrong! Musk be source concat target with tab')
            return
        for w in line_array[0].split(' '):
            sw_cnt.update({w: 1})
        for w in line_array[1].split(' '):
            tw_cnt.update({w: 1})
    s_freq_set = set([item[0] for item in sw_cnt.most_common(int(s_mc))])
    t_freq_set = set([item[0] for item in tw_cnt.most_common(int(t_mc))])

    fout = open(output_file, 'w')
    for line in open(input_file):
        line_array = line.strip().split('\t')
        s_sent_array = repalce(line_array[0].strip().split(' '), s_freq_set, 'UNK')
        t_sent_array = repalce(line_array[1].strip().split(' '), t_freq_set, 'UNK')
        fout.write(' '.join(s_sent_array) + '\t' + ' '.join(t_sent_array) + '\n')
    fout.flush()
    fout.close()

if __name__ == '__main__':
    main()
