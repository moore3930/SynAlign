import sys
from collections import *

def main():
    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    fin1 = open(input_file_1)
    fin2 = open(input_file_2)
    sent_1 = []
    sent_2 = []
    tmp_sent = []
    for line in fin1:
        if line == '\n':
            sent_1.append(' '.join(tmp_sent).replace('-LRB-', '(').replace('-RRB-', ')'))
            tmp_sent = []
        else:
            tmp_sent.append(line.strip().split('\t')[1])
    for line in fin2:
        sent_2.append(line.strip())
    fin1.close()
    fin2.close()
    print(len(sent_1))
    print(len(sent_2))
    cnt = 0
    for s, t in zip(sent_1, sent_2):
        cnt += 1
        s_array = s.strip().split(' ')
        t_array = t.strip().split(' ')
        if abs(len(s_array) - len(t_array)) > 5:
            print(s)
            print(t)
            print(cnt)
            break

if __name__ == '__main__':
    main()
