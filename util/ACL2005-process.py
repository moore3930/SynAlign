import os

def precess1():

    filelist_path = '/Users/Moore/Documents/paper/CWA/material/English-Romanian ACL2005/Romanian-English.train/FilePairs.training'
    output_path = '/Users/Moore/PycharmProjects/SynAlign/data/ro-en/ro-en-train.txt'
    cur_dir = '/Users/Moore/Documents/paper/CWA/material/English-Romanian ACL2005/Romanian-English.train/training/'
    fout = open(output_path, 'a')
    for line in open(filelist_path):
        line_array = line.strip().split(' ')
        r_fin = open(cur_dir + line_array[0], encoding='cp1252')
        e_fin = open(cur_dir + line_array[1], encoding='cp1252')
        for ro, en in zip(r_fin, e_fin):
            ro = ro.strip()
            en = en.strip()
            fout.write(ro + '\t' + en + '\n')
        r_fin.close()
        e_fin.close()
    fout.flush()
    fout.close()


def precess2():
    filelist_path = '/Users/Moore/Documents/paper/CWA/material/English-Romanian ACL2005/Romanian-English.test/FilePairs.test'
    eval_path_out = '/Users/Moore/PycharmProjects/SynAlign/data/ro-en/ro-en-test.txt'
    cur_dir = '/Users/Moore/Documents/paper/CWA/material/English-Romanian ACL2005/Romanian-English.test/'

    fout = open(eval_path_out, 'w')

    for line in open(filelist_path):
        line_array = line.strip().split(' ')
        r_fin = open(cur_dir + line_array[0], encoding='cp1252')
        e_fin = open(cur_dir + line_array[1], encoding='cp1252')
        for ro, en in zip(r_fin, e_fin):
            ro = ro.strip().split('> ')[1].split(' </s>')[0]
            en = en.strip().split('> ')[1].split(' </s>')[0]
            fout.write(ro + '\t' + en + '\n')
        r_fin.close()
        e_fin.close()
    fout.flush()
    fout.close()


def process3():
    eval_path_wa = '/Users/Moore/Documents/paper/CWA/material/English-Romanian ACL2005/Romanian-English.test/test.wa.nonullalign'
    fout = open('/Users/Moore/PycharmProjects/SynAlign/data/ro-en/ro-en-test-wa.txt', 'w')
    num = 0
    start_char = '$'
    wa_lst = []
    for line in open(eval_path_wa, encoding='cp1252'):
        line_array = line.strip().split(' ')
        if start_char != line_array[0]:
            num += 1
            start_char = line_array[0]
            if len(wa_lst) > 0:
                for l in wa_lst:
                    fout.write(l + '\n')
                wa_lst = []
        wa_lst.append('num-' + str(num) + ' ' + line_array[1] + ' -> ' + line_array[2] + ' ' + line_array[3])

    if len(wa_lst) > 0:
        for l in wa_lst:
            fout.write(l + '\n')
    fout.flush()
    fout.close()
    return

process3()
