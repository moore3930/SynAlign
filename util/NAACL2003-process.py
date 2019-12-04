import os
import re

def precess0():

    path = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French-2/training/'
    file_path = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French-2/FilePairs.training'
    fout = open(path + '/en-fr-lower.txt', 'w')

    for file_pair in open(file_path):
        en_file = file_pair.strip().split(' ')[0]
        fr_file = file_pair.strip().split(' ')[1]
        for en_line, fr_line in zip(open(path + en_file, encoding='cp1252'), open(path + fr_file, encoding='cp1252')):
            en_line = en_line.strip().lower()
            fr_line = fr_line.strip().lower()
            fout.write(en_line + '\t' + fr_line + '\n')
    fout.flush()
    fout.close()

    return

def precess1():

    path = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French-2/training'

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.e' in file:
                files.append(os.path.join(r, file))
            if '.f' in file:
                files.append(os.path.join(r, file))

    fout = open(path + '/en-fr.txt', 'w')
    files_set = set(files)

    for f in files:

        prefix = f[:-2]
        if f.endswith(".e") and (prefix + '.f') in files_set:
            fe = open(f, encoding='cp1252')
            ff = open(prefix + '.f', encoding='cp1252')
            for e, f in zip(fe, ff):
                e = e.strip()
                f = f.strip()
                fout.write(e + '\t' + f + '\n')
            fe.close()
            ff.close()
        if f.endswith(".f") and (prefix + '.e') in files_set:
            ff = open(f, encoding='cp1252')
            fe = open(prefix + '.e', encoding='cp1252')
            print(f)
            print(prefix + 'e')
            for e, f in zip(fe, ff):
                e = e.strip()
                f = f.strip()
                fout.write(e + '\t' + f + '\n')
            fe.close()
            ff.close()
        if prefix + 'e' in files_set:
            files_set.remove(prefix + '.e')
        if prefix + 'f' in files_set:
            files_set.remove(prefix + '.f')

    fout.flush()
    fout.close()

    return

def precess2():
    eval_path_e = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French/test/test.e'
    eval_path_f = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French/test/test.f'
    eval_path_out = '/Users/Moore/PycharmProjects/SynAlign/data/en-fr-test.txt'
    fout = open(eval_path_out, 'w')
    for e_line, f_line in zip(open(eval_path_e, encoding='cp1252'), open(eval_path_f, encoding='cp1252')):
        e_line = e_line.strip().split('> ')[1].split(' <')[0]
        f_line = f_line.strip().split('> ')[1].split(' <')[0]
        fout.write(e_line + '\t' + f_line + '\n')
    fout.flush()
    fout.close()

def process3():
    eval_path_wa = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French/answers/test.wa.nonullalign'
    fout = open('/Users/Moore/PycharmProjects/SynAlign/data/en-fr-test-wa.txt', 'w')
    num = 0
    start_char = '$'
    wa_lst = []
    for line in open(eval_path_wa):
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

precess0()


