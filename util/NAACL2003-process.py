import os
import re

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
    eval_path_e = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French-1/trial/trial.e'
    eval_path_f = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French-1/trial/trial.f'
    eval_path_out = '/Users/Moore/PycharmProjects/SynAlign/data/en-fr-eval.txt'
    fout = open(eval_path_out, 'w')
    for e_line, f_line in zip(open(eval_path_e, encoding='cp1252'), open(eval_path_f, encoding='cp1252')):
        e_line = e_line.strip().split('> ')[1].split(' <')[0]
        f_line = f_line.strip().split('> ')[1].split(' <')[0]
        fout.write(e_line + '\t' + f_line + '\n')
    fout.flush()
    fout.close()

def process3():
    eval_path_wa = '/Users/Moore/Documents/paper/CWA/material/English-French Hansards corpus - NAACL 2003/English-French-1/trial/trial.wa'
    fout = open('/Users/Moore/PycharmProjects/SynAlign/data/en-fr-wa.txt', 'w')
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
        wa_lst.append('num-' + str(num) + ' ' + line_array[1] + ' -> ' + line_array[2])

    if len(wa_lst) > 0:
        for l in wa_lst:
            fout.write(l + '\n')
    fout.flush()
    fout.close()
    return

process3()


