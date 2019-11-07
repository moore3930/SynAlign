import os

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

