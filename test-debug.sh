BATCH=32
NEG_NUM=16
LR=0.002
ALPHA=1.0
CUDA_VISIBLE_DEVICES=1 python SynAlignSent.py -epoch=30 -embed_dim=256 -result_path='data/ro-en.gd' -train_data='data/ro-en/ro-en-merge-2.txt' -eval_data='data/ro-en/ro-en-test.txt' -eval_data_wa='data/ro-en/ro-en-test-wa.txt' -output_prefix='data/ro-en' -num_neg=${NEG_NUM} -lr=${LR} -alpha=${ALPHA} -batch=${BATCH}
