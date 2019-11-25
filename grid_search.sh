for BATCH in 64
do
	for NEG_NUM in 32 48
	do
		for LR in 0.001
		do
			for ALPHA in 0.5 0.6 0.7 0.8 1.0
			do	
				CUDA_VISIBLE_DEVICES=1 python SynAlignSent.py -maxsentlen=60 -epoch=2 -embed_dim=256 -result_path='data/en-fr.gd' -train_data='data/en-fr/en-fr-merge.txt' -eval_data='data/en-fr/en-fr-test.txt' -eval_data_wa='data/en-fr/en-fr-test-wa.txt' -output_prefix='data/en-fr' -num_neg=${NEG_NUM} -lr=${LR} -alpha=${ALPHA} -batch=${BATCH}
			done
		done
	done
done
