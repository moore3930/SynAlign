for BATCH in 48 64
do
	for NEG_NUM in 16 32
	do
		for LR in 0.002 0.001 0.0005
		do
			for ALPHA in 0.5 1 1.5
			do	
				CUDA_VISIBLE_DEVICES=1 python SynAlignSent.py -maxsentlen=60 -epoch=15 -embed_dim=256 -result_path='data/en-fr.gd' -train_data='data/en-fr/en-fr-sample-merge.txt' -eval_data='data/en-fr/en-fr-test.txt' -eval_data_wa='data/en-fr/en-fr-test-wa.txt' -output_prefix='data/en-fr' -num_neg=${NEG_NUM} -lr=${LR} -alpha=${ALPHA} -batch=${BATCH}
			done
		done
	done
done
