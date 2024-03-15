#!/usr/bin/env python

import sys
import gzip 

import numpy 
import tensorflow as tf 
import itertools 

from pathlib import Path

from tensorflow import keras

from deeplucia_toolkit import make_model
from deeplucia_toolkit import make_dataset

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


def main():
	if len(sys.argv) < 10:
		print("Usage : " + sys.argv[0] + " [keras_model_filename] [chrom] [sample] [scan_start] [scan_end] [marker_type] [metric_type] [genome_version] [prediction_filename]")
		sys.exit()
	else:
		keras_model_filename = sys.argv[1]
		chrom = sys.argv[2]
		sample = sys.argv[3]
		scan_start = int(sys.argv[4])
		scan_end = int(sys.argv[5])
		marker_type = sys.argv[6]
		metric_type = sys.argv[7]
		genome_version = sys.argv[8]
		prediction_filename  = sys.argv[9]

		deeplucia_scan(keras_model_filename,chrom,sample,scan_start,scan_end,marker_type,metric_type,genome_version,prediction_filename)


def deeplucia_scan(keras_model_filename,chrom,sample,scan_start,scan_end,marker_type,metric_type,genome_version,prediction_filename):
	model = keras.models.load_model(keras_model_filename, custom_objects={'swish':tf.nn.swish})

	seq_array_dirname,epi_array_dirname,con_array_dirname = get_directory( chrom,sample,genome_version)
	chrom_sample_list = [(chrom,sample)]
	chrom_to_seq_array = make_dataset.load_seq_array_dir(chrom_sample_list, seq_array_dirname)
	chrom_sample_to_epi_array = make_dataset.load_epi_array_dir(chrom_sample_list , marker_type , epi_array_dirname)
	chrom_sample_to_con_array = make_dataset.load_con_array_dir(chrom_sample_list , metric_type , con_array_dirname)


	scanning_loop_candidate_gen = make_dataset.gen_scanning_loop_candidate(chrom,sample,scan_start,scan_end)

	pair_list = []
	prob_list = []

	for _,chunk in itertools.groupby(enumerate(scanning_loop_candidate_gen) , lambda x : x[0]//512):
		loop_candidate_list = []
		for _,loop_candidate in chunk:
			pair = loop_candidate[2]
			loop_candidate_list.append(loop_candidate)
			pair_list.append(pair)

		batched_feature,_ = make_dataset.extract_seq_epi_con_dataset_nonshuffle(loop_candidate_list, chrom_to_seq_array, chrom_sample_to_epi_array, chrom_sample_to_con_array)
		output = model.predict(batched_feature)
		batched_prob_pred = numpy.squeeze(output,axis=1)

		for prob in batched_prob_pred:
			prob_list.append(prob)

	if len(pair_list) == len(prob_list):
		with gzip.open(prediction_filename,"wt") as prediction_file:
			prediction_file.write("sample\tchrom\tindex_one\tindex_two\tprob\n")
			for pair,prob in zip(pair_list,prob_list):
				prediction_file.write("\t".join(map(str,[ sample, chrom, pair[0],pair[1], prob])) + "\n")

	else:
		print("length not matched")





def get_directory( chrom,sample,genome_version):
	feature_dir = Path("/tmp/temp_feature/")
	seq_array_dirname = feature_dir / "sliced_seq_array" / genome_version / "isHC"
	epi_array_dirname = feature_dir / "sliced_epi_array" / genome_version
	con_array_dirname = feature_dir / "sliced_con_array" / genome_version

	return seq_array_dirname,epi_array_dirname,con_array_dirname



if __name__ == "__main__":
	main()
