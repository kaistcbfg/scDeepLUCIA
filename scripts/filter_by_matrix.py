#!/usr/bin/env python

import sys
import gzip
import itertools 

import numpy

def main():
	if len(sys.argv) < 5:
		print("Usage : " + sys.argv[0] + " [contact_matrix_npy_filename] [count_quantile_cutoff] [prediction_txtgz_filename] [filtered_prediction_txtgz_filename]")
		sys.exit()
	else:
		contact_matrix_npy_filename = sys.argv[1]
		count_quantile_cutoff = float(sys.argv[2])
		prediction_txtgz_filename = sys.argv[3]
		filtered_prediction_txtgz_filename = sys.argv[4]
		filter_by_matrix(contact_matrix_npy_filename,count_quantile_cutoff,prediction_txtgz_filename,filtered_prediction_txtgz_filename)


def filter_by_matrix(contact_matrix_npy_filename,count_quantile_cutoff,prediction_txtgz_filename,filtered_prediction_txtgz_filename):
	contact_matrix = numpy.load(contact_matrix_npy_filename,mmap_mode="r")
	contact_cutoff = max(1.0,numpy.quantile(contact_matrix,count_quantile_cutoff))

	prediction_gen = load_prediction(prediction_txtgz_filename)
	filtered_prediction_gen = filter(lambda prediction : contact_matrix[int(prediction[2]),int(prediction[3])] >=contact_cutoff ,prediction_gen)
	write_filtered_prediction(filtered_prediction_gen,filtered_prediction_txtgz_filename)


def write_filtered_prediction(filtered_prediction_gen,filtered_prediction_txtgz_filename):
	with gzip.open(filtered_prediction_txtgz_filename,"wt") as filtered_prediction_txtgz_file:
		filtered_prediction_txtgz_file.write("sample\tchrom\tindex_one\tindex_two\tprob\n")
		for filtered_prediction in filtered_prediction_gen:
			filtered_prediction_txtgz_file.write("\t".join(filtered_prediction) + "\n")


def load_prediction(prediction_txtgz_filename):
	with gzip.open(prediction_txtgz_filename,"rt") as prediction_txtgz_file:
		for rawline in itertools.islice(prediction_txtgz_file,1,None):
			fields = rawline.strip().split()
			yield fields


if __name__ == "__main__":
	main()
