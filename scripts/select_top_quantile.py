#!/usr/bin/env python

import sys
import gzip
import itertools 

import numpy

def main():
	if len(sys.argv) < 4:
		print("Usage : " + sys.argv[0] + " [prediction_txtgz_filename] [prob_quantile_cutoff] [selected_prediction_txtgz_filename]")
		sys.exit()
	else:
		prediction_txtgz_filename = sys.argv[1]
		prob_quantile_cutoff = float(sys.argv[2])	
		selected_prediction_txtgz_filename = sys.argv[3]
		select_top_quantile(prediction_txtgz_filename , prob_quantile_cutoff , selected_prediction_txtgz_filename)


def select_top_quantile(prediction_txtgz_filename , prob_quantile_cutoff , selected_prediction_txtgz_filename):
	prob_cutoff = get_cutoff(prediction_txtgz_filename , prob_quantile_cutoff)
	prediction_gen = load_prediction(prediction_txtgz_filename)
	selected_prediction_gen = filter(lambda prediction : prob_cutoff <= float(prediction[4]) ,prediction_gen)
	write_selected_prediction(selected_prediction_gen,selected_prediction_txtgz_filename)


def write_selected_prediction(selected_prediction_gen,selected_prediction_txtgz_filename):
	with gzip.open(selected_prediction_txtgz_filename,"wt") as selected_prediction_txtgz_file:
		selected_prediction_txtgz_file.write("sample\tchrom\tindex_one\tindex_two\tprob\n")
		for selected_prediction in selected_prediction_gen:
			selected_prediction_txtgz_file.write("\t".join(selected_prediction) + "\n")


def load_prediction(prediction_txtgz_filename):
	with gzip.open(prediction_txtgz_filename,"rt") as prediction_txtgz_file:
		for rawline in itertools.islice(prediction_txtgz_file,1,None):
			fields = rawline.strip().split()
			yield fields


def get_cutoff(prediction_txtgz_filename , prob_quantile_cutoff):
	prob_list = []
	with gzip.open(prediction_txtgz_filename,"rt") as prediction_txtgz_file:
		for rawline in itertools.islice(prediction_txtgz_file,1,None):
			fields = rawline.strip().split()
			prob = float(fields[4])
			prob_list.append(prob)

	prob_cutoff = numpy.quantile(prob_list,prob_quantile_cutoff)
	print ( "value  @ " + str(prob_quantile_cutoff) + " : " + str(prob_cutoff) )
	
	return prob_cutoff


if __name__ == "__main__":
	main()
