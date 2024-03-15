#!/usr/bin/env python

import sys
import gzip
import itertools 
#import functools 


def main():
	if len(sys.argv) < 5:
		print("Usage : " + sys.argv[0] + " [prediction_txtgz_filename] [short_dist] [long_dist] [selected_prediction_txtgz_filename]")
		sys.exit()
	else:
		prediction_txtgz_filename = sys.argv[1]
		short_dist = int(sys.argv[2])	
		long_dist = int(sys.argv[3])	
		selected_prediction_txtgz_filename = sys.argv[4]
		discard_long_short_loop(prediction_txtgz_filename , short_dist , long_dist , selected_prediction_txtgz_filename)


def discard_long_short_loop(prediction_txtgz_filename , short_dist , long_dist , selected_prediction_txtgz_filename):
	prediction_gen = load_prediction(prediction_txtgz_filename)
	selected_prediction_gen = filter(lambda prediction : short_dist<int(prediction[3])-int(prediction[2])<=long_dist ,prediction_gen)
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


if __name__ == "__main__":
	main()
