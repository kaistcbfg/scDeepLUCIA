#!/usr/bin/env python

import sys
import gzip
import itertools 
import bisect

def main():
	if len(sys.argv) < 4:
		print ("Usage : " + sys.argv[0] + " [chromwise_anchor_index_bedgz_filename] [prediction_txtgz_filename] [loop_bedpe_filename]")
		sys.exit()
	else:
		chromwise_anchor_index_bedgz_filename = sys.argv[1]
		prediction_txtgz_filename = sys.argv[2]
		loop_bedpe_filename = sys.argv[3]
		extract_loop(chromwise_anchor_index_bedgz_filename , prediction_txtgz_filename, loop_bedpe_filename)


def extract_loop(chromwise_anchor_index_bedgz_filename , prediction_txtgz_filename, loop_bedpe_filename):
	chrom_index_to_interval = load_anchor_index(chromwise_anchor_index_bedgz_filename)
	chrom_index_pair_list = load_prediction(prediction_txtgz_filename)
	interval_pair_gen = map_interval_to_prediction(chrom_index_to_interval,chrom_index_pair_list)
	write_loop_bedpe(interval_pair_gen,loop_bedpe_filename)



def write_loop_bedpe(interval_pair_gen,loop_bedpe_filename):
	with open(loop_bedpe_filename,"wt") as loop_bedpe_file:
		for interval_pair in interval_pair_gen:
			interval_one,interval_two = interval_pair
			loop_bedpe_file.write("\t".join(interval_one) + "\t" + "\t".join(interval_two) + "\n")


def map_interval_to_prediction(chrom_index_to_interval,chrom_index_pair_list):
	for chrom_index_pair in chrom_index_pair_list:
		chrom_index_one , chrom_index_two = chrom_index_pair
		interval_one = chrom_index_to_interval[chrom_index_one]
		interval_two = chrom_index_to_interval[chrom_index_two]
		interval_pair = (interval_one,interval_two)
		yield interval_pair


def load_prediction(prediction_txtgz_filename):
	chrom_index_pair_list = []
	with gzip.open(prediction_txtgz_filename,"rt") as prediction_txtgz_file:
		for rawline in itertools.islice(prediction_txtgz_file,1,None):
			fields = rawline.strip().split()
			chrom = fields[1]
			index_one = int(fields[2])
			index_two = int(fields[3])
			chrom_index_one = (chrom,index_one)
			chrom_index_two = (chrom,index_two)
			chrom_index_pair =(chrom_index_one , chrom_index_two)
			bisect.insort(chrom_index_pair_list,chrom_index_pair)

	return chrom_index_pair_list 


def load_anchor_index(chromwise_anchor_index_bedgz_filename):
	chrom_index_to_interval = {}
	with gzip.open(chromwise_anchor_index_bedgz_filename,"rt") as chromwise_anchor_index_bedgz_file:
		for rawline in chromwise_anchor_index_bedgz_file:
			fields = rawline.strip().split()
			chrom = fields[0]
			index = int(fields[3])
			interval = fields[:3]
			interval[0] = interval[0][3:] # for juicer, "chr" should be removed.
			chrom_index = (chrom,index)
			chrom_index_to_interval[chrom_index] = interval

	return chrom_index_to_interval


if __name__ == "__main__":
	main()
