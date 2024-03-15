#!/usr/bin/env python

import gzip
import itertools 

from pathlib import Path

import numpy

################################################################################
# frontest-end for data feeding 
################################################################################

def gen_seq_epi_con_dataset(config_filename, n2p, marker_type ,metric_type , genome_version):
	#feature_dir = Path("/tmp/temp_feature/")
	feature_dir = Path("/work/dy109/scDeepLUCIA-v2/current/deeplucia_train/feature")
	chrom_sample_list = load_dataset_config(config_filename)

	loop_status_dirname = feature_dir / "loop_control" / genome_version
	seq_array_dirname = feature_dir / "sliced_seq_array" / genome_version / "isHC"
	epi_array_dirname = feature_dir / "sliced_epi_array" / genome_version
	con_array_dirname = feature_dir / "sliced_con_array" / genome_version

	chrom_to_seq_array = load_seq_array_dir(chrom_sample_list, seq_array_dirname)
	chrom_sample_to_epi_array = load_epi_array_dir(chrom_sample_list , marker_type , epi_array_dirname)
	chrom_sample_to_con_array = load_con_array_dir(chrom_sample_list , metric_type , con_array_dirname)

	rated_loop_candidate_gen = gen_rated_loop_candidate(chrom_sample_list,n2p,loop_status_dirname)

	for _,chunk in itertools.groupby(enumerate(rated_loop_candidate_gen) , lambda x : x[0]//200):
		loop_candidate_list = [loop_candidate for i,loop_candidate in chunk]
		yield extract_seq_epi_con_dataset(loop_candidate_list, chrom_to_seq_array, chrom_sample_to_epi_array, chrom_sample_to_con_array)


def sample_seq_epi_con_dataset(config_filename, n2p, marker_type ,metric_type, genome_version , sampling_count):
	#feature_dir = Path("/tmp/temp_feature/")
	feature_dir = Path("/work/dy109/scDeepLUCIA-v2/current/deeplucia_train/feature")
	chrom_sample_list = load_dataset_config(config_filename)

	loop_status_dirname = feature_dir / "loop_control" / genome_version
	seq_array_dirname = feature_dir / "sliced_seq_array" / genome_version / "isHC"
	epi_array_dirname = feature_dir / "sliced_epi_array" / genome_version
	con_array_dirname = feature_dir / "sliced_con_array" / genome_version

	chrom_to_seq_array = load_seq_array_dir(chrom_sample_list, seq_array_dirname)
	chrom_sample_to_epi_array = load_epi_array_dir(chrom_sample_list , marker_type , epi_array_dirname)
	chrom_sample_to_con_array = load_con_array_dir(chrom_sample_list , metric_type , con_array_dirname)

	sample_rated_loop_candidate_list = load_sample_rated_loop_candidate(chrom_sample_list,n2p,loop_status_dirname,sampling_count)

	return extract_seq_epi_con_dataset(sample_rated_loop_candidate_list, chrom_to_seq_array, chrom_sample_to_epi_array, chrom_sample_to_con_array)


def gen_nonrepeat_seq_epi_con_dataset(config_filename, n2p, marker_type,metric_type, genome_version):
	#feature_dir = Path("/tmp/temp_feature/")
	feature_dir = Path("/work/dy109/scDeepLUCIA-v2/current/deeplucia_train/feature")
	chrom_sample_list = load_dataset_config(config_filename)


	loop_status_dirname = feature_dir / "loop_control" / genome_version
	seq_array_dirname = feature_dir / "sliced_seq_array" / genome_version / "isHC"
	epi_array_dirname = feature_dir / "sliced_epi_array" / genome_version
	con_array_dirname = feature_dir / "sliced_con_array" / genome_version

	chrom_to_seq_array = load_seq_array_dir(chrom_sample_list, seq_array_dirname)
	chrom_sample_to_epi_array = load_epi_array_dir(chrom_sample_list , marker_type , epi_array_dirname)
	chrom_sample_to_con_array = load_con_array_dir(chrom_sample_list , metric_type , con_array_dirname)

	nonrepeat_rated_loop_candidate_gen = gen_nonrepeat_rated_loop_candidate(chrom_sample_list , n2p , loop_status_dirname)

	for _,chunk in itertools.groupby(enumerate(nonrepeat_rated_loop_candidate_gen) , lambda x : x[0]//1000):
		loop_candidate_list = [loop_candidate for i,loop_candidate in chunk]
		yield extract_seq_epi_con_dataset(loop_candidate_list, chrom_to_seq_array, chrom_sample_to_epi_array, chrom_sample_to_con_array)



################################################################################
# load chrom,sample dataset
################################################################################

def load_dataset_config(config_filename):
	chrom_sample_list = []
	with open(config_filename) as config_file:
		for rawline in itertools.islice(config_file,1,None):
			chrom_sample = tuple(rawline.strip().split())
			chrom_sample_list.append(chrom_sample)

	return chrom_sample_list



################################################################################
# load directory for the input feature matrix
################################################################################

def load_seq_array_dir(chrom_sample_list, seq_array_dirname):
	chrom_to_seq_array = {}
	for chrom,_ in chrom_sample_list : 
		seq_array_base_filename = "isHC.seq_onehot." + chrom + ".npy"
		seq_array_filename =  seq_array_dirname / seq_array_base_filename
		seq_array = numpy.load(seq_array_filename,mmap_mode="r")
		chrom_to_seq_array[chrom] = seq_array

	return chrom_to_seq_array


def load_epi_array_dir(chrom_sample_list , marker_type , epi_array_dirname):
	chrom_sample_to_epi_array = {}
	for chrom_sample in chrom_sample_list:
		chrom,sample = chrom_sample
		epi_array_base_filename = marker_type + "." + chrom + ".npy"
		epi_array_filename = epi_array_dirname / sample / epi_array_base_filename
		epi_array = numpy.load(epi_array_filename,mmap_mode="r")
		chrom_sample_to_epi_array[chrom_sample] = epi_array

	return chrom_sample_to_epi_array


def load_con_array_dir(chrom_sample_list , metric_type , con_array_dirname):
	chrom_sample_to_con_array = {}
	for chrom_sample in chrom_sample_list:
		chrom,sample = chrom_sample
		con_array_base_filename = metric_type + "." + chrom + ".npy"
		con_array_filename = con_array_dirname / sample / con_array_base_filename
		con_array = numpy.load(con_array_filename,mmap_mode="r")
		chrom_sample_to_con_array[chrom_sample] = con_array

	return chrom_sample_to_con_array



################################################################################
# make infinite/finite numbers of loops 
################################################################################

def gen_rated_loop_candidate(chrom_sample_list , n2p , loop_status_dirname):
	while True:
		chrom_sample_index = numpy.random.choice(len(chrom_sample_list))
		chrom,sample = chrom_sample_list[chrom_sample_index]
		loop_status_base_filename = sample + "." + chrom + ".stat.gz"
		loop_status_filename = loop_status_dirname / sample / loop_status_base_filename
		for loop_candidate in gen_loop_candidate_from_single(loop_status_filename,n2p):
			yield loop_candidate


def load_sample_rated_loop_candidate(chrom_sample_list,n2p,loop_status_dirname,sampling_count):
	rated_loop_candidate_list = load_rated_loop_candidate(chrom_sample_list,n2p,loop_status_dirname)
	sample_rated_loop_candidate_list = rated_loop_candidate_list[:(n2p+1)*sampling_count]
	return sample_rated_loop_candidate_list


def gen_nonrepeat_rated_loop_candidate(chrom_sample_list , n2p , loop_status_dirname):
	for chrom_sample in chrom_sample_list:
		chrom,sample = chrom_sample
		loop_status_base_filename = sample + "." + chrom + ".stat.gz"
		loop_status_filename = loop_status_dirname / sample / loop_status_base_filename
		for loop_candidate in gen_loop_candidate_from_single(loop_status_filename,n2p):
			yield loop_candidate


def gen_scanning_loop_candidate(chrom,sample,scan_start,scan_end):
	#for pair in itertools.filterfalse( lambda pair : 4<pair[1]-pair[0] <400 , itertools.combinations(range(scan_start,scan_end),2)):
	for pair in filter( lambda pair : 4<pair[1]-pair[0] <400 , itertools.combinations(range(scan_start,scan_end),2)):
		loop_candidate = (chrom,sample,pair,0)
		yield loop_candidate



################################################################################
# extract seq,epi,con feature for given candidates.
################################################################################

def extract_seq_epi_con_dataset(loop_candidate_list, chrom_to_seq_array, chrom_sample_to_epi_array, chrom_sample_to_con_array):
	numpy.random.shuffle(loop_candidate_list)
	label = numpy.asarray([ loop_candidate[3] for loop_candidate in loop_candidate_list ])
	seq_feature_one,seq_feature_two = take_seq_array(chrom_to_seq_array,loop_candidate_list)
	epi_feature_one,epi_feature_two = take_epi_array(chrom_sample_to_epi_array,loop_candidate_list)
	con_feature = take_con_array(chrom_sample_to_con_array,loop_candidate_list)
	return ((seq_feature_one,seq_feature_two,epi_feature_one,epi_feature_two,con_feature),label)


def extract_seq_epi_con_dataset_nonshuffle(loop_candidate_list, chrom_to_seq_array, chrom_sample_to_epi_array, chrom_sample_to_con_array):
	label = numpy.asarray([ loop_candidate[3] for loop_candidate in loop_candidate_list ])
	seq_feature_one,seq_feature_two = take_seq_array(chrom_to_seq_array,loop_candidate_list)
	epi_feature_one,epi_feature_two = take_epi_array(chrom_sample_to_epi_array,loop_candidate_list)
	con_feature = take_con_array(chrom_sample_to_con_array,loop_candidate_list)
	return ((seq_feature_one,seq_feature_two,epi_feature_one,epi_feature_two,con_feature),label)



################################################################################
# generate pos/neg loops from single file/directory 
################################################################################

def gen_loop_candidate_from_single(loop_status_filename,n2p):
	sample,chrom = loop_status_filename.name.split(".")[:2]
	with gzip.open(loop_status_filename,"rt") as loop_status_file:
		for rawline	in itertools.islice(loop_status_file,1,None):
			fields = rawline.strip().split()
			pos_pair = tuple(map(int,fields[1].split(",")))
			pos_loop_candidate = (chrom,sample,pos_pair,1)
			yield pos_loop_candidate

			neg_pair_str_list = fields[2].split(";")
			for neg_pair_str_index in numpy.random.choice(len(neg_pair_str_list),size=n2p):
				neg_pair_str = neg_pair_str_list[neg_pair_str_index]
				neg_pair = tuple(map(int,neg_pair_str.split(",")))
				neg_loop_candidate = (chrom,sample,neg_pair,0)
				yield neg_loop_candidate


def load_rated_loop_candidate(chrom_sample_list,n2p,loop_status_dirname):
	rated_loop_candidate_list = []
	for chrom,sample in chrom_sample_list:
		loop_status_base_filename = sample + "." + chrom + ".stat.gz"
		loop_status_filename = loop_status_dirname / sample / loop_status_base_filename
		for loop_candidate in gen_loop_candidate_from_single(loop_status_filename,n2p):
			rated_loop_candidate_list.append(loop_candidate)

	return rated_loop_candidate_list



################################################################################
# slice and stack seq,epi features from mmap matrix based on loop candidate. 
################################################################################

def take_seq_array(chrom_to_seq_array,loop_candidate_list):
	seq_feature_array_one_list = []
	seq_feature_array_two_list = []

	for loop_candidate in loop_candidate_list:
		#print(loop_candidate)
		chrom = loop_candidate[0]
		index_one,index_two = loop_candidate[2]
		seq_feature_array_one = chrom_to_seq_array[chrom][index_one]
		seq_feature_array_two = chrom_to_seq_array[chrom][index_two]
		seq_feature_array_one_list.append(seq_feature_array_one)
		seq_feature_array_two_list.append(seq_feature_array_two)

	seq_feature_one = numpy.stack(seq_feature_array_one_list,axis=0)
	seq_feature_two = numpy.stack(seq_feature_array_two_list,axis=0)
	return seq_feature_one,seq_feature_two


def take_epi_array(chrom_sample_to_epi_array,loop_candidate_list):
	epi_feature_array_one_list = []
	epi_feature_array_two_list = []

	for loop_candidate in loop_candidate_list:
		chrom = loop_candidate[0]
		sample = loop_candidate[1]
		chrom_sample = (chrom,sample)
		index_one,index_two = loop_candidate[2]
		epi_feature_array_one = chrom_sample_to_epi_array[chrom_sample][index_one]
		epi_feature_array_two = chrom_sample_to_epi_array[chrom_sample][index_two]
		epi_feature_array_one_list.append(epi_feature_array_one)
		epi_feature_array_two_list.append(epi_feature_array_two)

	epi_feature_one = numpy.stack(epi_feature_array_one_list,axis=0)
	epi_feature_two = numpy.stack(epi_feature_array_two_list,axis=0)
	return epi_feature_one,epi_feature_two


def take_con_array(chrom_sample_to_con_array,loop_candidate_list):
	con_feature_value_list = []

	for loop_candidate in loop_candidate_list:
		chrom = loop_candidate[0]
		sample = loop_candidate[1]
		chrom_sample = (chrom,sample)
		index_pair = loop_candidate[2]
		con_feature_value = chrom_sample_to_con_array[chrom_sample][index_pair]
		con_feature_value_list.append(con_feature_value)

	con_feature = numpy.array(con_feature_value_list).reshape(-1,1)
	return con_feature
