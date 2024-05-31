#!/usr/bin/env python

import gzip
import itertools 

from pathlib import Path
from tensorflow import keras
import tensorflow 

import numpy 
import pandas

from sklearn.cluster import AgglomerativeClustering

###################################################################################################

def load_model(model_filename):
	model = keras.models.load_model(model_filename, custom_objects={'swish':tensorflow.nn.swish})
	return model


def get_directory(feature_dirname,chrom,sample,genome_version):
    feature_dir = Path(feature_dirname)
    seq_array_dirname = feature_dir / "sliced_seq_array" / genome_version / "isHC"
    epi_array_dirname = feature_dir / "sliced_epi_array" / genome_version
    return seq_array_dirname,epi_array_dirname


def get_con_cutoff(con_array,con_rev_quantile=0.005):
	con_quantile = 1.0-con_rev_quantile
	con_cutoff = max(1,numpy.quantile(con_array,con_quantile))
	return con_cutoff


def get_prob_cutoff(loop_df,prob_rev_quantile=0.005):
	prob_quantile = 1.0-prob_rev_quantile
	prob_cutoff = numpy.quantile(loop_df["prob"].values,prob_quantile)
	return prob_cutoff


def filter_by_quantile(loop_df,con_array,con_rev_quantile=0.005,prob_rev_quantile=0.005):
	con_cutoff = get_con_cutoff(con_array,con_rev_quantile)
	prob_cutoff = get_prob_cutoff(loop_df,prob_rev_quantile)
	row_list = []

	for _,row in loop_df.iterrows():
		if (con_array[row.index_one,row.index_two]) >= con_cutoff and row.prob >= prob_cutoff:
			row_list.append(row)

	filtered_loop_df = pandas.DataFrame(row_list,columns=loop_df.columns)
	return filtered_loop_df


def filter_by_distance(loop_df,min_dist=6,max_dist=300):
	filtered_loop_df = loop_df.query( str(min_dist)+ "<=index_two-index_one<="+str(max_dist))
	return filtered_loop_df


def form_loop_cluster(loop_df,distance_threshold=3):
	row_count = len(loop_df.index)
	if row_count == 1:
		clustered_loop_df = loop_df.copy(deep=True)
	else:
		column_list = loop_df.columns
		row_list = []
		coordinate = loop_df[["index_one","index_two"]]
		agg_cluster = AgglomerativeClustering(n_clusters = None,distance_threshold=distance_threshold,linkage="single").fit(coordinate)
		loop_df["cluster"] = agg_cluster.labels_
		for cluster,loop_cluster_subdf in loop_df.groupby("cluster"):
			row = loop_cluster_subdf.drop("cluster",axis=1).loc[loop_cluster_subdf["prob"].idxmax()]
			row_list.append(row)
			
		clustered_loop_df = pandas.DataFrame(row_list,columns=column_list)
	
	return clustered_loop_df


def save_as_bedpe(loop_df,chrom,resolution,loop_bedpe_filename):
	tmp_df = pandas.DataFrame(index=loop_df.index)
	tmp_df["chrom_one"] = chrom
	tmp_df["start_one"] = loop_df["index_one"] * resolution
	tmp_df["end_one"] = loop_df["index_one"] * resolution + resolution
	tmp_df["chrom_two"] = chrom
	tmp_df["start_two"] = loop_df["index_two"] * resolution
	tmp_df["end_two"] = loop_df["index_two"] * resolution + resolution
	tmp_df.to_csv(loop_bedpe_filename,sep="\t",header=False,index=False)

