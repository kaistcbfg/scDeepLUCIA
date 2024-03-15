#!/usr/bin/env python
import sys 
import gzip 

import pandas 
from sklearn.cluster import AgglomerativeClustering

def main():
	if len(sys.argv) < 4:
		print ("Usage : " + sys.argv[0] + " [prediction_txtgz_filename] [distance_threshold] [clustered_prediction_txtgz_filename]")
		sys.exit()
	else:
		prediction_txtgz_filename = sys.argv[1]
		distance_threshold = int(sys.argv[2])
		clustered_prediction_txtgz_filename = sys.argv[3]
		loop_cluster_agg(prediction_txtgz_filename , distance_threshold , clustered_prediction_txtgz_filename)


def loop_cluster_agg(prediction_txtgz_filename , distance_threshold , clustered_prediction_txtgz_filename):
	loop_prediction_df = pandas.read_table(prediction_txtgz_filename)
	clustered_prediction_gen = cluster_covnorm_sigint(loop_prediction_df,distance_threshold)
	write_clustered_prediction(clustered_prediction_gen,clustered_prediction_txtgz_filename)


def write_clustered_prediction(clustered_prediction_gen,clustered_prediction_txtgz_filename):
	with gzip.open(clustered_prediction_txtgz_filename,"wt") as clustered_prediction_txtgz_file:
		clustered_prediction_txtgz_file.write("sample\tchrom\tindex_one\tindex_two\tprob\tmember_list\n")
		for clustered_prediction in clustered_prediction_gen:
			clustered_prediction_txtgz_file.write("\t".join(map(str,clustered_prediction)) + "\n")
	

def cluster_covnorm_sigint(loop_prediction_df,distance_threshold):
	if loop_prediction_df.shape[0] > 1:
		coordinate = loop_prediction_df[["index_one","index_two"]]
		#agg_cluster = AgglomerativeClustering(n_clusters = None, metric="l2", distance_threshold = distance_threshold,linkage="single").fit(coordinate)
		agg_cluster = AgglomerativeClustering(n_clusters = None, affinity="euclidean", distance_threshold = distance_threshold,linkage="single").fit(coordinate)
		loop_prediction_df["cluster"] = agg_cluster.labels_
		for cluster,loop_prediction_cluster_subdf in loop_prediction_df.groupby("cluster"):
			member_list = list(zip(loop_prediction_cluster_subdf.index_one,loop_prediction_cluster_subdf.index_two))
			peak_in_cluster = loop_prediction_cluster_subdf.loc[loop_prediction_cluster_subdf["prob"].idxmax()]
			#print(peak_in_cluster)
			sample,chrom,index_one,index_two,prob,_ = peak_in_cluster.to_list()
			clustered_loop = (sample,chrom,index_one,index_two,prob,member_list)
			yield clustered_loop

	elif loop_prediction_df.shape[0] == 1:
		only_loop = loop_prediction_df.iloc[0]
		sample,chrom,index_one,index_two,prob = only_loop.to_list()
		member_list = [(index_one,index_two)]
		clustered_loop = (sample,chrom,index_one,index_two,prob,member_list)
		yield clustered_loop
		

if __name__ == "__main__" :
	main()



