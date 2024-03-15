#!/usr/bin/env python

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers

from tensorflow.keras.models import Model

from tensorflow.keras.layers import concatenate , Conv2D , Dense , Dropout , Flatten , Input , MaxPooling2D , Reshape , ZeroPadding2D

#marker_type_to_epi_channel = {"05mark":5,"06mark":6,"12mark":12,"13mark":13,"pairst":6}
marker_type_to_epi_channel = {"05mark":5,"06mark":6,"12mark":12,"13mark":13,"pairst":6,"romepi":6,"openonly":1,"pairst_005M":6,"pairst_100K":6,"dnase_only_005M":1,"dnase_only_100K":1,"hicar_only":1,"r2_030M":1}



def var_model_seq_epi_con_20230920_v1(marker_type):
	#epi_channel = int(marker_type[:2])
	epi_channel = marker_type_to_epi_channel[marker_type]
	seq_dim = (5000,4)
	epi_dim = (200,epi_channel)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")
	con_input = layers.Input(shape=(1),name="con_feature")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=128, kernel_size=40, padding='same', activation=tf.nn.swish)(seq_path)
	epi_path = layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.swish)(epi_path)

	seq_path = layers.MaxPooling1D(50)(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=tf.nn.swish,name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10, padding='same',activation=tf.nn.swish)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)

	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(3,activation=tf.nn.swish)(combined_path)
	combined_path = layers.concatenate([combined_path,con_input],axis=1)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two,con_input),outputs=combined_path)

	return model



def var_model_seq_epi_20230920_v1(marker_type):
	#epi_channel = int(marker_type[:2])
	epi_channel = marker_type_to_epi_channel[marker_type]
	seq_dim = (5000,4)
	epi_dim = (200,epi_channel)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")
	con_input = layers.Input(shape=(1),name="con_feature")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=128, kernel_size=40, padding='same', activation=tf.nn.swish)(seq_path)
	epi_path = layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.swish)(epi_path)

	seq_path = layers.MaxPooling1D(50)(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=tf.nn.swish,name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10, padding='same',activation=tf.nn.swish)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)

	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(4,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two,con_input),outputs=combined_path)

	return model


################################################################################################################################################



def var_model_seq_epi_con_20230919_v1(marker_type):
	#epi_channel = int(marker_type[:2])
	epi_channel = marker_type_to_epi_channel[marker_type]
	seq_dim = (5000,4)
	epi_dim = (200,epi_channel)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")
	con_input = layers.Input(shape=(1),name="con_feature")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=128, kernel_size=40, padding='same', activation=tf.nn.swish)(seq_path)
	epi_path = layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.swish)(epi_path)

	seq_path = layers.MaxPooling1D(50)(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=tf.nn.swish,name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10, padding='same',activation=tf.nn.swish)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)


	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1023,activation=tf.nn.swish)(combined_path)
	combined_path = layers.concatenate([combined_path,con_input],axis=1)
	combined_path = layers.Dense(32,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two,con_input),outputs=combined_path)

	return model



def var_model_seq_epi_20230919_v1(marker_type):
	#epi_channel = int(marker_type[:2])
	epi_channel = marker_type_to_epi_channel[marker_type]
	seq_dim = (5000,4)
	epi_dim = (200,epi_channel)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")
	con_input = layers.Input(shape=(1),name="con_feature")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=128, kernel_size=40, padding='same', activation=tf.nn.swish)(seq_path)
	epi_path = layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.swish)(epi_path)

	seq_path = layers.MaxPooling1D(50)(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=tf.nn.swish,name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10, padding='same',activation=tf.nn.swish)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)


	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1024,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(32,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two,con_input),outputs=combined_path)

	return model


################################################################################################################################################


def var_model_seq_epi_20210709_v1(marker_type):
	#epi_channel = int(marker_type[:2])
	epi_channel = marker_type_to_epi_channel[marker_type]
	seq_dim = (5000,4)
	epi_dim = (200,epi_channel)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")
	con_input = layers.Input(shape=(1),name="con_feature")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=128, kernel_size=40, padding='same', activation=tf.nn.swish)(seq_path)
	epi_path = layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.swish)(epi_path)

	seq_path = layers.MaxPooling1D(50)(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=tf.nn.swish,name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10, padding='same',activation=tf.nn.swish)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)


	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two,con_input),outputs=combined_path)

	return model


def var_model_seq_epi_con_20230901_v1(marker_type):
	#epi_channel = int(marker_type[:2])
	epi_channel = marker_type_to_epi_channel[marker_type]
	seq_dim = (5000,4)
	epi_dim = (200,epi_channel)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")
	con_input = layers.Input(shape=(1),name="con_feature")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=128, kernel_size=40, padding='same', activation=tf.nn.swish)(seq_path)
	epi_path = layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=tf.nn.swish)(epi_path)

	seq_path = layers.MaxPooling1D(50)(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=tf.nn.swish,name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10, padding='same',activation=tf.nn.swish)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)

	combined_path = layers.concatenate([combined_path,con_input],axis=1)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two,con_input),outputs=combined_path)

	return model


	
