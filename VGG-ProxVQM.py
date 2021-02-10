import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from IPython.display import clear_output
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dense, Flatten, Add, LeakyReLU, Dropout, SpatialDropout2D
from tensorflow.keras.applications import VGG16

input_shape = (720,1280,3)

with strategy.scope():

    # Init
    VGGLayer = VGG16(weights="imagenet",include_top=False, input_tensor=Input(shape=input_shape))
    for layer in VGGLayer.layers:
        layer.trainable = False
        
    spatialDNN_Conv = Conv2D(128,3, kernel_regularizer='l2', activation=LeakyReLU(alpha=0.1))
    spatialDNN_SpatialDropout = SpatialDropout2D(0.2)
    spatialDNN_MaxPool = MaxPooling2D(2)
    
    temporalDNN_Conv = Conv2D(384,3, kernel_regularizer='l2', activation=LeakyReLU(alpha=0.1))
    temporalDNN_SpatialDropout = SpatialDropout2D(0.2)
    temporalDNN_MaxPool = MaxPooling2D(2)
    
    # Ref
    # t - 1
    t_min_1_ref_in = Input(shape=input_shape, name="t_min_1_ref_in")
    t_min_1_ref = VGGLayer(t_min_1_ref_in)
    t_min_1_ref = spatialDNN_Conv(t_min_1_ref)
    t_min_1_ref = spatialDNN_SpatialDropout(t_min_1_ref)
    t_min_1_ref = spatialDNN_MaxPool(t_min_1_ref)
    
    # t
    t_ref_in_Y = Input(shape=input_shape, name="t_ref_in_Y")
    t_ref_Y = VGGLayer(t_ref_in_Y)
    t_ref_Y = spatialDNN_Conv(t_ref_Y)
    t_ref_Y = spatialDNN_SpatialDropout(t_ref_Y)
    t_ref_Y = spatialDNN_MaxPool(t_ref_Y)


    # t + 1
    t_plus_1_ref_in = Input(shape=input_shape, name="t_plus_1_ref_in")
    t_plus_1_ref = VGGLayer(t_plus_1_ref_in)
    t_plus_1_ref = spatialDNN_Conv(t_plus_1_ref)
    t_plus_1_ref = spatialDNN_SpatialDropout(t_plus_1_ref)
    t_plus_1_ref = spatialDNN_MaxPool(t_plus_1_ref)

    ref_cat = concatenate([t_min_1_ref,t_ref_Y,t_plus_1_ref])
    ref_cat_Conv = temporalDNN_Conv(ref_cat)
    ref_cat_Conv = temporalDNN_SpatialDropout(ref_cat_Conv)
    ref_cat_Conv = temporalDNN_MaxPool(ref_cat_Conv)


    # Dist
    # t - 1
    t_min_1_dist_in = Input(shape=input_shape, name="t_min_1_dist_in")
    t_min_1_dist = VGGLayer(t_min_1_dist_in)
    t_min_1_dist = spatialDNN_Conv(t_min_1_dist)
    t_min_1_dist = spatialDNN_SpatialDropout(t_min_1_dist)
    t_min_1_dist = spatialDNN_MaxPool(t_min_1_dist)
    
     # t
    t_dist_in_Y = Input(shape=input_shape, name="t_dist_in_Y")
    t_dist_Y = VGGLayer(t_dist_in_Y)
    t_dist_Y = spatialDNN_Conv(t_dist_Y)
    t_dist_Y = spatialDNN_SpatialDropout(t_dist_Y)
    t_dist_Y = spatialDNN_MaxPool(t_dist_Y)

     # t + 1
    t_plus_1_dist_in = Input(shape=input_shape, name="t_plus_1_dist_in")
    t_plus_1_dist = VGGLayer(t_plus_1_dist_in)
    t_plus_1_dist = spatialDNN_Conv(t_plus_1_dist)
    t_plus_1_dist = spatialDNN_SpatialDropout(t_plus_1_dist)
    t_plus_1_dist = spatialDNN_MaxPool(t_plus_1_dist)

    dist_cat = concatenate([t_min_1_dist,t_dist_Y,t_plus_1_dist])
    dist_cat_Conv = temporalDNN_Conv(dist_cat)
    dist_cat_Conv = temporalDNN_SpatialDropout(dist_cat_Conv)
    dist_cat_Conv = temporalDNN_MaxPool(dist_cat_Conv)

    ref_dist_cat = concatenate([ref_cat_Conv,dist_cat_Conv])
    ref_dist_cat = Conv2D(256,(3,3), kernel_regularizer='l2')(ref_dist_cat)
    ref_dist_cat = SpatialDropout2D(0.2)(ref_dist_cat)
    ref_dist_cat = LeakyReLU(alpha=0.1)(ref_dist_cat)

    ref_dist_cat = Flatten()(ref_dist_cat)
    ref_dist_cat = Dense(256, kernel_regularizer='l2')(ref_dist_cat)
    ref_dist_cat = Dropout(0.3)(ref_dist_cat)
    ref_dist_cat = LeakyReLU(alpha=0.1)(ref_dist_cat)
    ref_dist_cat = Dense(128, kernel_regularizer='l2')(ref_dist_cat)
    ref_dist_cat = Dropout(0.3)(ref_dist_cat)
    ref_dist_cat = LeakyReLU(alpha=0.1)(ref_dist_cat)
    ref_dist_cat = Dense(1, kernel_regularizer='l2')(ref_dist_cat)
    ref_dist_cat = LeakyReLU(alpha=0.1)(ref_dist_cat)


    ref_list_inputs = [t_min_1_ref_in,t_ref_in_Y,t_plus_1_ref_in]
    dist_list_inputs = [t_min_1_dist_in,t_dist_in_Y,t_plus_1_dist_in]
    loss_fn = keras.losses.MeanSquaredError()
    model = keras.models.Model(inputs = ref_list_inputs+dist_list_inputs, outputs= ref_dist_cat)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mse'])
    model.summary()
