# Differentiable-PVQM

This repo consists of the code and trained model for an updated version of the Differentiable VMAF using Tensorflow.

The model was trained used the luma channel which is of the range 0-1. 

The input data to this model should be of the size BatchSizexNumberOfFramesxHeightxWidthxChannels.

To load the model, first clone this repo, and ensure you have all required packages installed.
The file test.py shows how to use this model. 