import tensorflow as tf
import keras 
import math 
import os 
import json  

def saveModel(folderName, model, arguments):
    os.makedirs(folderName, exist_ok=True)
    model.save_weights(os.path.join(folderName, "model.h5"))
    with open(os.path.join(folderName, "arguments.json"), "w") as f:
        json.dump(arguments, f)

def loadModel(modelClass, modelPath, returnArgs=False, additionalArgs={}, blankModel=False):
    with open(os.path.join(modelPath, "arguments.json")) as _f:
        modelArgs = json.load(_f)
    model = modelClass(**modelArgs).model(**additionalArgs)
    if blankModel == 'True':
        pass 
    else:
        model.load_weights(os.path.join(modelPath, "model.h5"))
    if returnArgs:
        return model, modelArgs
    else:
        return model 

class cnnOpPadded(keras.layers.Layer):
    def __init__(self, filter=32,ksize=5, padding='VALID', activation=tf.keras.layers.LeakyReLU(alpha=0.3), paddingAmt=2, strides=1, normalization=False, kernel_regularizer=None, bias_regularizer=None):
        super(cnnOpPadded, self).__init__()
        self.filter= filter 
        self.ksize = ksize 
        self.padding = padding 
        self.activation = activation
        self.paddingAmt = paddingAmt
        self.strides= strides 
        self.normalization = normalization
        self.conv2d = keras.layers.Conv2D(self.filter, self.ksize, padding=self.padding, activation=self.activation, strides=self.strides, use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform(seed=0), kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
        if self.normalization:
            self.normLayer = keras.layers.LayerNormalization()
    
    def call(self, x, training=False):
        x = tf.pad(x, [[0, 0], [self.paddingAmt, self.paddingAmt],[self.paddingAmt, self.paddingAmt], [0, 0]], mode='REFLECT')
        x = self.conv2d(x, training=training)
        if self.normalization:
            x = self.normLayer(x, training=training)
        return x

    def get_config(self):
        config = super(cnnOpPadded, self).get_config()
        config.update({'filter':self.filter,'ksize':self.ksize, 'padding':self.padding, 'activation':self.activation, 'paddingAmt':self.paddingAmt, 'strides':self.strides, 'normalization':self.normalization})
        return config    


class cnnBlock(keras.layers.Layer):
    def __init__(self, mode='DS', filter=12, ksize=5, numCNNs=2, normalization=False, resConn=True): 
        super().__init__()
        self.cnnLayers = []
        _paddingAmt = math.floor(ksize/2)
        for _ in range(numCNNs):
            self.cnnLayers.append(cnnOpPadded(filter, ksize, paddingAmt=_paddingAmt))
        self.resConn = resConn
        if self.resConn:
            self.resConn = len(self.cnnLayers)-1
        self.parallelConv = cnnOpPadded(filter, ksize=1, paddingAmt=0)
        if mode == 'DS':
            self.cnnLayers.append(cnnOpPadded(filter, ksize, paddingAmt=_paddingAmt, strides=2, normalization=normalization))
        if mode == 'US-bilinear':
            self.cnnLayers.append(keras.layers.UpSampling2D(interpolation='bilinear'))
            self.cnnLayers.append(cnnOpPadded(filter, 3, paddingAmt=1))
        if mode == 'US-nearest':
            self.cnnLayers.append(keras.layers.UpSampling2D())
            self.cnnLayers.append(cnnOpPadded(filter, 3, paddingAmt=1))
        if mode == 'US':
            self.cnnLayers.append(keras.layers.Conv2DTranspose(filter, 5, strides=2, padding='same', activation=keras.layers.LeakyReLU(alpha=0.3)))
        if mode == 'Constant':
            self.cnnLayers.append(cnnOpPadded(filter, ksize, paddingAmt=_paddingAmt, normalization=normalization))
            
    def call(self, x, training=False):
        if self.resConn:
            x_res = self.parallelConv(x, training=training)
        for idx, _layer in enumerate(self.cnnLayers):
            x = _layer(x, training=training)
            if ((idx == self.resConn) and (self.resConn)):
                x += x_res
        return x 

class vmafProxy(keras.Model):
    '''
    Class for proxy vmaf model. Inputs are BSxnumFramesxHxWx3
    '''
    def __init__(self, numFrames=3, patchSize=128, effNet=False, combineMethod='CONCAT', singleFrameFilters=[12, 24, 48], multiFrameFilters=[82, 96, 128], combinedBranchFilters=[212, 256, 268], finalDense=[256, 128, 32, 1]):
        super().__init__()
        self.numFrames = numFrames
        self.patchSize = patchSize
        self.combineMethod = combineMethod
        if effNet:
            concatLayer = keras.layers.Lambda(lambda x: 255.0*tf.concat([x, x, x], -1))
            effNetModel = keras.applications.EfficientNetB3(False, "imagenet", input_tensor=keras.Input((self.patchSize, self.patchSize, 3)))
            for _layer in effNetModel.layers:
                _layer.trainable = False
            self.singleFrameCnn = keras.Sequential([concatLayer, effNetModel], name='per_frame')
            self.singleFrameTraining = False
        else:
            singleFrameCnn = []
            for _sfFilter in singleFrameFilters:
                singleFrameCnn.append(cnnBlock('DS', _sfFilter, 5, 1, True))
            self.singleFrameCnn = keras.Sequential(singleFrameCnn, name='per_frame')
            self.singleFrameTraining = True

        if effNet:
            blockType = 'Constant'
        else:
            blockType = 'DS'

        multiFrameCnn = []
        for _mfFilter in multiFrameFilters:
            multiFrameCnn.append(cnnBlock(blockType, _mfFilter, 3, 1, True))
        self.multiFrameCnn = keras.Sequential(multiFrameCnn, name='multi_frame')

        combinedBranchCnn = []
        for _cbFilter in combinedBranchFilters:
            combinedBranchCnn.append(cnnBlock('DS', _cbFilter, 3, 1, True))
        combinedBranchCnn.append(keras.layers.Lambda(lambda x: tf.reduce_mean(x, [1,2])))
        self.combinedBranchCnn = keras.Sequential(combinedBranchCnn, name='joint_multi_frame')


        finalDenseMLP = []
        for _fd in finalDense[:-1]:
            finalDenseMLP.append(keras.layers.Dense(_fd, activation='LeakyReLU'))
        finalDenseMLP.append(keras.layers.Dense(finalDense[-1], activation='tanh'))
        self.finalDenseMLP = keras.Sequential(finalDenseMLP)

    def call(self, x, training=False):
        x_ref = x[0]
        x_deg = x[1]
        refFeature = [] 
        degFeature = []
        for _frame in range(self.numFrames):
            _x_ref = x_ref[:,_frame]
            _x_deg = x_deg[:,_frame]
            if self.singleFrameTraining:
                trainingFlag = training
            else:
                trainingFlag = False
            _x_ref, _x_deg = self.singleFrameCnn(_x_ref, training=trainingFlag), self.singleFrameCnn(_x_deg, training=trainingFlag)
            refFeature.append(_x_ref)
            degFeature.append(_x_deg)
        refFeature, degFeature = tf.concat(refFeature, -1), tf.concat(degFeature, -1)
        refFeature, degFeature = self.multiFrameCnn(refFeature, training=training), self.multiFrameCnn(degFeature, training=training)
        if self.combineMethod == 'CONCAT':
            combinedFeatures = tf.concat([refFeature, degFeature], -1)
        elif self.combineMethod == 'DIFFERENCE':
            combinedFeatures = refFeature - degFeature
        elif self.combineMethod == 'ABSOLUTE':
            combinedFeatures = tf.abs(refFeature - degFeature)
        combinedFeatures = self.combinedBranchCnn(combinedFeatures, training=training)
        combinedFeatures = self.finalDenseMLP(combinedFeatures, training=training)
        combinedFeatures = (combinedFeatures+1.0)*50.0
        return combinedFeatures

    def model(self):
        inputs = [keras.layers.Input((self.numFrames, 128, 128, 1)), keras.layers.Input((self.numFrames, 128, 128, 1))]
        return keras.Model(inputs, self.call(inputs))