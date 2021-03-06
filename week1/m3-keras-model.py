import os

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tensorflow as tf
import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Reshape, Input, MaxPooling2D, Conv2D, ReLU, \
     BatchNormalization, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Add, SpatialDropout2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint

# from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, VGG19

#MIT_SPLIT = 'C:\\Users\\Carmen\\CVMaster\\Databases\\MIT_split'
#GROUP_DIR = './'

GROUP_DIR = '.'
MIT_SPLIT='/home/capiguri/code/uab_cv_master/m3/Databases/MIT_split'

# GROUP_DIR = '/home/group06/code/w4/'
# MIT_SPLIT = '/home/mcv/datasets/MIT_split'

train_data_dir=f'{MIT_SPLIT}/train'
test_data_dir=f'{MIT_SPLIT}/test'
img_width = 64
img_height= 64
batch_size=32

print('TRAIN DIR', train_data_dir)
print('TEST DIR', test_data_dir)


def basic(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dense(512)(x)
    # x = Dense(256)(x)

    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model


def basic_lessdrop(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)
    x = Dropout(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dense(512)(x)
    # x = Dense(256)(x)

    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

def basic_lessdrop_residual(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)

    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = Dropout(0.4)(x)

    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = Dropout(0.4)(x)

    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = Dropout(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dense(512)(x)
    # x = Dense(256)(x)

    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

def basic_spdrop_residual(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)
    
    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x)

    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x)

    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x)

    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dense(512)(x)
    # x = Dense(256)(x)

    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model


def bigger_spdrop_residual(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)

    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    resin = x
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x)

    # x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dense(512)(x)
    # x = Dense(256)(x)

    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

def smaller_spdrop_residual(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x) # 0.3
    
    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x) # 0.1

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x) # 0.3

    resin = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x) # 0.1

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x) # 0.3

    x = GlobalAveragePooling2D()(x)

    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

def smaller_spdrop_residual_bottleneck(*args, **kwargs):
    def bottleneck_conv(num_ker, kernel_size, x, dimred=0.5):
        x = Conv2D(int(num_ker*dimred), (1, 1), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
        x = Conv2D(int(num_ker), kernel_size, strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
        x = Conv2D(int(num_ker), (1, 1), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
        return x

    input_layer = Input(shape=(img_height, img_width, 3))
    x = bottleneck_conv(64, (3, 3), input_layer)#, strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x) # 0.3
    
    resin = x
    #x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = bottleneck_conv(64, (3, 3), x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    # x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = bottleneck_conv(64, (3, 3), x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x) # 0.1

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x) # 0.3

    resin = x
    # x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = bottleneck_conv(64, (3, 3), x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x)

    # x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = bottleneck_conv(64, (3, 3), x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.2)(x) # 0.1

    x = Add()([resin, x])
    x = MaxPooling2D()(x)
    x = SpatialDropout2D(0.2)(x) # 0.3

    x = GlobalAveragePooling2D()(x)

    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model



def preresidual(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))

    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)

    x = Dropout(0.2)(x)
    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)

    x = Dropout(0.2)(x)
    x = MaxPooling2D()(x)

    # Block 3
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)


    x = Dropout(0.2)(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

def smaller_blocks_res(*args, **kwargs):
    input_layer = Input(shape=(img_height, img_width, 3))
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(input_layer)

    # Block 1
    inblock = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Add()([inblock, x])
    x = MaxPooling2D()(x)

    # Block 2
    inblock = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Add()([inblock, x])

    x = MaxPooling2D()(x)

    # Block 3
    inblock = x
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Add()([inblock, x])
    
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    softout = Dense(units=8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=softout)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

    return model


test_configurations=[
    # {
    # 'name' : 'ImgSize112',
    # 'subname': ['ResidualSmall'],
    # 'args' : [[]],
    # 'epochs' : 200,
    # 'model' : smaller_blocks_res,
    # 'load': False
    # },
    {
    'name' : 'ImgSize64',
    'subname': ['ResidualBottleneck1'],
    'args' : [[]],
    'epochs' : 200,
    'model' : smaller_spdrop_residual_bottleneck,
    'load': True
    },
]

if not os.path.exists('./output'):
    print('Creating output dir')
    os.mkdir('./output')
else:
    print('output dir already exists')

if not os.path.exists('./tensorboard_out'):
    print('Creating tensorboard output dir')
    os.mkdir('./tensorboard_out')
else:
    print('tensorboard output dir already exists')

for conf in test_configurations:
    if not os.path.exists(f'./output/{conf["name"]}'):
        os.mkdir(f'./output/{conf["name"]}')
    
    print('Running test', conf)

    for arg, subname in zip(conf['args'], conf['subname']):
        checkpoint_path = f'{GROUP_DIR}/output/{conf["name"]}/model_{conf["name"]}_{subname}.h5'
        save_path = f'{GROUP_DIR}/output/{conf["name"]}/final_model_{conf["name"]}_{subname}.h5'
        
        if os.path.exists(save_path) and conf['load']:
            print('Loading checkpoint...')
            print(save_path)
            model = tf.keras.models.load_model(save_path)
            subname += '_reloaded' 
            checkpoint_path = f'{GROUP_DIR}/output/{conf["name"]}/model_{conf["name"]}_{subname}.h5'
            save_path = f'{GROUP_DIR}/output/{conf["name"]}/final_model_{conf["name"]}_{subname}.h5'
        else:
            model = conf['model'](*arg)

        # Callbacks
        cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto')

        early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=40, min_delta=0.001)

        tbCallBack = TensorBoard(log_dir=f'{GROUP_DIR}/tensorboard_out/{conf["name"]}_{subname}',
            update_freq='batch', histogram_freq=1, write_graph=True, profile_batch=0)


        print(model.summary())

        datagen = ImageDataGenerator(featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            preprocessing_function=None,
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            validation_split=0.2)

        train_generator = datagen.flow_from_directory(train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                subset='training',
                class_mode='categorical')

        validation_generator = datagen.flow_from_directory(train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                subset='validation',
                class_mode='categorical')

        test_generator = datagen.flow_from_directory(test_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical')


        history=model.fit(train_generator,
                epochs=conf['epochs'],
                validation_data=validation_generator,
                callbacks=[tbCallBack, cp_callback])

        result = model.evaluate(test_generator)
        print(f'Test {conf["name"]}_{subname} result:', result)

        model.save(save_path)
        
        pkl_dir = f'{GROUP_DIR}/output/{conf["name"]}/history_{conf["name"]}_{subname}.pkl'

        with open(pkl_dir, 'wb') as fp:
            pkl.dump(history.history, fp)
