# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:07:30 2022

@author: Danie
Contains ResNet Model Structure for stage 3 classification, as well as training script.
To run training found in __main__, please gaurentee directory structure found at base_dir is consistent
with output from previous steps. 
ResNet equivelent code, built from medium article found here: https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba

Model Structure:
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 1, 193, 126  0           []                               
                                    )]                                                                
                                                                                                      
     batch_normalization_34 (BatchN  (None, 1, 193, 126)  504        ['input_1[0][0]']                
     ormalization)                                                                                    
                                                                                                      
     conv2d_29 (Conv2D)             (None, 64, 193, 126  640         ['batch_normalization_34[0][0]'] 
                                    )                                                                 
                                                                                                      
     re_lu (ReLU)                   (None, 64, 193, 126  0           ['conv2d_29[0][0]']              
                                    )                                                                 
                                                                                                      
     batch_normalization_35 (BatchN  (None, 64, 193, 126  504        ['re_lu[0][0]']                  
     ormalization)                  )                                                                 
                                                                                                      
     conv2d_30 (Conv2D)             (None, 64, 193, 126  36928       ['batch_normalization_35[0][0]'] 
                                    )                                                                 
                                                                                                      
     re_lu_1 (ReLU)                 (None, 64, 193, 126  0           ['conv2d_30[0][0]']              
                                    )                                                                 
                                                                                                      
     batch_normalization_36 (BatchN  (None, 64, 193, 126  504        ['re_lu_1[0][0]']                
     ormalization)                  )                                                                 
                                                                                                      
     conv2d_31 (Conv2D)             (None, 64, 193, 126  36928       ['batch_normalization_36[0][0]'] 
                                    )                                                                 
                                                                                                      
     add (Add)                      (None, 64, 193, 126  0           ['batch_normalization_35[0][0]', 
                                    )                                 'conv2d_31[0][0]']              
                                                                                                      
     re_lu_2 (ReLU)                 (None, 64, 193, 126  0           ['add[0][0]']                    
                                    )                                                                 
                                                                                                      
     batch_normalization_37 (BatchN  (None, 64, 193, 126  504        ['re_lu_2[0][0]']                
     ormalization)                  )                                                                 
                                                                                                      
     conv2d_32 (Conv2D)             (None, 64, 193, 126  36928       ['batch_normalization_37[0][0]'] 
                                    )                                                                 
                                                                                                      
     re_lu_3 (ReLU)                 (None, 64, 193, 126  0           ['conv2d_32[0][0]']              
                                    )                                                                 
                                                                                                      
     batch_normalization_38 (BatchN  (None, 64, 193, 126  504        ['re_lu_3[0][0]']                
     ormalization)                  )                                                                 
                                                                                                      
     conv2d_33 (Conv2D)             (None, 64, 193, 126  36928       ['batch_normalization_38[0][0]'] 
                                    )                                                                 
                                                                                                      
     add_1 (Add)                    (None, 64, 193, 126  0           ['batch_normalization_37[0][0]', 
                                    )                                 'conv2d_33[0][0]']              
                                                                                                      
     re_lu_4 (ReLU)                 (None, 64, 193, 126  0           ['add_1[0][0]']                  
                                    )                                                                 
                                                                                                      
     batch_normalization_39 (BatchN  (None, 64, 193, 126  504        ['re_lu_4[0][0]']                
     ormalization)                  )                                                                 
                                                                                                      
     conv2d_34 (Conv2D)             (None, 128, 97, 63)  73856       ['batch_normalization_39[0][0]'] 
                                                                                                      
     re_lu_5 (ReLU)                 (None, 128, 97, 63)  0           ['conv2d_34[0][0]']              
                                                                                                      
     batch_normalization_40 (BatchN  (None, 128, 97, 63)  252        ['re_lu_5[0][0]']                
     ormalization)                                                                                    
                                                                                                      
     conv2d_36 (Conv2D)             (None, 128, 97, 63)  8320        ['batch_normalization_39[0][0]'] 
                                                                                                      
     conv2d_35 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_40[0][0]'] 
                                                                                                      
     add_2 (Add)                    (None, 128, 97, 63)  0           ['conv2d_36[0][0]',              
                                                                      'conv2d_35[0][0]']              
                                                                                                      
     re_lu_6 (ReLU)                 (None, 128, 97, 63)  0           ['add_2[0][0]']                  
                                                                                                      
     batch_normalization_41 (BatchN  (None, 128, 97, 63)  252        ['re_lu_6[0][0]']                
     ormalization)                                                                                    
                                                                                                      
     conv2d_37 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_41[0][0]'] 
                                                                                                      
     re_lu_7 (ReLU)                 (None, 128, 97, 63)  0           ['conv2d_37[0][0]']              
                                                                                                      
     batch_normalization_42 (BatchN  (None, 128, 97, 63)  252        ['re_lu_7[0][0]']                
     ormalization)                                                                                    
                                                                                                      
     conv2d_38 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_42[0][0]'] 
                                                                                                      
     add_3 (Add)                    (None, 128, 97, 63)  0           ['batch_normalization_41[0][0]', 
                                                                      'conv2d_38[0][0]']              
                                                                                                      
     re_lu_8 (ReLU)                 (None, 128, 97, 63)  0           ['add_3[0][0]']                  
                                                                                                      
     batch_normalization_43 (BatchN  (None, 128, 97, 63)  252        ['re_lu_8[0][0]']                
     ormalization)                                                                                    
                                                                                                      
     conv2d_39 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_43[0][0]'] 
                                                                                                      
     re_lu_9 (ReLU)                 (None, 128, 97, 63)  0           ['conv2d_39[0][0]']              
                                                                                                      
     batch_normalization_44 (BatchN  (None, 128, 97, 63)  252        ['re_lu_9[0][0]']                
     ormalization)                                                                                    
                                                                                                      
     conv2d_40 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_44[0][0]'] 
                                                                                                      
     add_4 (Add)                    (None, 128, 97, 63)  0           ['batch_normalization_43[0][0]', 
                                                                      'conv2d_40[0][0]']              
                                                                                                      
     re_lu_10 (ReLU)                (None, 128, 97, 63)  0           ['add_4[0][0]']                  
                                                                                                      
     batch_normalization_45 (BatchN  (None, 128, 97, 63)  252        ['re_lu_10[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_41 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_45[0][0]'] 
                                                                                                      
     re_lu_11 (ReLU)                (None, 128, 97, 63)  0           ['conv2d_41[0][0]']              
                                                                                                      
     batch_normalization_46 (BatchN  (None, 128, 97, 63)  252        ['re_lu_11[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_42 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_46[0][0]'] 
                                                                                                      
     add_5 (Add)                    (None, 128, 97, 63)  0           ['batch_normalization_45[0][0]', 
                                                                      'conv2d_42[0][0]']              
                                                                                                      
     re_lu_12 (ReLU)                (None, 128, 97, 63)  0           ['add_5[0][0]']                  
                                                                                                      
     batch_normalization_47 (BatchN  (None, 128, 97, 63)  252        ['re_lu_12[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_43 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_47[0][0]'] 
                                                                                                      
     re_lu_13 (ReLU)                (None, 128, 97, 63)  0           ['conv2d_43[0][0]']              
                                                                                                      
     batch_normalization_48 (BatchN  (None, 128, 97, 63)  252        ['re_lu_13[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_44 (Conv2D)             (None, 128, 97, 63)  147584      ['batch_normalization_48[0][0]'] 
                                                                                                      
     add_6 (Add)                    (None, 128, 97, 63)  0           ['batch_normalization_47[0][0]', 
                                                                      'conv2d_44[0][0]']              
                                                                                                      
     re_lu_14 (ReLU)                (None, 128, 97, 63)  0           ['add_6[0][0]']                  
                                                                                                      
     batch_normalization_49 (BatchN  (None, 128, 97, 63)  252        ['re_lu_14[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_45 (Conv2D)             (None, 256, 49, 32)  295168      ['batch_normalization_49[0][0]'] 
                                                                                                      
     re_lu_15 (ReLU)                (None, 256, 49, 32)  0           ['conv2d_45[0][0]']              
                                                                                                      
     batch_normalization_50 (BatchN  (None, 256, 49, 32)  128        ['re_lu_15[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_47 (Conv2D)             (None, 256, 49, 32)  33024       ['batch_normalization_49[0][0]'] 
                                                                                                      
     conv2d_46 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_50[0][0]'] 
                                                                                                      
     add_7 (Add)                    (None, 256, 49, 32)  0           ['conv2d_47[0][0]',              
                                                                      'conv2d_46[0][0]']              
                                                                                                      
     re_lu_16 (ReLU)                (None, 256, 49, 32)  0           ['add_7[0][0]']                  
                                                                                                      
     batch_normalization_51 (BatchN  (None, 256, 49, 32)  128        ['re_lu_16[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_48 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_51[0][0]'] 
                                                                                                      
     re_lu_17 (ReLU)                (None, 256, 49, 32)  0           ['conv2d_48[0][0]']              
                                                                                                      
     batch_normalization_52 (BatchN  (None, 256, 49, 32)  128        ['re_lu_17[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_49 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_52[0][0]'] 
                                                                                                      
     add_8 (Add)                    (None, 256, 49, 32)  0           ['batch_normalization_51[0][0]', 
                                                                      'conv2d_49[0][0]']              
                                                                                                      
     re_lu_18 (ReLU)                (None, 256, 49, 32)  0           ['add_8[0][0]']                  
                                                                                                      
     batch_normalization_53 (BatchN  (None, 256, 49, 32)  128        ['re_lu_18[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_50 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_53[0][0]'] 
                                                                                                      
     re_lu_19 (ReLU)                (None, 256, 49, 32)  0           ['conv2d_50[0][0]']              
                                                                                                      
     batch_normalization_54 (BatchN  (None, 256, 49, 32)  128        ['re_lu_19[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_51 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_54[0][0]'] 
                                                                                                      
     add_9 (Add)                    (None, 256, 49, 32)  0           ['batch_normalization_53[0][0]', 
                                                                      'conv2d_51[0][0]']              
                                                                                                      
     re_lu_20 (ReLU)                (None, 256, 49, 32)  0           ['add_9[0][0]']                  
                                                                                                      
     batch_normalization_55 (BatchN  (None, 256, 49, 32)  128        ['re_lu_20[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_52 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_55[0][0]'] 
                                                                                                      
     re_lu_21 (ReLU)                (None, 256, 49, 32)  0           ['conv2d_52[0][0]']              
                                                                                                      
     batch_normalization_56 (BatchN  (None, 256, 49, 32)  128        ['re_lu_21[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_53 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_56[0][0]'] 
                                                                                                      
     add_10 (Add)                   (None, 256, 49, 32)  0           ['batch_normalization_55[0][0]', 
                                                                      'conv2d_53[0][0]']              
                                                                                                      
     re_lu_22 (ReLU)                (None, 256, 49, 32)  0           ['add_10[0][0]']                 
                                                                                                      
     batch_normalization_57 (BatchN  (None, 256, 49, 32)  128        ['re_lu_22[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_54 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_57[0][0]'] 
                                                                                                      
     re_lu_23 (ReLU)                (None, 256, 49, 32)  0           ['conv2d_54[0][0]']              
                                                                                                      
     batch_normalization_58 (BatchN  (None, 256, 49, 32)  128        ['re_lu_23[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_55 (Conv2D)             (None, 256, 49, 32)  590080      ['batch_normalization_58[0][0]'] 
                                                                                                      
     add_11 (Add)                   (None, 256, 49, 32)  0           ['batch_normalization_57[0][0]', 
                                                                      'conv2d_55[0][0]']              
                                                                                                      
     re_lu_24 (ReLU)                (None, 256, 49, 32)  0           ['add_11[0][0]']                 
                                                                                                      
     batch_normalization_59 (BatchN  (None, 256, 49, 32)  128        ['re_lu_24[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_56 (Conv2D)             (None, 512, 25, 16)  1180160     ['batch_normalization_59[0][0]'] 
                                                                                                      
     re_lu_25 (ReLU)                (None, 512, 25, 16)  0           ['conv2d_56[0][0]']              
                                                                                                      
     batch_normalization_60 (BatchN  (None, 512, 25, 16)  64         ['re_lu_25[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_58 (Conv2D)             (None, 512, 25, 16)  131584      ['batch_normalization_59[0][0]'] 
                                                                                                      
     conv2d_57 (Conv2D)             (None, 512, 25, 16)  2359808     ['batch_normalization_60[0][0]'] 
                                                                                                      
     add_12 (Add)                   (None, 512, 25, 16)  0           ['conv2d_58[0][0]',              
                                                                      'conv2d_57[0][0]']              
                                                                                                      
     re_lu_26 (ReLU)                (None, 512, 25, 16)  0           ['add_12[0][0]']                 
                                                                                                      
     batch_normalization_61 (BatchN  (None, 512, 25, 16)  64         ['re_lu_26[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_59 (Conv2D)             (None, 512, 25, 16)  2359808     ['batch_normalization_61[0][0]'] 
                                                                                                      
     re_lu_27 (ReLU)                (None, 512, 25, 16)  0           ['conv2d_59[0][0]']              
                                                                                                      
     batch_normalization_62 (BatchN  (None, 512, 25, 16)  64         ['re_lu_27[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     conv2d_60 (Conv2D)             (None, 512, 25, 16)  2359808     ['batch_normalization_62[0][0]'] 
                                                                                                      
     add_13 (Add)                   (None, 512, 25, 16)  0           ['batch_normalization_61[0][0]', 
                                                                      'conv2d_60[0][0]']              
                                                                                                      
     re_lu_28 (ReLU)                (None, 512, 25, 16)  0           ['add_13[0][0]']                 
                                                                                                      
     batch_normalization_63 (BatchN  (None, 512, 25, 16)  64         ['re_lu_28[0][0]']               
     ormalization)                                                                                    
                                                                                                      
     average_pooling2d (AveragePool  (None, 512, 6, 4)   0           ['batch_normalization_63[0][0]'] 
     ing2D)                                                                                           
                                                                                                      
     flatten_3 (Flatten)            (None, 12288)        0           ['average_pooling2d[0][0]']      
                                                                                                      
     dense_3 (Dense)                (None, 10)           122890      ['flatten_3[0][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 15,718,834
    Trainable params: 15,715,294
    Non-trainable params: 3,540
    __________________________________________________________________________________________________
"""

import os
from tensorflow import Tensor
from test_iterator import TestIterator
from tensorflow.keras.layers import Conv2D, Input, AveragePooling2D, ReLU, BatchNormalization, Dense, Flatten, Add
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

def relu_bn(inputs: Tensor) -> Tensor:
    """
    Relu and Batch Normalization Block for ResNet. 

    Parameters
    ----------
    inputs : Tensor
        Layer state.

    Returns
    -------
    Tensor
        Layer after batch normalization and ReLu.

    """
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    """
    

    Parameters
    ----------
    x : Tensor
        DESCRIPTION.
    downsample : bool
        DESCRIPTION.
    filters : int
        DESCRIPTION.
    kernel_size : int, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net() -> Model:
    """
    Creates the Resnet Model. To see Architecture
    Here is a good page for understanding the layer architecture:
    https://iq.opengenus.org/resnet50-architecture/#:~:text=ResNet50%20is%20a%20variant%20of,explored%20ResNet50%20architecture%20in%20depth.

    Returns
    -------
    Model
        keras Non-Sequential Model.

    """
    
    inputs = Input(shape=(1, 193, 126))
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__=="__main__":
    
    base_dir = "E:\\Coding\\Dataset"
    train_image_dir = os.path.join(base_dir, "Test_Map_Heat")
    val_image_dir = os.path.join(base_dir, "Train_Map_Heat")
    batch_size = 32
    train_csvf = "test_labels.csv"
    val_csvf = "train_map.csv"
    
    epochsN = 32
    
    train_csv_dir = os.path.join(base_dir, train_csvf)
    val_csv_dir = os.path.join(base_dir, val_csvf)
    
    train_batches = TestIterator(batch_size, train_csv_dir, train_image_dir)
    val_batches = TestIterator(batch_size, val_csv_dir, val_image_dir)
    
    file_name = 'network_T_1'
    
    saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "log"), histogram_freq=1)
    
    cllbcks= [saveWeigts, tensorboard_callback]
    
    model_path = os.path.join(os.getcwd(),file_name+'_best_weights.h5')
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(file_name+'_best_weights.h5')
    else:
        model = create_res_net()
    
    model.summary()
    
    model.fit(x=train_batches, epochs=epochsN, steps_per_epoch=len(train_batches), callbacks = cllbcks,
              validation_data=val_batches, verbose=1)
    
    model.save(model_path, include_optimizer=False)
