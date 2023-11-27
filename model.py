import tensorflow as tf
import numpy as np
import segmentation_models as sm

HEIGHT = 512

class ModelCreator:
    #unet
    #encoder x:
    def encoder_x(this,input,n_filters):
        x = tf.keras.layers.Conv2D(n_filters, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    #decoder_x:
    def decoder_x(this,input,skip_connections,n_filters,dropout = 0):
        x = tf.keras.layers.concatenate([input,skip_connections])
        x = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    #output
    def output(this,x):
        return tf.keras.layers.Conv2DTranspose(1,3,padding = "same", kernel_initializer = "he_normal",use_bias=False,activation='sigmoid')(x)


    def buildmodel(this):
        inputlayer = tf.keras.layers.Input(shape=(HEIGHT,HEIGHT,3))
        print(inputlayer)
        e1 = this.encoder_x(inputlayer,48)
        e2 = this.encoder_x(e1,48)
        e3 = this.encoder_x(e2,176)
        e4 = this.encoder_x(e3,416)
        x = this.encoder_x(e4,128)
        x = tf.keras.layers.Conv2DTranspose(128, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = this.decoder_x(x,e4,416)
        x = this.decoder_x(x,e3,176)
        x = this.decoder_x(x,e2,48)
        x = this.decoder_x(x,e1,48)
        outputlayer = this.output(x)
        print(outputlayer)
        
        unet_model = tf.keras.Model(inputlayer, outputlayer, name="U-Net")

        return unet_model



    def build_tunermodel(this,hp):
        inputlayer = tf.keras.layers.Input(shape=(HEIGHT,HEIGHT,3))
        print(inputlayer)
        layer1 = hp.Int('layer1', min_value=8, max_value=64, step=8)
        e1 = this.encoder_x(inputlayer,layer1)
        layer2 = hp.Int('layer2', min_value=32, max_value=96, step=8)
        e2 = this.encoder_x(e1,layer2)
        layer3 = hp.Int('layer3', min_value=64, max_value=256, step=16)
        e3 = this.encoder_x(e2,layer3)
        layer4 = hp.Int('layer4', min_value=64, max_value=512, step=32)
        e4 = this.encoder_x(e3,layer4)
        layer5 = hp.Int('layer5', min_value=64, max_value=512, step=64)
        x = this.encoder_x(e4,layer5)
        x = tf.keras.layers.Conv2DTranspose(layer5, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.ReLU()(x)
        dropout = hp.Float('dropout', min_value=0, max_value=0.7, step=0.1)
        x = this.decoder_x(x,e4,layer4,dropout)
        x = this.decoder_x(x,e3,layer3,dropout)
        x = this.decoder_x(x,e2,layer2,dropout)
        x = this.decoder_x(x,e1,layer1,dropout)
        outputlayer = this.output(x)
        print(outputlayer)
        
        unet_model = tf.keras.Model(inputlayer, outputlayer, name="U-Net")

        unet_model.compile('Adam', loss=sm.losses.binary_focal_dice_loss, metrics=[sm.metrics.iou_score])

        return unet_model