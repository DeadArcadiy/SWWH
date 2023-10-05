import tensorflow as tf
import numpy as np

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
    def decoder_x(this,input,skip_connections,n_filters):
        x = tf.keras.layers.concatenate([input,skip_connections])
        x = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    #output
    def output(this,x):
        return tf.keras.layers.Conv2DTranspose(1,3,padding = "same", kernel_initializer = "he_normal",use_bias=False,activation='sigmoid')(x)


    def buildmodel(this):
        inputlayer = tf.keras.layers.Input(shape=(HEIGHT,HEIGHT,3))
        print(inputlayer)
        e1 = this.encoder_x(inputlayer,32)
        e2 = this.encoder_x(e1,64)
        e3 = this.encoder_x(e2,128)
        e4 = this.encoder_x(e3,256)
        x = this.encoder_x(e4,512)
        x = tf.keras.layers.Conv2DTranspose(512, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.ReLU()(x)
        x = this.decoder_x(x,e4,256)
        x = this.decoder_x(x,e3,128)
        x = this.decoder_x(x,e2,64)
        x = this.decoder_x(x,e1,32)
        outputlayer = this.output(x)
        print(outputlayer)
        
        unet_model = tf.keras.Model(inputlayer, outputlayer, name="U-Net")

        return unet_model

