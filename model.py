import tensorflow as tf
import numpy as np

class ModelCreator:
    #unet
    #encoder x:
    def encoder_x(this,x,n_filters):
        x = tf.keras.layers.Conv2D(n_filters, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    #decoder_x:
    def decoder_x(this,x,skip_connections,n_filters):
        x = tf.keras.layers.concatenate([x,skip_connections])
        x = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    #output
    def output(this,x):
        return tf.keras.layers.Conv2DTranspose(1,3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False,activation='sigmoid')(x)


    def buildmodel(this):
        inputlayer = tf.keras.layers.Input(shape=(256,256,3))
        print(inputlayer)
        e1 = this.encoder_x(inputlayer,64)
        e2 = this.encoder_x(e1,128)
        e3 = this.encoder_x(e2,256)
        e4 = this.encoder_x(e3,512)
        e5 = this.encoder_x(e4,512)
        e6 = this.encoder_x(e5,512)
        e7 = this.encoder_x(e6,512)
        x = this.decoder_x(e7,e7,512)
        x = this.decoder_x(x,e6,512)
        x = this.decoder_x(x,e5,512)
        x = this.decoder_x(x,e4,256)
        x = this.decoder_x(x,e3,128)
        x = this.decoder_x(x,e2,64)
        x = tf.keras.layers.concatenate([x,e1])
        outputlayer = this.output(x)
        print(outputlayer)
        
        unet_model = tf.keras.Model(inputlayer, outputlayer, name="U-Net")

        return unet_model
    
    def getpreprocess(this):
        def normalize(input_image):
            input_image = tf.cast(input_image, np.float32) / 255.0
            return input_image
        return normalize
