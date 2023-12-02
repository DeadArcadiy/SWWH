import tensorflow as tf
import numpy as np
import segmentation_models as sm

HEIGHT = 512

class ModelCreator:
    #unet
    #encoder x:
    def encoder_x(self,input,n_filters):
        x = tf.keras.layers.Conv2D(n_filters, 3,padding = "same", kernel_initializer = "he_normal",use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        return x

    #decoder_x:
    def decoder_x(self,input,skip_connections,n_filters):
        x = tf.keras.layers.concatenate([input,skip_connections])
        x = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides= 2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    #output
    def output(self,x):
        return tf.keras.layers.Conv2DTranspose(1,3,padding = "same", kernel_initializer = "he_normal",use_bias=False,activation='sigmoid')(x)


    def buildmodel(self):
        inputlayer = tf.keras.layers.Input(shape=(HEIGHT,HEIGHT,3))
        print(inputlayer)
        e1 = self.encoder_x(inputlayer,32)
        e2 = self.encoder_x(e1,64)
        e3 = self.encoder_x(e2,128)
        e4 = self.encoder_x(e3,256)
        e5 = self.encoder_x(e4,512)
        x = tf.keras.layers.Conv2DTranspose(512, 3,strides=2,padding = "same", kernel_initializer = "he_normal",use_bias=False)(e5)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.decoder_x(x,e4,256)
        x = self.decoder_x(x,e3,128)
        x = self.decoder_x(x,e2,64)
        x = self.decoder_x(x,e1,32)
        outputlayer = self.output(x)
        print(outputlayer)
        
        unet_model = tf.keras.Model(inputlayer, outputlayer, name="U-Net")

        unet_model.compile('Adam', loss=sm.losses.binary_focal_dice_loss, metrics=[sm.metrics.iou_score])

        return unet_model



    def build_tunermodel(self,hp):
        inputlayer = tf.keras.layers.Input(shape=(HEIGHT,HEIGHT,3))
        print(inputlayer)
        layer1 = hp.Int('layer1', min_value=16, max_value=64, step=16)
        e1 = self.encoder_x(inputlayer,layer1)
        layer2 = hp.Int('layer2', min_value=32, max_value=96, step=32)
        e2 = self.encoder_x(e1,layer2)
        layer3 = hp.Int('layer3', min_value=64, max_value=256, step=64)
        e3 = self.encoder_x(e2,layer3)
        layer4 = hp.Int('layer4', min_value=128, max_value=512, step=128)
        e4 = self.encoder_x(e3,layer4)
        layer5 = hp.Int('layer5', min_value=256, max_value=1024, step=256)
        x = tf.keras.layers.Conv2D(layer5, 3,padding = "same", kernel_initializer = "he_normal",use_bias=False)(e4)
        x = tf.keras.layers.ReLU()(x)
        x = self.decoder_x(x,e4,layer4)
        x = self.decoder_x(x,e3,layer3)
        x = self.decoder_x(x,e2,layer2)
        x = self.decoder_x(x,e1,layer1)
        outputlayer = self.output(x)
        print(outputlayer)
        
        unet_model = tf.keras.Model(inputlayer, outputlayer, name="U-Net")

        unet_model.compile('Adam', loss=sm.losses.binary_focal_dice_loss, metrics=[sm.metrics.iou_score])

        return unet_model