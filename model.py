from __future__ import print_function


import os
import time
import random

from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from skimage import color,filters
import numpy as np

from utils import *

def grad_loss(input_r_low, input_r_high):
    input_r_low_gray = tf.image.rgb_to_grayscale(input_r_low)
    input_r_high_gray = tf.image.rgb_to_grayscale(input_r_high)
    x_loss = tf.square(gradient_(input_r_low_gray, 'x') - gradient_(input_r_high_gray, 'x'))
    y_loss = tf.square(gradient_(input_r_low_gray, 'y') - gradient_(input_r_high_gray, 'y'))
    grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    return grad_loss_all

def ssim_loss(output_r, input_high_r):
    output_r_1 = output_r[:,:,:,0:1]
    input_high_r_1 = input_high_r[:,:,:,0:1]
    ssim_r_1 = tf_ssim(output_r_1, input_high_r_1)
    output_r_2 = output_r[:,:,:,1:2]
    input_high_r_2 = input_high_r[:,:,:,1:2]
    ssim_r_2 = tf_ssim(output_r_2, input_high_r_2)
    output_r_3 = output_r[:,:,:,2:3]
    input_high_r_3 = input_high_r[:,:,:,2:3]
    ssim_r_3 = tf_ssim(output_r_3, input_high_r_3)
    ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
    loss_ssim1 = 1-ssim_r
    return loss_ssim1

def cross_entropy_loss(Label, Seg_map, patch_size=48, batch_size=10):
    labels = tf.reshape(Label, [batch_size*patch_size*patch_size, 3])
    logits = tf.reshape(Seg_map, [batch_size*patch_size*patch_size, 3])
    CE_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return CE_loss

def concat(layers):
    return tf.concat(layers, axis=3)

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)
    
def my_lrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)
    
def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output

def LayerSegNet(image, channel=64, kernel_size=3):
    with tf.variable_scope("SegNet", reuse=tf.AUTO_REUSE):
        conv1_1 = tf.layers.conv2d(image, channel, kernel_size, strides=1, padding='same', activation=my_lrelu, name='conv1_1')
        conv1_2 = tf.layers.conv2d(conv1_1, channel, kernel_size, strides=1, padding='same', activation=my_lrelu, name='conv1_2')
        pool1 = tf.nn.avg_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        conv2_1 = tf.layers.conv2d(pool1, 2*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        conv2_2 = tf.layers.conv2d(conv2_1, 2*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        pool2 = tf.nn.avg_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        conv3_1 = tf.layers.conv2d(pool2, 4*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        conv3_2 = tf.layers.conv2d(conv3_1, 4*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        conv3_3 = tf.layers.conv2d(conv3_2, 4*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        conv3_4 = tf.layers.conv2d(conv3_3, 4*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        pool3 = tf.nn.avg_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        
        conv4_1 = tf.layers.conv2d(pool3, 4*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        conv4_2 = tf.layers.conv2d(conv4_1, 4*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
        conv4_3 = tf.layers.conv2d(conv4_2, 4*channel, kernel_size, strides=1, padding='same', activation=my_lrelu)

        # now to upscale to actual image size
        with tf.variable_scope("Upscale"):
            fuse_1 = tf.add(conv4_3, pool3)
            up1 = tf.image.resize_nearest_neighbor(fuse_1, (tf.shape(pool2)[1], tf.shape(pool2)[2]))
            deconv1 = tf.layers.conv2d(up1, 2*channel, kernel_size, strides=1, padding='same')
    
            fuse_2 = tf.add(deconv1, pool2)
            up2 = tf.image.resize_nearest_neighbor(fuse_2, (tf.shape(pool1)[1], tf.shape(pool1)[2]))
            deconv2 = tf.layers.conv2d(up2, channel, kernel_size, strides=1, padding='same')
            
            fuse_3 = tf.add(deconv2, pool1)
            up3 = tf.image.resize_nearest_neighbor(fuse_3, (tf.shape(image)[1], tf.shape(image)[2]))
            deconv3 = tf.layers.conv2d(up3, channel, kernel_size, strides=1, padding='same')
            
            deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
            deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
            feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])                           
            conv_t1 = tf.layers.conv2d(feature_gather, channel, kernel_size, strides=1, padding='same', activation=my_lrelu)
            conv_t2 = tf.layers.conv2d(conv_t1, channel, kernel_size, strides=1, padding='same')
            conv_t3 = tf.layers.conv2d(conv_t2, 3, 1, strides=1, padding='same')   
            
            annotation_pred = tf.nn.softmax(conv_t3)
            #annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return conv_t3, annotation_pred

def DecomNet(input):
    # KinD-DecomNet_Simple
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        up8 =  upsample_and_concat( conv3, conv2, 64, 128 , 'g_up_1')
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'g_up_2')
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv10=slim.conv2d(conv9,3,[1,1], rate=1, activation_fn=None, scope='g_conv10')
        R_out = tf.sigmoid(conv10)

        l_conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='l_conv1_2')
        l_conv3=tf.concat([l_conv2, conv9],3)
        l_conv4=slim.conv2d(l_conv3,1,[1,1], rate=1, activation_fn=None,scope='l_conv1_4')
        L_out = tf.sigmoid(l_conv4)

    return R_out, L_out

def RelightNet(input_i, input_R, input_ratio):
    with tf.variable_scope('RelightNet', reuse=tf.AUTO_REUSE):
        input_all = tf.concat([input_i, input_R], 3)
        input_L = tf.concat([input_i, input_ratio], 3)
        
        conv1=slim.conv2d(input_all,32,[3,3], rate=1, activation_fn=lrelu,scope='o_conv1_1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='o_conv2_1')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='o_conv3_1')
        up8 =  upsample_and_concat( conv3, conv2, 64, 128 , 'g_up_1')
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='o_conv4_1')
        up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'g_up_2')
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='o_conv5_1')
        F_all=slim.conv2d(conv9,32,[1,1], rate=1, activation_fn=None, scope='o_conv6')

        l_conv1=slim.conv2d(input_L, 32, [3,3], rate=1, activation_fn=lrelu, scope='l_conv1')
        l_conv2=slim.conv2d(l_conv1, 32, [3,3], rate=1, activation_fn=lrelu, scope='l_conv2')
        l_conv3=tf.concat([l_conv2, F_all],3)
        l_conv4=slim.conv2d(l_conv3, 32, [3,3], rate=1, activation_fn=lrelu, scope='l_conv4')
        l_conv5=slim.conv2d(l_conv4, 32, [3,3], rate=1, activation_fn=lrelu, scope='l_conv5')
        l_conv6=slim.conv2d(l_conv5, 1,[1,1], rate=1, activation_fn=None,scope='l_conv6')
        L_out = tf.sigmoid(l_conv6)

    return L_out
    
def DenoiseNet(input_R, input_I, input_S, is_training=True, layer_num=15, channel=64):
    with tf.variable_scope('DenoiseNet'):
        S_F0 = tf.layers.conv2d(input_S, 32, 3, padding='same', activation=my_lrelu)
        Scale = tf.layers.conv2d(S_F0, 64, 3, padding='same', activation=tf.nn.sigmoid)       
        S_F2 = tf.layers.conv2d(input_S, 32, 3, padding='same', activation=my_lrelu)
        Shift = tf.layers.conv2d(S_F2, 64, 3, padding='same', activation=tf.nn.sigmoid)
        
        out_R = RDN(input_R, input_I, Scale, Shift, 3, 64)
        
    return out_R

def resBlock(x, nChannels_, index):
    with tf.variable_scope('resBlock%d' % index):
        out0 = tf.layers.conv2d(x, nChannels_, 3, padding='same', activation=my_lrelu)
        out1 = tf.layers.conv2d(out0, nChannels_, 3, padding='same')
        out = tf.add(x, out1)
    return out
    
def RIRB(input_R, scale, shift, nChannels, nResBlock, index):
    with tf.variable_scope('RIRB%d' % index):
        outp = input_R
        outp = tf.layers.conv2d(outp, nChannels, 1, padding='same')
        for i in range(2):
            outp = resBlock(outp, nChannels, i)
        
        outp = tf.layers.conv2d(outp, nChannels, 1, padding='valid', use_bias=False)
        SFA = tf.add(tf.multiply(outp, scale+0.1), shift)
        outp = tf.layers.conv2d(SFA, nChannels, 1, padding='same')
        
        for i in range(2):
            outp = resBlock(outp, nChannels, i+2)
            
        out_res = concat([outp, input_R])
        oute = tf.layers.conv2d(out_res, nChannels, 1, padding='valid')
        
    return oute

def RDB(input_R, nChannels, nResBlock, index):
    nChannels_ = nChannels
    with tf.variable_scope('RDB%d' % index):
        outp = input_R
        outp = tf.layers.conv2d(outp, nChannels, 1, padding='same')
        for i in range(nResBlock):
            outp = resBlock(outp, nChannels_, i)

        # number of channels!
        outp = concat([outp, input_R])
        outp = tf.layers.conv2d(outp, nChannels, 1, padding='valid')
        
    return outp

# Multiscale or not
def RDN(input_R, input_I, scale, shift, nChannel, nfeat, nResBlock=4, growthRate=0):
    with tf.variable_scope('RDN'):    
        input_all = concat([input_R, input_I])
        
        F_ = tf.layers.conv2d(input_all, nfeat, 3, padding='same')
        F_0 = tf.layers.conv2d(F_, nfeat, 3, padding='same')
        F_1 = RDB(F_0, nfeat, nResBlock, 1)
        F_2 = RIRB(F_1, scale, shift, nfeat, nResBlock, 2)
        F_3 = RDB(F_2, nfeat, nResBlock, 3)
        F_4 = RDB(F_3, nfeat, nResBlock, 4)
        F_F1 = concat([F_4, F_])
        F_5 = RDB(F_F1, nfeat, nResBlock, 5)
        F_6 = RDB(F_5, nfeat, nResBlock, 6)
        F_7 = RIRB(F_6, scale, shift, nfeat, nResBlock, 7)
        F_8 = RDB(F_7, nfeat, nResBlock, 8)
        FF = concat([F_0, F_F1, F_8])
        
        FdLF = tf.layers.conv2d(FF, nfeat, 1, padding='valid')
        FGF = tf.layers.conv2d(FdLF, nfeat, 3, padding='same')
        FDF = tf.add(FGF, F_)
        us = tf.layers.conv2d(FDF, nfeat, 3, padding='same', activation = my_lrelu)
        outputs = tf.layers.conv2d(us, nChannel, 3, padding='same')
        #outputs = tf.add(tf.layers.conv2d(us, nChannel, 3, padding='same'), input_R)
    
    return outputs

class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5
        self.DenoiseNet_layer_num = 10

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')
        self.ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_ratio')
        self.input_seg = tf.placeholder(tf.float32, [None, None, None, 3], name='input_seg')

        [R_low, I_low] = DecomNet(self.input_low)
        [R_high, I_high] = DecomNet(self.input_high)
        
        S_low_l, S_low = LayerSegNet(R_low)
        S_high_l, S_high = LayerSegNet(self.input_high)
        
        R_den = DenoiseNet(R_low, I_low, S_low, layer_num=self.DenoiseNet_layer_num)
        
        I_delta = RelightNet(I_low, R_den, self.ratio)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])
        
        output_S = R_den * I_delta_3

        self.output_R_low = R_low
        self.output_R_high = R_high
        self.output_I_high = I_high_3
        self.output_I_low = I_low_3
        self.output_I_delta = I_delta_3
        self.output_S = output_S
        self.output_R_den = R_den
        self.S_low = S_low
        self.S_high = S_high
        input_s = tf.stop_gradient(self.input_seg)

        #------Decom
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - self.input_high))
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
        self.i_mutual_loss = self.mutual_i_loss(I_low, I_high)
        self.i_input_mutual_loss_high = self.mutual_i_input_loss(I_high, self.input_high)
        self.i_input_mutual_loss_low = self.mutual_i_input_loss(I_low, self.input_low)
        
        #------Segmantation
        self.S_loss_mutual = tf.reduce_mean(tf.abs(S_low -  S_high))
        self.S_loss_low = tf.reduce_mean(cross_entropy_loss(input_s, S_low_l, 128, 16))
        self.S_loss_high = tf.reduce_mean(cross_entropy_loss(input_s, S_high_l, 128, 16))
        
        #------Relight
        self.relight_loss_final_square = tf.reduce_mean(tf.square(I_delta_3 * R_den - self.input_high))
        self.relight_loss_final_grad = grad_loss(I_delta_3 * R_den, self.input_high)
        self.relight_loss_ssim = ssim_loss(I_delta_3 * R_den, self.input_high)
        self.relight_loss_ratio = tf.abs(tf.reduce_mean(self.ratio * self.input_low - I_delta_3 * R_den))

        
        #------Restoration
        self.ssim_loss = ssim_loss(R_den, R_high)        
        self.denoise_square_loss = tf.losses.mean_squared_error(R_den, R_high)
        self.denoise_grad_loss = grad_loss(R_den, R_high)

        self.loss_Decom = 1 * self.recon_loss_high + 1 * self.recon_loss_low \
                            + 0.01 * self.equal_R_loss + 0.2 * self.i_mutual_loss \
                            + 0.15 * self.i_input_mutual_loss_high + 0.15 * self.i_input_mutual_loss_low        
        self.loss_Seg0 = self.S_loss_high
        self.loss_Seg1 = self.S_loss_low + self.S_loss_high + 0.1 * self.S_loss_mutual
        self.loss_Relight = self.relight_loss_final_square + 0.5 * self.relight_loss_ssim + 0.5 * self.relight_loss_final_grad + 0.1*self.relight_loss_ratio
        self.loss_Denoise = self.denoise_square_loss + 0.5 * self.denoise_grad_loss + 0.5 * self.ssim_loss
        

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.var_Seg = [var for var in tf.trainable_variables() if 'SegNet' in var.name]
        self.var_Denoise = [var for var in tf.trainable_variables() if 'DenoiseNet' in var.name] + [var for var in tf.trainable_variables() if 'SegNet' in var.name]
        self.var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name] # + [var for var in tf.trainable_variables() if 'DenoiseNet' in var.name]
        

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
        self.train_op_Seg = optimizer.minimize(self.loss_Seg1, var_list = self.var_Seg)
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)
        self.train_op_Denoise = optimizer.minimize(self.loss_Denoise, var_list = self.var_Denoise)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom)
        self.saver_Seg = tf.train.Saver(var_list = self.var_Seg)
        self.saver_Relight = tf.train.Saver(var_list = self.var_Relight)
        self.saver_Denoise = tf.train.Saver(var_list = self.var_Denoise)

        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
        grad_min = tf.reduce_min(gradient_orig)
        grad_max = tf.reduce_max(gradient_orig)
        grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
        return grad_norm
        
    def mutual_i_loss(self, input_I_low, input_I_high):
        low_gradient_x = self.gradient(input_I_low, "x")
        high_gradient_x = self.gradient(input_I_high, "x")
        x_loss = (low_gradient_x + high_gradient_x)* tf.exp(-10*(low_gradient_x+high_gradient_x))
        low_gradient_y = self.gradient(input_I_low, "y")
        high_gradient_y = self.gradient(input_I_high, "y")
        y_loss = (low_gradient_y + high_gradient_y) * tf.exp(-10*(low_gradient_y+high_gradient_y))
        mutual_loss = tf.reduce_mean( x_loss + y_loss) 
        return mutual_loss

    def mutual_i_input_loss(self, input_I_low, input_im):
        input_gray = tf.image.rgb_to_grayscale(input_im)
        low_gradient_x = self.gradient(input_I_low, "x")
        input_gradient_x = self.gradient(input_gray, "x")
        x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
        low_gradient_y = self.gradient(input_I_low, "y")
        input_gradient_y = self.gradient(input_gray, "y")
        y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
        mut_loss = tf.reduce_mean(x_loss + y_loss) 
        return mut_loss
        
    def evaluate(self, epoch_num, eval_low_data, eval_high_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            print(eval_low_data[idx].shape)
            h, w, _ = eval_low_data[idx].shape

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], feed_dict={self.input_low: input_low_eval})
                save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)
                continue
            
            #decom_i_low, decom_i_high = self.sess.run([self.output_I_low, self.output_I_high], feed_dict={self.input_low: input_low_eval, self.input_high: input_high_eval})
            ratio = (1/np.mean(((input_low_eval))/(input_high_eval+0.0001)))+0.0001
            i_low_data_ratio = np.ones([h, w])*ratio
            i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
            i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)

            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta], feed_dict={self.input_low: input_low_eval, self.ratio: i_low_ratio_expand2}) #------0------#
                save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)
            if train_phase == "Seg":
                result_1, result_2 = self.sess.run([self.S_low, self.input_low], feed_dict={self.input_low: input_low_eval})
                save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)
            if train_phase == "Denoise":
                result_1, result_2 = self.sess.run([self.output_R_den, self.output_R_low], feed_dict={self.input_low: input_low_eval, self.ratio: i_low_ratio_expand2})
                save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

            #save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

    def train(self, train_low_data, train_high_data, train_seg_data, eval_low_data, eval_high_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase): #--------0-------#
        assert len(train_low_data) == len(train_high_data)
        assert len(train_low_data) == len(train_seg_data) #-------0-------#
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            R_loss = self.equal_R_loss
            S_loss = self.recon_loss_high
            saver = self.saver_Decom
            lr[:]=0.001
            epoch = 2000
        elif train_phase == "Seg":
            train_op = self.train_op_Seg
            train_loss = self.loss_Seg1
            epoch = 500
            patch_size = 128
            batch_size = 16
            lr[:]=0.0001
            lr[50:]=lr[0] / 5.0
            eval_every_epoch = 50
            R_loss = self.S_loss_high
            S_loss = self.S_loss_low
            saver = self.saver_Seg
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            epoch = 500
            patch_size = 128
            lr[200:] = lr[0] / 10.0
            eval_every_epoch = 100
            R_loss = self.relight_loss_final_grad
            S_loss = self.relight_loss_final_square
            saver = self.saver_Relight
        elif train_phase == "Denoise":
            train_op = self.train_op_Denoise
            train_loss = self.loss_Denoise
            R_loss = self.denoise_grad_loss
            S_loss = self.denoise_square_loss
            epoch = 1000
            lr[400:] = lr[0] / 10.0
            eval_every_epoch = 100
            saver = self.saver_Denoise

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        
        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        if train_phase != "Decom":
            self.evaluate(1, eval_low_data, eval_high_data, sample_dir=sample_dir, train_phase=train_phase)

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
                batch_input_seg = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    train_low_crop = train_low_data[image_id][x : x+patch_size, y : y+patch_size, :]
                    train_high_crop = train_high_data[image_id][x : x+patch_size, y : y+patch_size, :]
                    train_seg_crop = train_seg_data[image_id][x : x+patch_size, y : y+patch_size, :]
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_crop, rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_crop, rand_mode)
                    batch_input_seg[patch_id, :, :, :] = data_augmentation(train_seg_crop, rand_mode)
                    
                    ratio = np.mean(train_low_crop/(train_high_crop+0.0001))
                    #print(ratio)
                    i_low_data_ratio = np.ones([patch_size,patch_size])*(1/ratio+0.0001)
                    i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
                    batch_input_ratio[patch_id, :, :, :] = i_low_ratio_expand
                    
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data, train_seg_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data, train_seg_data  = zip(*tmp)

                # train
                _, loss, Rloss, Sloss = self.sess.run([train_op, train_loss, R_loss, S_loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, self.input_seg: batch_input_seg, \
                                                                           self.lr: lr[epoch], self.ratio: batch_input_ratio})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, R_loss: %.6f, S_loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss, Rloss, Sloss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, eval_high_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './ckpts/Decom')
        load_model_status_Seg, _ = self.load(self.saver_Seg, './ckpts/Seg')
        load_model_status_Denoise, _ = self.load(self.saver_Denoise, './ckpts/Denoise')
        load_model_status_Relight, _ = self.load(self.saver_Relight, './ckpts/Relight')
               
        if load_model_status_Decom and load_model_status_Seg and load_model_status_Relight and load_model_status_Denoise:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            input_high_test = np.expand_dims(test_high_data[idx], axis=0)
            h, w, _ = test_low_data[idx].shape
            
            # [decom_i_low, decom_i_high] = self.sess.run([self.output_I_low, self.output_I_high], feed_dict={self.input_low: input_low_test, self.input_high: input_high_test})
            
            ''' ratio for synthetic image set '''
            ratio = (1/np.mean(((input_low_test))/(input_high_test+0.0001)))+0.0001
            
            ''' ratio for real-world image set '''
            # Fixed ratio
            # ratio = 5.0 
            
            # Non-fixed raio
            # mean_low = np.mean(input_low_test)
            # ratio = 1.2 / mean_low 
            
            i_low_data_ratio = np.ones([h, w])*ratio
            i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
            i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)
            
            [R_den, R_low, I_low, I_delta, S] = self.sess.run([self.output_R_den, self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S], feed_dict = {self.input_low: input_low_test, self.ratio: i_low_ratio_expand2}) #----0----#

            if decom_flag != 0:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                #save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                #save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "_S."   + suffix), S)
