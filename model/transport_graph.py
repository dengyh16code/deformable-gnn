#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from models import ResNet43_8s,GCN,GCN_func
import utils

class Pick_model:

    def __init__(self, input_shape,add_mask):

        d_in, d_out = ResNet43_8s(input_shape, 1)
        self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
        self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.add_mask = add_mask

    def forward(self, in_img, mask, apply_softmax=True, save_id=0,save_dir='1'):
        """Forward pass.

        in_img.shape: (240, 320, 6)
        input_data.shape: (240, 320, 6), output (None, 240, 320, 6)
        """
        input_data = tf.convert_to_tensor(in_img, dtype=tf.float32)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)

        # Forward pass
        output = self.model(input_data)
        #self.visualize_logits(output, name='pick',save_id=save_id,save_dir=save_dir)
        if self.add_mask:
            output =  tf.multiply(output,mask) #add_mask

        
        #self.visualize_heatmap(mask, name='pick_Q_value',save_id=save_id,save_dir=save_dir)
        

        output = tf.reshape(output, (output.shape[0], np.prod(output.shape[1:])))
        if apply_softmax:
            output = np.float32(output).reshape(output.shape)
               
        return output

    def train(self, in_img, pick,masks):
        with tf.GradientTape() as tape:
            output = self.forward(in_img, masks,apply_softmax=False)

            # Compute label
            label_size = in_img.shape[0:3]
            label = np.zeros(label_size)
            for i in range(len(pick)): 
                label[i][int(pick[i][1])][int(pick[i][0])] = 1
            label = label.reshape(label.shape[0], np.prod(label.shape[1:]))
            label = tf.convert_to_tensor(label, dtype=tf.float32)
            
            loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
            loss = tf.reduce_mean(loss)

        # Backpropagate
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(
            zip(grad, self.model.trainable_variables))

        return np.float32(loss)

    def test(self, in_img, pick,masks):
        with tf.GradientTape() as tape:
            output = self.forward(in_img, masks,apply_softmax=False)
            output = tf.reshape(output, (output.shape[0],240,320))
            output_np = output.numpy()
            all_dis = 0
            for i in range(len(pick)): 
                output_map = output_np[i]
                max_loc = np.where(output_map==np.max(output_map))         
                output_location = [np.where(output_map==np.max(output_map))[0][0],np.where(output_map==np.max(output_map))[1][0]]
                dis = np.sqrt((output_location[0]-pick[i][1])**2+(output_location[1]-pick[i][0])**2)
                all_dis += dis

        return all_dis
        

    def load(self, path):
        self.model.load_weights(path)

    def save(self, filename):
        self.model.save(filename)


    def visualize_heatmap(self,Q_heatmap,name,save_id,save_dir="1"):
        original_shape = Q_heatmap.shape
        Q_heatmap = tf.reshape(Q_heatmap, (1, np.prod(original_shape)))
        vis_transport = np.float32(Q_heatmap).reshape(original_shape)
        vis_transport = vis_transport[0]
        #vis_transport = vis_transport - vis_transport[0][0]
        #vis_transport = np.maximum(vis_transport, 0)
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)
        vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)

        #plt.subplot(1, 1, 1)
        #plt.axis('off')
        #plt.title(f'place q: {name}', fontsize=15)
        #plt.imshow(vis_transport)
        #plt.tight_layout()
        dir_name = os.path.join('record/',save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_file_name = str(save_id) +'_'+name+".jpg"
        save_file_name = os.path.join(dir_name,save_file_name)
        cv2.imwrite(save_file_name,vis_transport)
        #plt.savefig(save_file_name)
        #plt.show()
    
    def visualize_logits(self, logits, name, save_id,save_dir="1"):
        """Given logits (BEFORE tf.nn.convolution), get a heatmap.

        Here we apply a softmax to make it more human-readable. However, the
        tf.nn.convolution with the learned kernels happens without a softmax
        on the logits. [Update: wait, then why should we have a softmax,
        then? I forgot why we did this ...]
        """
        original_shape = logits.shape
        logits = tf.reshape(logits, (1, np.prod(original_shape)))
        # logits = tf.nn.softmax(logits)  # Is this necessary?
        vis_transport = np.float32(logits).reshape(original_shape)
        vis_transport = vis_transport[0]
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)

        # Only if we're saving with cv2.imwrite()
        vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(f'tmp/logits_{name}.png', vis_transport)

        plt.subplot(1, 1, 1)
        plt.axis('off')
        #plt.title(f'Logits: {name}', fontsize=15)
        plt.imshow(vis_transport)
        plt.tight_layout()
        dir_name = os.path.join('record/',save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_file_name = str(save_id) +'_'+name+".jpg"
        save_file_name = os.path.join(dir_name,save_file_name)
        plt.savefig(save_file_name)
        #plt.show()
   

class Place_model:

    def __init__(self,input_shape,crop_size,action_dim,add_mask,add_gcn=False):

        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.odim = output_dim = 3
        self.add_mask = add_mask
        self.add_gcn = add_gcn

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        input_shape = np.array(input_shape)
        input_shape[0:2] += self.pad_size * 2
        input_shape = tuple(input_shape)

        # 2 fully convolutional ResNets.
        in0, out0 = ResNet43_8s(input_shape, output_dim, prefix='s0_')  #query network
        in1, out1 = ResNet43_8s(input_shape, output_dim, prefix='s1_')  #key network
        if self.add_gcn:
            in2,in2_1,out2 = GCN_func((action_dim,2),(action_dim,action_dim))
            in3,in3_1,out3 = GCN_func((action_dim,2),(action_dim,action_dim))    
        
        if self.add_gcn:
            self.model = tf.keras.Model(inputs=[in0,in1,in2,in2_1,in3,in3_1], outputs=[out0,out1,out2,out3])
        else:
            self.model = tf.keras.Model(inputs=[in0, in1], outputs=[out0, out1])
        self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def forward(self, in_img, pick, mask, kp_c, c_edge, kp_g, g_edge, apply_softmax=True, save_id=1,save_dir='1'):
        # can forward only one image every_time
        # input image --> TF tensor
        input_data = np.pad(in_img, self.padding, mode='constant')    # (1,304,384,6)           
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)                    # (1,304,384,6)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,304,384,6)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)

        if self.add_gcn:
            kp_c = tf.expand_dims(kp_c, 0)
            kp_g = tf.expand_dims(kp_g, 0)
            c_edge = tf.expand_dims(c_edge, 0)
            g_edge = tf.expand_dims(g_edge, 0)
            query_logits, key_logits,mat_1,mat_2 = self.model([in_tensor,in_tensor, kp_c, c_edge, kp_g, g_edge])
            mat_err = mat_2-mat_1
            mat_err = tf.reshape(mat_err,(1,64,64,3))

        else:
            query_logits, key_logits = self.model([in_tensor, in_tensor])

        crop = tf.identity(query_logits)                            # (1,304,384,3)
        kernel = crop[:,
                      int(pick[1]):int(pick[1] + self.crop_size),
                      int(pick[0]):int(pick[0] + self.crop_size),
                      :]

        if self.add_gcn:
            kernel = kernel + mat_err
        # Cross-convolve `in_x_goal_logits`. Padding kernel: (1,64,64,3) --> (65,65,3,1).
        kernel_paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        kernel = tf.pad(kernel, kernel_paddings, mode='CONSTANT')
        kernel = tf.transpose(kernel, [1, 2, 3, 0])
        output = tf.nn.convolution(key_logits, kernel, data_format="NHWC")
        output = (1 / (self.crop_size**2)) * output #(1,240,320,1)
        if self.add_mask:
            output =  tf.multiply(output,mask) #add_mask

        if apply_softmax:
            output_shape = output.shape
            output = tf.reshape(output, (1, np.prod(output.shape)))
            output = tf.nn.softmax(output)
            output = np.float32(output).reshape(output_shape[1:])
        
        
        #self.visualize_transport(crop, kernel,save_id=save_id,save_dir=save_dir)
        #self.visualize_logits(key_logits, name='key',save_id=save_id,save_dir=save_dir)
        #self.visualize_logits(query_logits, name='query',save_id=save_id,save_dir=save_dir)
        #self.visualize_heatmap(output, name='place_Q_value',save_id=save_id,save_dir=save_dir)
        

        return output
        

    def train(self, in_img, pick, place, masks,  kp_cs, c_edges, kp_gs ,g_edges):

        with tf.GradientTape() as tape:
  
            logits = ()
            for i in range(len(in_img)):
                a_img = in_img[i]
                a_pick = pick[i]
                a_mask = masks[i]
                a_c_edge = c_edges[i]
                a_kp_c = kp_cs[i]
                a_g_edge = g_edges[i]
                a_kp_g = kp_gs[i]
                output = self.forward(a_img, a_pick, a_mask, a_kp_c, a_c_edge, a_kp_g, a_g_edge, apply_softmax=False)
                output = tf.reshape(output, (1, np.prod(output.shape)))
                logits += (output,)
            logits = tf.concat(logits, axis=0)

            # Compute label
            label_size = in_img.shape[0:3]
            label = np.zeros(label_size)  
            for i in range(len(in_img)):  
                label[i][int(place[i][1])][int(place[i][0])] = 1
            label = label.reshape(label.shape[0], np.prod(label.shape[1:]))
            label = tf.convert_to_tensor(label, dtype=tf.float32)
            
            # Compute loss after re-shaping the output.
            loss = tf.nn.softmax_cross_entropy_with_logits(label, logits)
            loss = tf.reduce_mean(loss)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grad, self.model.trainable_variables))


        return np.float32(loss)


    def test(self, in_img, pick, place, masks,  kp_cs, c_edges, kp_gs ,g_edges):

        with tf.GradientTape() as tape:
  
            logits = ()
            for i in range(len(in_img)):
                a_img = in_img[i]
                a_pick = pick[i]
                a_mask = masks[i]
                a_c_edge = c_edges[i]
                a_kp_c = kp_cs[i]
                a_g_edge = g_edges[i]
                a_kp_g = kp_gs[i]
                output = self.forward(a_img, a_pick, a_mask, a_kp_c, a_c_edge, a_kp_g, a_g_edge, apply_softmax=False)
                output = tf.reshape(output, (1, np.prod(output.shape)))
                logits += (output,)
            logits = tf.concat(logits, axis=0)

            logits = tf.reshape(logits, (logits.shape[0],240,320))
            output_np = logits.numpy()
            dis = 0
            for i in range(len(place)): 
                output_map = output_np[i]
                output_location = [np.where(output_map==np.max(output_map))[0][0],np.where(output_map==np.max(output_map))[1][0]]
                dis += np.sqrt((output_location[0]-place[i][1])**2+(output_location[1]-place[i][0])**2)

        return dis


    def save(self, fname):
        self.model.save(fname)

    def load(self, fname):
        self.model.load_weights(fname)

    #-------------------------------------------------------------------------
    # Visualization.
    #-------------------------------------------------------------------------
    def visualize_heatmap(self,Q_heatmap,name,save_id,save_dir='1'):
        original_shape = Q_heatmap.shape
        Q_heatmap = tf.reshape(Q_heatmap, (1, np.prod(original_shape)))
        vis_transport = np.float32(Q_heatmap).reshape(original_shape)
        vis_transport = vis_transport[0]
        #vis_transport = np.where(vis_transport > vis_transport[0][0], vis_transport, 0)
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW) #RAINBOW
        #vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)


        #plt.axis('off')
        #plt.subplot(1, 1, 1)
        #plt.title(f'place q: {name}', fontsize=15)
        #plt.imshow(vis_transport)
        #plt.tight_layout()

        dir_name = os.path.join('record/',save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_file_name = str(save_id) +'_'+name+".jpg"
        save_file_name = os.path.join(dir_name,save_file_name)
        #plt.savefig(save_file_name)
        cv2.imwrite(save_file_name,vis_transport)
        #plt.show()


    def visualize_transport(self, crop, kernel, save_id,save_dir='1'):
        '''
        crop.shape: (1,64,64,3)
        kernel.shape = (65,65,3,1)
        '''
        print(crop.shape)
        
        def colorize(img):
            # I don't think we have to convert to BGR here...
            img = img - np.min(img)
            img = 255 * img / np.max(img)
            img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_RAINBOW)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        kernel = (tf.transpose(kernel, [3, 0, 1, 2])).numpy() # 1,3,65,65
        plt.axis('off')
        #plt.imshow(crop[0])
        #plt.show()
        processed = colorize(img=kernel[0])
        plt.subplot(1, 1, 1)
        plt.axis('off')
        plt.imshow(processed)
        plt.tight_layout()
        dir_name = os.path.join('record/',save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_file_name = str(save_id) +"_kernel"+".jpg"
        save_file_name = os.path.join(dir_name,save_file_name)
        plt.savefig(save_file_name)
        #plt.show()


    def visualize_logits(self, logits, name, save_id,save_dir='1'):
        """Given logits (BEFORE tf.nn.convolution), get a heatmap.

        Here we apply a softmax to make it more human-readable. However, the
        tf.nn.convolution with the learned kernels happens without a softmax
        on the logits. [Update: wait, then why should we have a softmax,
        then? I forgot why we did this ...]
        """
        original_shape = logits.shape
        logits = tf.reshape(logits, (1, np.prod(original_shape)))
        # logits = tf.nn.softmax(logits)  # Is this necessary?
        vis_transport = np.float32(logits).reshape(original_shape)
        vis_transport = vis_transport[0]
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)
        
               # Only if we're saving with cv2.imwrite()
        vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(f'tmp/logits_{name}.png', vis_transport)

        plt.subplot(1, 1, 1)
        plt.axis('off')
        #plt.title(f'Logits: {name}', fontsize=15)
        plt.imshow(vis_transport)
        plt.tight_layout()
        dir_name = os.path.join('record/',save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_file_name = str(save_id) +'_'+name+".jpg"
        save_file_name = os.path.join(dir_name,save_file_name)
        plt.savefig(save_file_name)
        #plt.show()



if __name__ == "__main__":
    my_model = Place_model(input_shape=[240,320,6],crop_size=64)
    test_examples,pick_labels,place_labels= utils.load_data_from_h5("test_ring.h5",'transporter_graph')
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples,(pick_labels,place_labels)))
    BATCH_SIZE = 10
    test_dataset = test_dataset.shuffle(2021).batch(BATCH_SIZE)
    for test_images,(pick_pos,place_pos) in test_dataset:
        my_model.train(test_images,pick_pos,place_pos)

        

        
