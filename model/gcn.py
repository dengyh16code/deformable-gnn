import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import activations,layers,Model,backend,regularizers

class GCN_layer(layers.Layer):
    def __init__(self, fea_dim, out_dim):
        super().__init__()
        self.fea_dim = fea_dim
        self.out_dim = out_dim

    def build(self, input_shape):
        self.wei = self.add_weight(name='wei', shape=[self.fea_dim, self.out_dim], initializer=tf.zeros_initializer())

    def call(self, inputs, support):
        inputs = tf.cast(inputs, dtype=tf.float32)
        support = tf.cast(support, dtype=tf.float32)
        H_t = tf.matmul(support, inputs)
        output = tf.matmul(H_t, self.wei)
        return tf.sigmoid(output)
    
    def get_config(self):  
        config = {"fea_dim":self.fea_dim, "out_dim":self.out_dim}
        base_config = super(GCN_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def GCN_func(input_shape_1, input_shape_2, node_dim=6, input_dim=2, fea_dim=64, out_dim=64, kernel_size = 64*64*3):
    input_data_1 = tf.keras.layers.Input(shape=input_shape_1)
    input_data_2 = tf.keras.layers.Input(shape=input_shape_2)

    '''
    batch_num = input_data_1.shape[0]  
    x = tf.matmul(input_data_2, input_data_1)
    print(x.shape)
    x = tf.reshape(x,(batch_num*node_dim,input_dim))
    x= tf.keras.layers.Dense(fea_dim, activation='softmax')(x)  # a GCN OPE

    x = tf.reshape(x,(batch_num,node_dim,fea_dim))
    x = tf.matmul(input_data_2, x)
    x = tf.reshape(x,(batch_num*node_dim,fea_dim))
    x= tf.keras.layers.Dense(out_dim, activation='softmax')(x)  # a GCN OPE
    x = tf.reshape(x,(batch_num,node_dim,out_dim))
    '''
   
    x = GCN_layer(input_dim, fea_dim)(input_data_1,input_data_2)
    x = GCN_layer(fea_dim, out_dim)(x,input_data_2)

    x = tf.keras.layers.Flatten(input_shape=[node_dim, out_dim])(x)
    out= tf.keras.layers.Dense(kernel_size, activation='softmax')(x)
    return input_data_1,input_data_2,out
    

class GCN(tf.keras.Model):
    def __init__(self, node_dim=6, input_dim=2, fea_dim=64, out_dim=64, kernel_size = 64*64):
        super().__init__()
        self.node_dim = node_dim
        self.input_dim = input_dim
        self.fea_dim = fea_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size

        self.con1 = GCN_layer(self.input_dim, self.fea_dim)
        self.con2 = GCN_layer(self.fea_dim, self.out_dim)

        self.fla1 = tf.keras.layers.Flatten(input_shape=[self.node_dim, self.out_dim])
        self.den1 = tf.keras.layers.Dense(self.kernel_size, activation='softmax')

    def call(self, input_data, support_data):
        inputs_feats = tf.convert_to_tensor(input_data, dtype=tf.float32)
        support_data = tf.convert_to_tensor(support_data, dtype=tf.float32)
       
        hidden1 = self.con1(inputs_feats, support_data)
        unflattened = self.con2(hidden1, support_data)

        undensed = self.fla1(unflattened)
        output = self.den1(undensed)
        return output

