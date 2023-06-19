#!/usr/bin/env python

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from models import Pick_model
from models import Place_model
import tensorflow as tf
import utils
import logging
import cv2 as cv
import argparse


class Transporter_graph_Agent:

    def __init__(self,name,task,action_dim):

        self.name = name
        self.task = task
        self.total_iter = 0
        self.crop_size = 64
        self.input_shape = (240, 320, 6)
        self.action_dim = action_dim
        self.add_mask = True
        self.add_gcn = True
        self.models_dir = os.path.join('checkpoints', self.name)
        self.pick_model = Pick_model(input_shape=self.input_shape,add_mask=self.add_mask )
        self.place_model = Place_model(input_shape=self.input_shape,crop_size=self.crop_size,action_dim=action_dim,add_mask=self.add_mask,add_gcn=self.add_gcn)


    def train(self, dataset, num_iter, writer):
        tf.keras.backend.set_learning_phase(1)
        for i in range(num_iter):
            obs, act, goal, kp_pos = dataset.random_sample()
            
            pick_pos = [kp_pos[act[0]]]
            place_pos = [kp_pos[self.action_dim+act[1]]]

            kp_pos_c = kp_pos[:self.action_dim,:]
            kp_pos_g = kp_pos[self.action_dim:,:]
            c_edge,c_map,kp_c = self.get_graph_feature(kp_pos_c)
            g_edge,g_map,kp_g = self.get_graph_feature(kp_pos_g)
            
            images = np.concatenate((obs,goal),axis=2)
            images = np.expand_dims(images,0)
            
            view_data = False
            if view_data:
                self.process_show_action(obs,goal,pick_pos,place_pos,c_map,g_map)

            loss0 = self.pick_model.train(images, pick_pos,c_map)
            loss1 = self.place_model.train(images, pick_pos, place_pos,g_map,kp_c,c_edge,kp_g,g_edge) 
            with writer.as_default():
                tf.summary.scalar('pick_loss', loss0, step=self.total_iter+i)  
                tf.summary.scalar('place_loss', loss1, step=self.total_iter+i) 

            print(f'Train Iter: {self.total_iter + i} Loss: {loss0:.4f} {loss1:.4f}')

        self.total_iter += num_iter
        self.save()

    def test(self, test_dataset_dir):
        images_data,pick_labels,place_labels,c_images,g_images= utils.load_data_from_h5(test_dataset_dir,'transporter_graph')
        test_dataset = tf.data.Dataset.from_tensor_slices(((c_images,g_images,images_data),(pick_labels,place_labels)))
        BATCH_SIZE = 8
        test_dataset = test_dataset.shuffle(2021).batch(BATCH_SIZE)
        err_pick_all = 0
        err_place_all = 0
        for (c_image,g_image,images),(pick_pos,place_pos) in test_dataset:
            c_edge,c_map,kp_c = self.get_feature(c_image)
            g_edge,g_map,kp_g = self.get_feature(g_image)
            err_pick = self.pick_model.test(images, pick_pos,c_map)
            err_place = self.place_model.test(images, pick_pos, place_pos,g_map,kp_c,c_edge,kp_g,g_edge) 
            print(err_pick)
            print(err_place)
            err_pick_all += err_pick/(BATCH_SIZE*len(test_dataset))
            err_place_all += err_place/(BATCH_SIZE*len(test_dataset))
        print("err_pick:",err_pick_all)
        print("err_place:",err_place_all)
 
    def process_show_action(self,obs,goal,pick_pos,place_pos,c_map,g_map):

        #assert current_loc.shape == (32,2), current_loc.shape
        pick_kp = pick_pos[0]
        place_kp = place_pos[0]
        kp_num = int(len(pick_kp))
        for p in range(5):
            for q in range(5):
                obs[int(pick_kp[1]+p)][int(pick_kp[0]+q)] = [0,255,0]
                goal[int(place_kp[1]+p)][int(place_kp[0]+q)] = [0,255,0]
            
        obs_I = obs.swapaxes(0,1)
        goal_I = goal.swapaxes(0,1)
                
        barrier = np.zeros((4,240,3))
        combo = np.concatenate((
            cv2.cvtColor(obs_I, cv2.COLOR_BGR2RGB),
            barrier,
            cv2.cvtColor(goal_I, cv2.COLOR_RGB2BGR),
            ), axis=0)

        c_map_norm = (c_map[0]-np.min(c_map[0]))/(np.max(c_map[0])-np.min(c_map[0]))
        g_map_norm = (g_map[0]-np.min(g_map[0]))/(np.max(g_map[0])-np.min(g_map[0]))
        
        
        combo_1 = np.concatenate((
            cv2.cvtColor(c_map_norm.swapaxes(0,1)*255.0, cv2.COLOR_GRAY2RGB),
            barrier,
            cv2.cvtColor(g_map_norm.swapaxes(0,1)*255.0, cv2.COLOR_GRAY2RGB),
            ), axis=0)
        

        barrier_1 = np.zeros((644,4,3))
        combo_2 = np.concatenate((combo,barrier_1,combo_1),axis=1)
        cv2.imwrite("combo.jpg",combo_2)
        exit(0)

        return combo_2

    def act(self, obs_dic):
        
        obs = obs_dic["curr"]
        kp_pos = obs_dic["kp_pos"]
        goal = obs_dic["goal"]

        kp_pos_c = kp_pos[:self.action_dim,:]
        kp_pos_g = kp_pos[self.action_dim:,:]
        c_edge,c_map,kp_c = self.get_graph_feature(kp_pos_c)
        g_edge,g_map,kp_g = self.get_graph_feature(kp_pos_g)
            
        images = np.concatenate((obs,goal),axis=2)
        images = np.expand_dims(images,0)

        pick_q = self.pick_model.forward(images,c_map,apply_softmax=False)
        pick_q_reshape = tf.reshape(pick_q,(240,320))
        pick_q_np = pick_q_reshape.numpy()
        max_loc = np.where(pick_q_np==np.max(pick_q_np))
        pick_pos = [max_loc[1][0],max_loc[0][0]] 

        images = images[0]
        g_map = g_map[0]
        kp_c = kp_c[0]
        c_edge = c_edge[0]
        kp_g = kp_g[0]
        g_edge = g_edge[0]
    
        place_q = self.place_model.forward(images, pick_pos,g_map,kp_c,c_edge,kp_g,g_edge,apply_softmax=False)   
        place_q_reshape = tf.reshape(place_q,(240,320))
        place_q_np = place_q_reshape.numpy()
        max_loc_p = np.where(place_q_np==np.max(place_q_np))
        place_pos = [max_loc_p[1][0],max_loc_p[0][0]]  #notice
        
        act_dic  = {}
        act_dic['primitive'] = 'pick_place'
        params = {'pose0':pick_pos,'pose1':place_pos}
        act_dic['params'] = params

        return act_dic
 

    def get_graph_feature(self,kp_list,std=10.0):
        edge_np = utils.get_edge_matrix(kp_list)
        kp_shape =  (1,) + kp_list.shape
        edge_shape = (1,) + edge_np.shape
        edge_np = edge_np.reshape(edge_shape)
        kp_list = np.float32(kp_list.reshape(kp_shape))    
        gauss_map = utils._get_gaussian_maps(kp_list,[240,320],std)
        guass_map_reduce = tf.reduce_sum(gauss_map,3)
        gauss_map_np = guass_map_reduce.numpy()
        return edge_np,gauss_map_np,kp_list

    def load(self, total_iter):
        """Load pre-trained models."""
        #pick_fname = 'pick-ckpt-%d.h5' % train_epoch
        pick_fname = 'pick-ckpt-%d.h5' % total_iter
        place_fname = 'place-ckpt-%d.h5' % total_iter
        pick_fname = os.path.join(self.models_dir, pick_fname)
        place_fname = os.path.join(self.models_dir, place_fname)
        self.pick_model.load(pick_fname)
        self.place_model.load(place_fname)
        self.total_iter = total_iter
        print("load wieght")

    def save(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        pick_fname = 'pick-ckpt-%d.h5' % self.total_iter
        place_fname = 'place-ckpt-%d.h5' % self.total_iter
        pick_fname = os.path.join(self.models_dir, pick_fname)
        place_fname = os.path.join(self.models_dir, place_fname)
        self.pick_model.save(pick_fname)
        self.place_model.save(place_fname)


if __name__ == "__main__":
    my_agent = Transporter_graph_Agent()
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_type',default="cloth_fold_a",type=str) 
    parser.add_argument('-data_dir', default='/apdcephfs/share_1150325/francisdeng/deformable_dataset')
    dataset_dir = os.path.join(args.data_dir,args.task_type)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)    
    dataset = Dataset(dataset_dir,args.agent,args.remote)
    #rope:rope_line, rope_l, rope_v, rope_n
    #ring:ring_circle, ring_square, ring_move
    #cloth:cloth_fold, cloth_flatten,multi_task
    my_agent.load(train_epoch=14)
    my_agent.train(train_dataset_dir="train.h5",max_epoch=40)
