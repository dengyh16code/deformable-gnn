from environment import Environment
import time
import cv2 as cv
import numpy as np
import pybullet as p
from geometry import point_geometry,rope_geometry,ring_geometry,cloth_geometry
from datetime import datetime
import utils
import math
import os
import h5py
import random



class Task(object):
    def __init__(self,task_type,normalize_length,target_state = None,transform_angle=0):
        self.dis_para = 10  #10 8
        if ("ring_square" in task_type) or ("cloth_flatten" in task_type):
            self.dis_para = 8
        print("dis_para:",self.dis_para)
        self.normalize_length = normalize_length
        self.task_type = task_type  
        #rope:rope_line, rope_l, rope_v, rope_n
        #ring:ring_circle, ring_square, move_ring
        #cloth:cloth_fold, cloth_flatten, move_cloth
        self.dist_ct = np.float('inf') 
        self.is_done = False
        
        if target_state is not None:
            self.target_state = target_state
            self.target_matrix = utils.get_edge_matrix(self.target_state)
  
        else:
        # generate random target gemo
            if 'rope' in self.task_type: # 1-D
                my_gemo = rope_geometry(geometry_type=self.task_type)
                point_np= utils.transform_points(np.array(my_gemo.keypoints))
                normalized_ratio = 1
                center_point = [160,120]  
                self.target_state = utils.normalize_points(point_np,normalized_ratio,center_point) 
            
            elif 'ring' in self.task_type:
                my_gemo = ring_geometry(geometry_type=self.task_type)
                point_np= utils.transform_points(np.array(my_gemo.keypoints))
                normalized_ratio = 1
                center_point = [160,120]  
                self.target_state = utils.normalize_points(point_np,normalized_ratio,center_point) 
   
            elif 'cloth' in self.task_type: #2-D
                my_gemo = cloth_geometry(geometry_type=self.task_type)
                angle = transform_angle
                self.target_state= utils.transform_points(np.array(my_gemo.keypoints)-[160,120],i=8-angle)+[160,120]

            else:
                raise KeyError
                    
            if "move" in self.task_type:
                direction = np.random.randint(4,size=1)[0]
                pixel_size  = 0.001644
                move_pixel = 30
                if direction == 0:
                    self.target_state[:,0] = self.target_state[:,0]+move_pixel
                    self.move_pos = np.array([0,move_pixel*pixel_size,0])
                elif direction == 1:
                    self.target_state[:,0] = self.target_state[:,0]-move_pixel
                    self.move_pos =  np.array([0,-move_pixel*pixel_size,0])
                elif direction == 2:
                    self.target_state[:,1] = self.target_state[:,1]+move_pixel
                    self.move_pos =  np.array([move_pixel*pixel_size,0,0])
                elif direction == 3:
                    self.target_state[:,1] = self.target_state[:,1]-move_pixel
                    self.move_pos =  np.array([-move_pixel*pixel_size,0,0])

            self.target_matrix = utils.get_edge_matrix(self.target_state)
        

    def get_state_space(self,current_state):
        """get action_space from gemometry"""
        self.current_state = current_state
        self.current_matrix = utils.get_edge_matrix(self.current_state) 
        if 'rope' in self.task_type: 
            self.dist_ct = utils.get_dis_rope(self.current_state,self.target_state)
        else:
            self.dist_ct = utils.get_dis(self.current_state,self.target_state)
        #self.iou = utils.get_iou(self.current_state,self.target_state)
        current_state_return = np.vstack((self.current_state,self.target_state))
        current_matrix_return = np.vstack((self.current_matrix,self.target_matrix))
        current_matrix_return = utils.tosquare(current_matrix_return)

        return current_state_return,current_matrix_return

    def get_reward(self,current_state):
        if 'rope' in self.task_type:
            done = False
            dist_ct_now = utils.get_dis_rope(current_state,self.target_state)
            if dist_ct_now < self.dis_para: #and dist_ct_var<2*self.dis_para:
                done = True
            dis_change = (self.dist_ct-dist_ct_now)/self.dist_ct
        else:
            done = False
            dist_ct_now = utils.get_dis(current_state,self.target_state)
            if dist_ct_now < self.dis_para: #and dist_ct_var<2*self.dis_para:
                done = True
            dis_change = (self.dist_ct-dist_ct_now)/self.dist_ct
        
        if dis_change>0:
            reward = 1
        else:
            reward = 0


        return dis_change,done
        





