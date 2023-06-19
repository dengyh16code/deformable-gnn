#!/usr/bin/env python
from enum import Flag
import os
import sys
import json
import argparse
import cv2
import pickle
import numpy as np
import utils as U
import task
import cameras
from collections import defaultdict
import utils
import os


class Dataset:

    def __init__(self, path,model_type,remote):
        """A simple RGB-D image dataset."""
        self.path = path
        self.episode_id = []
        self.episode_set = []
        self.model_type = model_type
        self.remote = remote

        # Track existing dataset if it exists.
        self.num_episodes = 0
        self.pixel_size = 0.002500#0.003125 0.001875
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.30, 0.70], [-0.4, 0.4], [0, 0.28]])
        
        if ("rope_n" in path) or ("rope_v" in path)  or ("ring_square" in path):
            self.pixel_size = 0.001875
            self.bounds = np.array([[0.35, 0.65], [-0.3, 0.3], [0, 0.28]])
        

        color_path = os.path.join(self.path, 'tp_rgb')
        if os.path.exists(color_path) or (self.remote == 1):
            for fname in sorted(os.listdir(color_path)):
                if '.pkl' in fname:
                    num_samples = int(fname[(fname.find('-') + 1):-4])
                    self.episode_id += [self.num_episodes] * num_samples
                    self.num_episodes += 1

    def add(self, episode_dict):
        current_state_list = episode_dict['c_state'] 
        next_state_list = episode_dict['n_state']
        action_list = episode_dict['action']
        action_index_list = episode_dict['action_index']  
        tp_rgb_list = np.uint8(episode_dict['tp_rgb']) 
        multi_rgb_list = np.uint8(episode_dict['multi_rgb'])
        depth_list =  np.float32(episode_dict['depth'])
        reward_list = episode_dict['reward']
        eposide_id = self.num_episodes
        num_samples = len(reward_list)
                
        def dump(data, field):
            field_path = os.path.join(self.path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f'{eposide_id:06d}-{num_samples}.pkl'
            pickle.dump(data, open(os.path.join(field_path, fname), 'wb'))

        dump(current_state_list, 'current')
        dump(next_state_list, 'next')
        dump(action_list, 'action')
        dump(action_index_list, 'action_index')
        dump(tp_rgb_list, 'tp_rgb')
        dump(multi_rgb_list, 'multi_rgb')
        dump(depth_list, 'depth')
        dump(reward_list, 'reward')

        self.episode_id += [eposide_id] * num_samples
        self.num_episodes += 1

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.episode_set = episodes

    def random_sample(self):
        iepisode = np.random.choice(self.episode_set)
        # Get length of episode, then sample time within it.
        is_episode_sample = np.int32(self.episode_id) == iepisode
        episode_samples = np.argwhere(is_episode_sample).squeeze().reshape(-1)
        i_choose = np.random.choice(range(len(episode_samples)))

        def load(iepisode, field):
            field_path = os.path.join(self.path, field)
            fname = f'{iepisode:06d}-{len(episode_samples)}.pkl'
            return pickle.load(open(os.path.join(field_path, fname), 'rb'))

        if self.model_type == 'transporter' or self.model_type == 'conv_mlp':
            obs = {}
            obs['color'] = load(iepisode, 'multi_rgb')[i_choose]
            obs['depth'] = load(iepisode, 'depth')[i_choose]
            act = load(iepisode, 'action')[i_choose][1]
            assert obs['color'].shape == (3, 480, 640, 3), obs['color'].shape
            assert obs['depth'].shape == (3, 480, 640), obs['depth'].shape
            return obs, act
        
        elif self.model_type == "transporter-goal":
            obs = {}
            obs['color'] = load(iepisode, 'multi_rgb')[i_choose]
            obs['depth'] = load(iepisode, 'depth')[i_choose]
            goal = {}
            goal_index = len(episode_samples)
            goal['color'] = load(iepisode, 'multi_rgb')[goal_index]
            goal['depth'] = load(iepisode, 'depth')[goal_index]         
            act = load(iepisode, 'action')[i_choose][1]           
            assert obs['color'].shape == (3, 480, 640, 3), obs['color'].shape
            assert obs['depth'].shape == (3, 480, 640), obs['depth'].shape
            assert goal['color'].shape == (3, 480, 640, 3), goal['color'].shape
            assert goal['depth'].shape == (3, 480, 640), goal['depth'].shape
            return obs, act, goal
        
        elif "gnn" in self.model_type :
            obs = load(iepisode, 'current')[i_choose]
            act = load(iepisode, 'action_index')[i_choose]           
            return obs, act
        
        elif self.model_type == "transporter-graph":
            goal_index = len(episode_samples)
            obs = load(iepisode, 'tp_rgb')[i_choose]
            kp_pos = load(iepisode, 'current')[i_choose]
            goal = load(iepisode, 'tp_rgb')[goal_index]
            act = load(iepisode, 'action_index')[i_choose]  
            return obs, act, goal, kp_pos

    def process_depth(self, img, cutoff=10):
    # Turn to three channels and zero-out values beyond cutoff.
        w,h = img.shape
        d_img = np.zeros([w,h,3])
        img = img.flatten()
        img[img > cutoff] = 0.0
        img = img.reshape([w,h])
        for i in range(3):
            d_img[:,:,i] = img

        # Scale values into [0,255) and make type uint8.
        assert np.max(d_img) > 0.0
        d_img = 255.0 / np.max(d_img) * d_img
        d_img = np.array(d_img, dtype=np.uint8)
        for i in range(3):
            d_img[:,:,i] = cv2.equalizeHist(d_img[:,:,i])
        return d_img

    def get_heightmap(self,obs):
        """Following same implementation as in transporter.py."""
        heightmaps, colormaps = U.reconstruct_heightmaps(
            obs['color'], obs['depth'], self.camera_config, self.bounds, self.pixel_size)
        colormaps = np.float32(colormaps)
        heightmaps = np.float32(heightmaps)

        # Fuse maps from different views.
        valid = np.sum(colormaps, axis=3) > 0
        repeat = np.sum(valid, axis=0)
        repeat[repeat == 0] = 1
        colormap = np.sum(colormaps, axis=0) / repeat[..., None]
        colormap = np.uint8(np.round(colormap))
        heightmap = np.max(heightmaps, axis=0)
        return colormap, heightmap
        

    def inspect(self, i_ep, save_imgs=True):

        def _load(i_ep, episode_len, field):
            field_path = os.path.join(self.path, field)
            fname = f'{i_ep:06d}-{episode_len}.pkl'
            return pickle.load(open(os.path.join(field_path, fname), 'rb'))

        is_episode_sample = np.int32(self.episode_id) == i_ep
        episode_samples = np.argwhere(is_episode_sample).squeeze().reshape(-1)
        episode_len = len(episode_samples)
        current_l = np.array(_load(i_ep, episode_len, 'current'))
        action_l = np.array(_load(i_ep, episode_len, 'action'))
        tp_rgb_list =  _load(i_ep, episode_len, 'tp_rgb')
        multi_rgb_list  = _load(i_ep, episode_len, 'multi_rgb')
        depth_list  = _load(i_ep, episode_len, 'depth')
        #reward_list = np.array(_load(i_ep, episode_len, 'reward'))

        if save_imgs:
            self._save_images(episode_len,multi_rgb_list,depth_list,tp_rgb_list,action_l,current_l,"check_image/",i_ep)
        
        return tp_rgb_list

    def process_concat_heatmap(self,multi_color, depth,act_loc):
        assert multi_color.shape == (3, 480, 640, 3), multi_color.shape
        assert depth.shape == (3, 480, 640), depth.shape
        for k in range(3):
                c_img = multi_color[k]
                d_img = depth[k]
                assert c_img.dtype == 'uint8', c_img.dtype
                assert d_img.dtype == 'float32', d_img.dtype
                d_img = self.process_depth(img=d_img)
         
        obs_input = {'color': multi_color, 'depth': depth}
        colormap, heightmap =self.get_heightmap(obs_input)
        heightmap_proc = self.process_depth(img=heightmap)

        for p in range(5):
            for q in range(5):
                draw_x,draw_y = utils.position_to_bound_pixel(act_loc,self.bounds,self.pixel_size)
                colormap[int(draw_x+p)][int(draw_y+q)] = [255,0,0]

        c_img_front = multi_color[0]  # Shape (480, 640, 3)
        c_img_front = cv2.resize(c_img_front, (426,320)) # numpy shape: (320,426)
        barrier = np.zeros((320,4,3))  # Black barrier of 4 pixels
        combo = np.concatenate((
                    cv2.cvtColor(c_img_front, cv2.COLOR_BGR2RGB),
                    barrier,
                    cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR),
                    barrier,
                    heightmap_proc),
                axis=1)
        return combo

    def process_show_action(self,current_I,goal_l,act_loc,current_loc):
        assert current_I.shape == (240, 320,3), current_I.shape
        assert goal_l.shape == (240, 320, 3), goal_l.shape
        assert act_loc.shape == (6,), act_loc.shape
        #assert current_loc.shape == (32,2), current_loc.shape
        pick_kp = act_loc[:2]
        place_kp = act_loc[-3:-1]
        kp_num = int(len(current_loc))
        for i in range(kp_num):
            kp = current_loc[i]
            for p in range(5):
                for q in range(5):
                    if i<int(kp_num/2):
                        current_I[int(kp[1]+p)][int(kp[0]+q)] = [0,255,0]
                    else:
                        goal_l[int(kp[1]+p)][int(kp[0]+q)] = [0,255,0]
        for p in range(5):
            for q in range(5):
                current_I[int(pick_kp[1]+p)][int(pick_kp[0]+q)] = [255,0,0]
                goal_l[int(place_kp[1]+p)][int(place_kp[0]+q)] = [255,0,0]
            
        current_I = current_I.swapaxes(0,1)
        goal_l = goal_l.swapaxes(0,1)
                
        barrier = np.zeros((4,240,3))
        combo = np.concatenate((
            cv2.cvtColor(current_I, cv2.COLOR_BGR2RGB),
            barrier,
            cv2.cvtColor(goal_l, cv2.COLOR_RGB2BGR)), axis=0)

        return combo

    def _save_images(self, episode_len, multi_color_l, depth_l,tp_color_l, action_pos,current_pos,outdir, i_ep):       
        for t in range(episode_len):
            goal_multi_color = multi_color_l[episode_len]
            goal_depth = depth_l[episode_len]
            goal_combo = self.process_concat_heatmap(goal_multi_color,goal_depth,action_pos[t][1][3:])
            goal_tp_color = tp_color_l[episode_len]
            current_combo = self.process_concat_heatmap(multi_color_l[t],depth_l[t],action_pos[t][1][:3])
            current_tp_color = tp_color_l[t]
            top_combo = self.process_show_action(current_tp_color,goal_tp_color,action_pos[t][0],current_pos[t])
            barrier = np.zeros((4,754,3))  # Black barrier of 4 pixels
            combo_1 = np.concatenate((
                    current_combo,
                    barrier,
                    goal_combo),
                axis=0)
            barrier_1 = np.zeros((644,4,3))
            combo = np.concatenate((
                    combo_1,
                    barrier_1,
                    top_combo),
                axis=1)
            
            # Optionally include title with more details, but env dependent.
            suffix_all = f'{i_ep:06d}-{t:02d}-OVERALL.png'
            cv2.imwrite(os.path.join(outdir,suffix_all), combo)

class multi_task_Dataset:
    def __init__(self, path,model_type,max_len):
        """A simple RGB-D image dataset."""
        self.task_types = ["rope_line","rope_l","rope_n","rope_v",
                                  "ring_circle","ring_move","ring_square",
                                  "cloth_flatten","cloth_fold","cloth_fold_a"]
        self.choose_posibility = [0.1,0.1,0.1,0.1,
                                               0.1,0.1,0.1,
                                               0.1,0.1,0.1]
        self.path = path
        self.model_type = model_type
        self.max_len = max_len
        for task_type in self.task_types:
            data_path = os.path.join(self.path,task_type)
            assert os.path.exists(data_path),data_path
        
    def random_sample(self):        
        choose_task_type = np.random.choice(self.task_types,p=self.choose_posibility)
        temp_data_path = os.path.join(self.path,choose_task_type)
        temp_dataset = Dataset(temp_data_path,self.model_type,1)
        temp_dataset.set(self.episode_set)
        obs,act =  temp_dataset.random_sample()
        obs_len = int(len(obs)/2)
        expand_num = self.max_len - obs_len
        mask = [1]*len(obs)
        if expand_num>0:
            for i in range(expand_num):
                obs = np.insert(obs,obs_len+i,[0,0],axis=0)
                mask.insert(obs_len+i,0)
                obs = np.insert(obs,2*(obs_len+i)+1,[0,0],axis=0)
                mask.insert(2*(obs_len+i)+1,0)
        assert  len(obs) == self.max_len*2, len(obs) 
        return obs,act,mask

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.episode_set = episodes



if __name__ == '__main__':
    # Assumes we call: `python ravens/dataset.py --path [...]`.
    parser = argparse.ArgumentParser()
        #rope:rope_line, rope_l, rope_v, rope_n
        #ring:ring_circle, ring_square, move_ring
        #cloth:cloth_fold, cloth_flatten, move_cloth
    parser.add_argument('--path', default='/apdcephfs/share_1150325/francisdeng/deformable_dataset')
    parser.add_argument('-task_type',default="ring_move",type=str)
    parser.add_argument('-i_ep',default=1,type=int)
    args = parser.parse_args()
    data_path = os.path.join(args.path,args.task_type)
    dataset = Dataset(data_path,'transporter',0)
    for ep_index in range(10):
        i_ep = ep_index*111
        tp_image_list = dataset.inspect(i_ep=i_ep,save_imgs=False)
        sequence_len = len(tp_image_list) - 1
        if not os.path.exists(args.task_type):
            os.makedirs(args.task_type)
        goal_l = tp_image_list[sequence_len]
        for i in range (sequence_len):
            file_name = os.path.join(args.task_type,str(i_ep)+"_"+str(i)+".png")
            current_I =  tp_image_list[i]
            barrier = np.zeros((240,4,3))
            combo = np.concatenate((
                cv2.cvtColor(current_I, cv2.COLOR_BGR2RGB),
                barrier,
                cv2.cvtColor(goal_l, cv2.COLOR_RGB2BGR)), axis=1)
            cv2.imwrite(file_name,combo)
