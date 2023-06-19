import time
import argparse
from datetime import datetime
import numpy as np
import os
from environment import Environment
from task import Task
from termcolor import colored
from dataset import Dataset
import utils
import cv2 as cv

object_param ={"rope":{"action_dim":24,"env_type":0},
             "ring":{"action_dim":32,"env_type":1},
             "cloth":{"action_dim":16,"env_type":2}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #simulation params
    parser.add_argument('-disp',default=False,type=bool)  
    parser.add_argument('-max_action',default=20,type=int)

    parser.add_argument('-task_type',default="rope_line",type=str) 
        #rope:rope_line, rope_l, rope_v, rope_n
        #ring:ring_circle, ring_square, ring_move
        #cloth:cloth_fold, cloth_flatten, cloth_fold_a
    parser.add_argument('-data_dir', default='/apdcephfs/share_1150325/francisdeng/deformable_dataset')
    parser.add_argument('-eposide_num',default=1100)   #1100 scene*  20 manipulation
    parser.add_argument('--agent',  default='transporter',type=str)

    args = parser.parse_args()
    args.time_id = time.strftime("%m_%d_%H:%M")

    if not os.path.exists(args.data_dir):
        print("errror")
        exit(0)   
    dataset_dir = os.path.join(args.data_dir,args.task_type)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    object_type = args.task_type[:(args.task_type.find('_'))] 
    action_dim = object_param[object_type]['action_dim']
    env_type = object_param[object_type]['env_type']
    print("...............................")
    print("task_type:",args.task_type)
    print("env_type:",env_type)
    print("...............................")
    
    simu_env = Environment(disp=args.disp,env_type=env_type)
    dataset = Dataset(dataset_dir,args.agent,0)
    simu_env.reset()
    simu_env.start()
    
    '''
    obs_state =simu_env.get_current_state()
    print(obs_state)
    top_down_image,_ = simu_env.render(simu_env.camera_config_up)
    cv.imwrite("test.jpg",top_down_image)
    exit(0)
    '''
    

    while(dataset.num_episodes <args.eposide_num):  
        #load dataser remember     
        task_kwargs = {'task_type':args.task_type,"normalize_length":simu_env.normalize_length,"transform_angle":simu_env.index_angle}
        new_eposide_task = Task(**task_kwargs)

        pick_random,place_random = 0, 0
        if "fold" not in args.task_type:
            pick_random,place_random = simu_env.random_initialize()

        task_total_reward, task_act_num= 0.,0.

        current_state_list = []
        next_state_list = []
        act_list = []
        act_index_list = []
        reward_list = []
        tp_rgb_list = []
        depth_list = []
        multi_rgb_list = []
        done = False
        
        for j in range(args.max_action):
            obs_state =simu_env.get_current_state()   #get observation
            new_eposide_task.get_state_space(obs_state) 
            _,done = new_eposide_task.get_reward(obs_state)
            if done : 
                break
            else:
                current_state,current_matrix = new_eposide_task.get_state_space(obs_state)              
                top_down_image,_ = simu_env.render(simu_env.camera_config_up)
                front_image,front_depth = simu_env.render(simu_env.camera_config[0])
                left_image,left_depth = simu_env.render(simu_env.camera_config[1])
                right_image,right_depth = simu_env.render(simu_env.camera_config[2])

                tp_rgb_list.append(top_down_image)
                multi_rgb_list.append([front_image,left_image,right_image])
                depth_list.append([front_depth,left_depth,right_depth])
                current_state_list.append(current_state)  #save obseravetion
        
                pick_location,place_location = utils.random_action(current_state,env_type)

                pick_point = current_state[pick_location]
                place_point = current_state[place_location+action_dim]
                pick_world,place_world =  simu_env.step(pick_point,place_point)
            
                obs_state_1 = simu_env.get_current_state()   #get observation
                reward,done = new_eposide_task.get_reward(obs_state_1)
                next_state,next_matrix = new_eposide_task.get_state_space(obs_state_1) 

                task_total_reward += reward
                task_act_num += 1 

                next_state_list.append(next_state)
                act_list.append([[pick_point[0],pick_point[1],0,place_point[0],place_point[1],0],
                                          [pick_world[0],pick_world[1],pick_world[2],place_world[0],place_world[1],place_world[2]]])
                act_index_list.append([pick_location,place_location])
                reward_list.append(reward)
                # save action

        if  "ring" in args.task_type: 
            if "move" in args.task_type:
                simu_env.remove_deform()
                print("reset to goal")
                simu_env.reset(center_pos = new_eposide_task.move_pos)
            if "circle" in args.task_type:
                simu_env.remove_deform()
                print("reset to goal")
                simu_env.reset()
           
            top_down_image,_ = simu_env.render(simu_env.camera_config_up)
            front_image,front_depth = simu_env.render(simu_env.camera_config[0])
            left_image,left_depth = simu_env.render(simu_env.camera_config[1])
            right_image,right_depth = simu_env.render(simu_env.camera_config[2])
            #last frame
            tp_rgb_list.append(top_down_image)
            multi_rgb_list.append([front_image,left_image,right_image])
            depth_list.append([front_depth,left_depth,right_depth])

        if task_act_num>0:
            avg_reward = task_total_reward / task_act_num       # caculate the average reward after one task           
            print(colored('Results of eposide{}'.format(dataset.num_episodes),color='blue', attrs=['bold']))
            print(colored('Avg_Reward:{}'.format(avg_reward),color='blue', attrs=['bold']))
            print(colored('Task finish:{}'.format(done),color='blue', attrs=['bold']))
            print('#################################################################')

        if (done or "move" in args.task_type) and task_act_num>0:
            episode_dict = {}
            episode_dict['c_state'] = current_state_list
            episode_dict['n_state'] = next_state_list
            episode_dict['action']  = act_list
            episode_dict['action_index'] = act_index_list
            episode_dict['tp_rgb']  = tp_rgb_list
            episode_dict['multi_rgb'] = multi_rgb_list
            episode_dict['depth']= depth_list
            episode_dict['reward']= reward_list
            episode_dict['done']= done
            episode_dict['id']= dataset.num_episodes
            dataset.add(episode_dict)
        
        simu_env.remove_deform()
        simu_env.reset()
        
    



            
   


     


