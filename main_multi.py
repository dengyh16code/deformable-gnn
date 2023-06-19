
from cgitb import reset
import datetime
import os
import time
import argparse
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from environment import Environment
from task import Task
from dataset import multi_task_Dataset,Dataset
import transporter
import gnn_agent
import cv2 as cv
import utils

MAX_ORDER = 3
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

object_param ={"rope":{"action_dim":24,"env_type":0},
             "ring":{"action_dim":32,"env_type":1},
             "cloth":{"action_dim":16,"env_type":2}}

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    #simulation params
    parser.add_argument('-disp',default=False,type=bool)  
    parser.add_argument('-remote',default=0,type=int)  
    parser.add_argument('-max_action',default=20,type=int)

    parser.add_argument('-task_type',default="multi_task",type=str) 
    parser.add_argument('--agent',  default='local_gnn',type=str)
    #local_gnn, global_gnn, transporter, transporter-goal
 
    #transporter params
    parser.add_argument('--num_rots',  default=1, type=int)
    parser.add_argument('--subsamp_g',     action='store_true')
    parser.add_argument('--crop_bef_q',     default=0, type=int, help='CoRL paper used 1')
    
    #dataset params
    parser.add_argument('-data_dir', default='/apdcephfs/share_1150325/francisdeng/deformable_dataset')
    parser.add_argument('-eposide_num',default=1100,type=int)   #10 100 1000
    parser.add_argument('-train_sample_num',default=1000,type=int)    #must less than eposide_num 1000
    parser.add_argument('-test_sample_num',default=20,type=int) #100
    parser.add_argument('-num_train_iters',default=2000000,type=int) # 1000*20 1000*20*100
    parser.add_argument('-train_run',default=0,type=int) #random index
    parser.add_argument('-mode',default="train",type=str) # train test
    # only useful for gnn agent
    parser.add_argument('-learning_rate', default=0.001,type=float) #rope_line,rope_l,rope_v:0.001,rope_n:0.0001
    parser.add_argument('-batch_size',default=128,type=int) 

    args = parser.parse_args()
    args.time_id = time.strftime("%m_%d_%H:%M")

    action_dim = 32
    #dataset initialization
    dataset = multi_task_Dataset(args.data_dir,args.agent,action_dim)
    # Evaluate on increasing orders of magnitude of demonstrations.
    test_dir = os.path.join(args.data_dir,'test_results')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    check_point_dir = os.path.join(args.data_dir,'checkpoints')
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    log_dir = os.path.join(args.data_dir,'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set the beginning of the agent name.
    train_run = args.train_run
    name = f'{args.task_type}-{args.agent}-{args.train_sample_num}-{train_run}'

    if 'transporter' in args.agent:
        # GPU devices
        physical_devices = tf.config.list_physical_devices('GPU') 
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        # Set up tensorboard logger.
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(log_dir, args.agent, args.task_type, current_time, 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)           
        # Initialize agent and limit random dataset sampling to fixed set.
        tf.random.set_seed(train_run)
    
    if args.agent == 'transporter':
        agent = transporter.OriginalTransporterAgent(name,
                                            args.task_type,
                                            num_rotations=args.num_rots,
                                            crop_bef_q=(args.crop_bef_q == 1))
    elif 'transporter-goal' in args.agent:
        agent = transporter.GoalTransporterAgent(name,
                                            args.task_type,
                                            num_rotations=args.num_rots)
    elif 'gnn' in args.agent:
        agent = gnn_agent.GNN_Agent(name,
                                    args.task_type,
                                    action_dim,
                                    args.learning_rate,
                                    args.batch_size,
                                    args.agent)
    
    agent.models_dir = os.path.join(check_point_dir, name)  
    if not os.path.exists(agent.models_dir):
        os.makedirs(agent.models_dir)   
    # Limit random data sampling to fixed set.
    
    np.random.seed(train_run) 
    total_data_list = list(range(args.eposide_num))
     
    choose_episodes = np.random.choice(total_data_list, args.train_sample_num+args.test_sample_num, False)
    train_episodes = choose_episodes[:args.train_sample_num]
    test_episodes = choose_episodes[args.train_sample_num:]
            
    if args.mode == "train":
        dataset.set(train_episodes)
        train_interval = int((args.num_train_iters)/10)                  
        while agent.total_iter < args.num_train_iters:
            # Train agent.
            if "transporter" in args.agent:                                    
                agent.train(dataset, num_iter=train_interval, writer=train_summary_writer)
            elif "gnn" in args.agent:
                agent.train(dataset,num_iter=train_interval)
    
    elif args.mode == "test":
        performance = [ ]  
        dataset.set(test_episodes)
        test_num_iter_list = []
        for fname in sorted(os.listdir(agent.models_dir)):
            if 'attention' in fname:
                train_num_iter = int(fname[15:-3])
                test_num_iter_list.append(train_num_iter)
            elif 'gnn' in fname:
                train_num_iter = int(fname[fname.find('-') + 1:-3])
                test_num_iter_list.append(train_num_iter)       
        test_num_iter_list.sort()
        if len(test_num_iter_list) > 0:
            test_num_iter = test_num_iter_list[len(test_num_iter_list)-1]
            print("test iter:",test_num_iter)
            if 'attention' in fname:               
                tf.keras.backend.set_learning_phase(0)    
                agent.load(test_num_iter)
            elif 'gnn' in fname:
                agent.agent.eval()
                agent.load(test_num_iter)
            
            #"rope_line","rope_l","rope_n","rope_v","ring_square","cloth_flatten",
            task_types = ["cloth_fold","cloth_fold_a"]
            
            for choose_task_type in task_types:
                object_type = choose_task_type[:(choose_task_type.find('_'))]     
                env_type = object_param[object_type]['env_type']
                choose_action_dim = object_param[object_type]['action_dim']
                
                simu_env = Environment(disp=args.disp,env_type=env_type)
                simu_env.reset()
                simu_env.start()

                dataset_dir = os.path.join(args.data_dir,choose_task_type) 
                choose_dataset = Dataset(dataset_dir,args.agent,args.remote)

                done_list = []
                act_num_list = []          
                for iepisode in test_episodes:
                    is_episode_sample = np.int32(choose_dataset.episode_id) == iepisode
                    episode_samples = np.argwhere(is_episode_sample).squeeze().reshape(-1)
                    print("test task:",choose_task_type+"_"+str(iepisode))
                    
                    def load(iepisode, field):
                        field_path = os.path.join(choose_dataset.path, field)
                        ep_fname = f'{iepisode:06d}-{len(episode_samples)}.pkl'
                        return pickle.load(open(os.path.join(field_path, ep_fname), 'rb'))

                    goal = {}
                    goal_index = len(episode_samples)
                    goal['color'] = load(iepisode, 'multi_rgb')[goal_index]
                    goal['depth'] = load(iepisode, 'depth')[goal_index] 
                    goal_image =   load(iepisode, 'tp_rgb')[goal_index]  
                    goal_state =  load(iepisode, 'next')[goal_index-1][choose_action_dim:]
            
                    if 'cloth' in choose_task_type:
                        goal_angle  = utils.get_transform_angle(goal_state,choose_task_type)
                        simu_env.remove_deform()
                        simu_env.reset(give_angle=goal_angle)
            
                    task_kwargs = {'task_type':choose_task_type,
                                    "normalize_length":simu_env.normalize_length,
                                    "target_state":goal_state,
                                    "transform_angle":simu_env.index_angle}
                    new_eposide_task = Task(**task_kwargs)
                    
                    if "fold" not in choose_task_type:
                        simu_env.random_initialize()
                    
                    task_act_num= 0
                    done = False
                    for j in range(args.max_action):
                        obs_state =simu_env.get_current_state()   #get observation
                        new_eposide_task.get_state_space(obs_state) 
                        _,done = new_eposide_task.get_reward(obs_state)
                        if done:
                            break
                        else:
                            front_image,front_depth = simu_env.render(simu_env.camera_config[0])
                            left_image,left_depth = simu_env.render(simu_env.camera_config[1])
                            right_image,right_depth = simu_env.render(simu_env.camera_config[2]) 
                            tp_image,tp_depth = simu_env.render(simu_env.camera_config_up) 
                            current_state,current_matrix = new_eposide_task.get_state_space(obs_state)  
                                                         
                            if "transporter" in args.agent:
                                obs = {}
                                obs['color'] = np.uint8([front_image,left_image,right_image])
                                obs['depth'] = np.float32([front_depth,left_depth,right_depth])
                            elif "gnn" in args.agent:
                                obs = current_state
                                obs_len = int(len(obs)/2)  #padding
                                expand_num = action_dim - obs_len
                                mask = [1]*len(obs)
                                if expand_num>0:
                                    for i in range(expand_num):
                                        obs = np.insert(obs,obs_len+i,[0,0],axis=0)
                                        mask.insert(obs_len+i,0)
                                        obs = np.insert(obs,2*(obs_len+i)+1,[0,0],axis=0)
                                        mask.insert(2*(obs_len+i)+1,0)
                                assert len(obs) == action_dim*2,len(obs)
                            
                            if "goal" in args.agent:
                                act,extra =  agent.act(obs,goal,debug_imgs=True)
                            
                            elif "transporter" in args.agent:
                                act,extra= agent.act(obs,debug_imgs=True)
                                print("inference time:",extra["inference_time"])
                            
                            elif "gnn" in args.agent:
                                act= agent.act(obs,mask)

                            if "transporter" in args.agent:
                                pick_position = [act['params']['pose0'][0],[0,0,0,1]]
                                place_position = [act['params']['pose1'][0],[0,0,0,1]]
                                simu_env.pick_place(pick_position,place_position) 
                            elif "gnn" in args.agent:
                                simu_env.step(act['params']['pose0'],act['params']['pose1'])
                            
                            task_act_num += 1
                            
                    print("task finish:",done)
                    done_list.append(done) 
                    act_num_list.append(task_act_num)
                    simu_env.remove_deform()
                    simu_env.reset()
                        
                test_dic = {}
                test_dic["successful"] = done_list
                test_dic["act_num"] = act_num_list
                test_dic["iter"] = test_num_iter
                test_dic["task_type"] = choose_task_type
                performance.append(test_dic)
                simu_env.pause()
                simu_env.stop()
                del simu_env

        pickle_fname = os.path.join(test_dir, f'{name}.pkl')
        pickle.dump(performance, open(pickle_fname, 'wb'))
        print("dump:",name)



        


