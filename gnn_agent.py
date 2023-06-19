"""
    -- This agent is utilizing the CNN+8 U-net layer as for the Deep part (8 U-net with 8 different direction and 2 different depth)
    -- and the input state is 4-channel screen (with different history and memory)
"""
import os
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch
from termcolor import colored
from torch.autograd import Variable
from models import Local_Transformer, Global_Transformer,KeyPointer,GNN_MAX, GNN_MEAN, GNN_ADD, GNN_LIN
import cv2 as cv
#from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time
import utils



class GNN_Agent(object):
    def __init__(self,name, task, action_dim,learning_rate,batch_size,model_type):
        self.name = name
        self.task = task
        self.agent_multi = ("multi" in self.task)
        self.total_iter = 0
        self.action_dim = action_dim #pick and place action space
        self.models_dir = os.path.join('checkpoints', self.name)
        #self.keypointer = KeyPointer()
        self.batch_size = batch_size

        if model_type == "global_gnn":
            self.agent = Global_Transformer(action_dim=2*self.action_dim)
        elif model_type == "local_gnn":
            self.agent = Local_Transformer(action_dim=2*self.action_dim)
        elif model_type == "gnn_max":
            self.agent = GNN_MAX(in_channels=2, hidden_channels=64, out_channels=4)
        elif model_type == "gnn_mean":
            self.agent = GNN_MEAN(in_channels=2, hidden_channels=64, out_channels=4)
        elif model_type == "gnn_add":
            self.agent = GNN_ADD(in_channels=2, hidden_channels=64, out_channels=4)
        elif model_type == "gnn_lin":
            self.agent = GNN_LIN(in_channels=2, hidden_channels=64, out_channels=4)
        
        if (model_type == "global_gnn") or (model_type == "local_gnn"):
            self.mini_batch = False
        else:
            self.mini_batch = True
            self.edge_index_np = utils.get_edge(self.action_dim,self.batch_size)
            self.edge_index = Variable(torch.LongTensor(np.array(self.edge_index_np)).cuda())
        '''
        self.keypointer.cuda()
        tensor = (Variable(torch.rand(1, 3, 240, 320)).cuda(),)
        flops = FlopCountAnalysis(self.keypointer, tensor)
        tensor_1 = Variable(torch.randn(1, 64, 2)).cuda()
        self.agent.cuda()
        flops_1 = FlopCountAnalysis(self.agent, tensor_1)
        
        print("FLOPs: ", flops.total())
        print("param:",parameter_count_table(self.keypointer))
        print("FLOPs1: ", flops_1.total())
        print("param1:",parameter_count_table(self.agent))
        exit(0)
        '''

        print(colored("Build agent", color='red', attrs=['bold']))


        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.agent.parameters()), lr= self.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()
        

    def train(self, dataset, num_iter):
        train_iter = 0
        self.agent.train()
        self.agent.cuda()
        self.loss_list = []
        while train_iter < num_iter:
            current_states = []
            acts = []
            if self.agent_multi:
                masks = []
            for i in range(self.batch_size):              
                if self.agent_multi:
                    current_state, act, mask= dataset.random_sample()
                    masks.append(mask)
                else:
                     current_state,act= dataset.random_sample()
                current_states.append(current_state)
                acts.append(act)
            
            current_states_var =  Variable(torch.FloatTensor(np.array(current_states)).cuda())
            acts_var = Variable(torch.LongTensor(np.array(acts)).cuda()) #batch * 2
            if self.agent_multi:
                masks_var = Variable(torch.IntTensor(np.array(masks)).cuda())
                predict_acts = self.agent.forward(current_states_var,masks_var)
            elif self.mini_batch:
                predict_acts = self.agent.forward(current_states_var,self.edge_index,self.batch_size)
            else:
                predict_acts = self.agent.forward(current_states_var) #batch * (2*action_dim)
            
            predict_pick = predict_acts[:,:self.action_dim]
            predict_place = predict_acts[:,self.action_dim:]
            truth_pick = acts_var[:,0]
            truth_place = acts_var[:,1]
           
            loss_pick = self.loss_func(predict_pick+1e-8,truth_pick)
            loss_place = self.loss_func(predict_place+1e-8,truth_place) #avoid mask err
           
            loss = 0.5*loss_pick + 0.5*loss_place
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_iter += self.batch_size
            self.total_iter += self.batch_size            
            print(f'Train Iter: {self.total_iter} Loss: {loss_pick.item():.4f} {loss_place.item():.4f}')
            self.loss_list.append([loss_pick.item(),loss_place.item()])
        
        self.save()

    def act(self,c_state,mask=None):
        self.agent.eval()
        self.agent.cuda()
        #torch.cuda.synchronize()

        if mask is not None:
            mask_var = Variable(torch.IntTensor(np.array(mask)).cuda())
            mask_var = mask_var.unsqueeze(0)
        else:
            mask_var= None
        c_state_var = Variable(torch.FloatTensor(np.array(c_state)).cuda())
        c_state_var = c_state_var.unsqueeze(0)
        #start = time.clock() 
        if self.mini_batch:
            edge_index_np_1 = utils.get_edge(self.action_dim,1)
            edge_index_1 = Variable(torch.LongTensor(np.array(edge_index_np_1)).cuda())
            action_q_value= self.agent.forward(c_state_var,edge_index_1,1)
        else:
            action_q_value= self.agent.forward(c_state_var,mask_var)
        #torch.cuda.synchronize()
        #end = time.clock()
        #print("inference time:",end-start)
        
        pick_q_value = action_q_value[:,:self.action_dim]
        place_q_value = action_q_value[:,self.action_dim:]
        pick_reshape = pick_q_value.view(1,-1)
        place_reshape = place_q_value.view(1,-1)
        pick_location = torch.max(pick_reshape,1)[1].cpu().data.numpy()[0]
        place_location = torch.max(place_reshape,1)[1].cpu().data.numpy()[0]
        pick_pixel = c_state[pick_location]
        place_pixel = c_state[place_location+self.action_dim]

        act_dic  = {}
        act_dic['primitive'] = 'pick_place'
        params = {'pose0':pick_pixel,'pose1':place_pixel}
        act_dic['params'] = params
        
        return act_dic

    def load(self, num_iter):
        """Load pre-trained models."""
        checkpoint_fname = 'gnn-%d.pt' % num_iter
        checkpoint_fname = os.path.join(self.models_dir, checkpoint_fname)
        checkpoint = torch.load(checkpoint_fname)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        checkpoint_fname = 'gnn-%d.pt' % self.total_iter
        checkpoint_fname = os.path.join(self.models_dir, checkpoint_fname)
        torch.save({
                    'item': self.total_iter,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss_list,
                    }, checkpoint_fname)



  




