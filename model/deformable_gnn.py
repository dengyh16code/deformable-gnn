

from __future__ import division
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Variable
import numpy as np
import utils


# ----------- point detect -----------

class KeyPointer(nn.Module):
    def __init__(self,               
                num_keypoints=16,
                filters = [16, 16, 32, 32, 64, 64],
                kernel_sizes = [7, 3, 3, 3, 3, 3],
                strides = [1, 1, 2, 1, 2, 1]):  # ninp, dropout 1,1,2,1,2,1
        super(KeyPointer, self).__init__()
        self.conv_1 = Conv_Block(3,16,7,1)
        self.conv_2 = Conv_Block(16,16,3,1)
        self.conv_3 = Conv_Block(16,32,3,2)
        self.conv_4 = Conv_Block(32,32,3,1)
        self.conv_5 = Conv_Block(32,64,3,2)
        self.conv_6 = Conv_Block(64,64,3,1)
        self.conv_finial = nn.Conv2d(64,num_keypoints,1,1)
    def forward(self,input_image):
        _,_,H,W = input_image.shape
        x = self.conv_1(input_image)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        f_map = self.conv_finial(x)
        f_map_np = f_map.cpu().data.numpy()[0]
        print(f_map_np.shape)
        np.save("data_dr.npy",f_map_np)
        kp_corrd = utils.get_coord(f_map)
        gmap = utils.get_gaussian_maps(kp_corrd,map_size=[H,W])
        return gmap


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv_Block, self).__init__()			
        self.main_module = nn.Sequential(
            Conv2d_same(in_channels,out_channels,kernel_size,stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):        
        return self.main_module(x)


class ConvNd_new(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(ConvNd_new, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d_same(ConvNd_new): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_same, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)



# ----------- act -----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=40):  # ninp, dropout
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 40 * 128
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 40 * 1 * 64
        self.register_buffer('pe', pe)

    def forward(self, x):
        c = self.pe[:x.size()[0], :]
        x = x + self.pe[:x.size()[0], :]        # torch.Size([40, 1, 64])
        return self.dropout(x)

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len =40):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)



class Global_Transformer(nn.Module):
        def __init__(self,
                 input_dim=2,
                 tranfromer_input_dim=64,
                 tranfromer_feedforward_dim=64,
                 position_encoder_dropout=0.2,
                 tranfromer_n_head=16,
                 encoder_layers = 12,
                 tranfromer_encoder_dropout=0,
                 action_dim = 32,
                 ):

            super(Global_Transformer, self).__init__()
            self.action_dim = action_dim
            self.tranfromer_input_dim = tranfromer_input_dim
            self.linear_layer = nn.Linear(input_dim ,tranfromer_input_dim,bias=False)
            #self.pos_encoder_src = PositionalEncoding(d_model=tranfromer_input_dim)
            
            self.pos_encoder = PositionalEncoding(d_model=tranfromer_input_dim,max_len=action_dim)
            self.transformer_layer_all = nn.TransformerEncoderLayer(d_model=self.tranfromer_input_dim, nhead=tranfromer_n_head,dim_feedforward=tranfromer_feedforward_dim)
            encoder_norm_all = nn.LayerNorm(self.tranfromer_input_dim)
            self.transformer_encoder_all = nn.TransformerEncoder(self.transformer_layer_all, encoder_layers, encoder_norm_all)
                             
            self.decoder_layer = nn.Linear(self.tranfromer_input_dim, 1)

        def forward(self, src, mask=None):

            src = F.relu(self.linear_layer(src))
            src = src.view(-1,self.action_dim,self.tranfromer_input_dim)
            src = src.transpose(0,1)
            src = self.pos_encoder(src)
            if mask is not None:
                mask_a = (mask==0)
                output = self.transformer_encoder_all(src,src_key_padding_mask=mask_a)  #global attention
            else:
                output = self.transformer_encoder_all(src)  #global attention   
            output = output.view(-1,self.tranfromer_input_dim)  
            output = self.decoder_layer(output)
            output = output.view(self.action_dim,-1)  
            output = output.transpose(0,1) #batch*ation_dim
            output = F.relu(output)    
            return output

class Local_Transformer(nn.Module):
        def __init__(self,
                 input_dim=2,
                 tranfromer_input_dim=64,
                 tranfromer_feedforward_dim=64,
                 position_encoder_dropout=0.2,
                 tranfromer_n_head=16,
                 encoder_layers = 6,
                 tranfromer_encoder_dropout=0,
                 action_dim = 32,
                 ):

            super(Local_Transformer, self).__init__()
            self.action_dim = action_dim
            self.tranfromer_input_dim = tranfromer_input_dim
            self.linear_layer = nn.Linear(input_dim ,tranfromer_input_dim,bias=False)
            
            self.pos_encoder_c = PositionalEncoding(d_model=tranfromer_input_dim,max_len=action_dim//2)
            self.transformer_layer_c = nn.TransformerEncoderLayer(d_model=self.tranfromer_input_dim, nhead=tranfromer_n_head,dim_feedforward=tranfromer_feedforward_dim)
            encoder_norm_c = nn.LayerNorm(self.tranfromer_input_dim)
            self.transformer_encoder_c = nn.TransformerEncoder(self.transformer_layer_c, encoder_layers, encoder_norm_c)

            self.pos_encoder_t = PositionalEncoding(d_model=tranfromer_input_dim,max_len=action_dim//2)
            self.transformer_layer_t = nn.TransformerEncoderLayer(d_model=self.tranfromer_input_dim, nhead=tranfromer_n_head,dim_feedforward=tranfromer_feedforward_dim)
            encoder_norm_t = nn.LayerNorm(self.tranfromer_input_dim)
            self.transformer_encoder_t = nn.TransformerEncoder(self.transformer_layer_t, encoder_layers, encoder_norm_t)
            
            self.pos_encoder_a = PositionalEncoding(d_model=tranfromer_input_dim,max_len=action_dim)
            self.transformer_layer_all = nn.TransformerEncoderLayer(d_model=self.tranfromer_input_dim, nhead=tranfromer_n_head,dim_feedforward=tranfromer_feedforward_dim)
            encoder_norm_all = nn.LayerNorm(self.tranfromer_input_dim)
            self.transformer_encoder_all = nn.TransformerEncoder(self.transformer_layer_all, encoder_layers, encoder_norm_all)
                     
            self.decoder_layer = nn.Linear(self.tranfromer_input_dim, 1)

        def forward(self, src,mask=None):

            src = F.relu(self.linear_layer(src))
            src = src.view(-1,self.action_dim,self.tranfromer_input_dim)
            src = src.transpose(0,1) #sequence length * batch * dim
            
            src_c = src[:self.action_dim//2,:,:]
            src_t = src[self.action_dim//2:,:,:]
           
            src_c = self.pos_encoder_c(src_c)
            src_t = self.pos_encoder_t(src_t)
            
            if mask is not None:
                mask_c = (mask[:,:self.action_dim//2]==0)
                output_c = self.transformer_encoder_c(src_c,src_key_padding_mask=mask_c)
            else:
                output_c = self.transformer_encoder_c(src_c)

            if mask is not None:
                mask_t = (mask[:,self.action_dim//2:]==0)
                output_t = self.transformer_encoder_t(src_t,src_key_padding_mask=mask_t)
            else:
                output_t = self.transformer_encoder_t(src_t)  #local attention

            output = torch.cat([output_c,output_t],0) #ation_dim*batch*64
            output = self.pos_encoder_a(output)
        
            mask_np_c = np.zeros((self.action_dim,self.action_dim))
            mask_np_c[:self.action_dim//2,self.action_dim//2:] = 1
            mask_np_c[self.action_dim//2:,:self.action_dim//2] = 1
            mask_var_c = Variable(torch.IntTensor(mask_np_c).cuda())
            mask_var_c = mask_var_c.float().masked_fill(mask_var_c==0,float("-inf")).masked_fill(mask_var_c==1,float("0.0"))
            
            if mask is not None:
                mask_a = (mask==0)
                output = self.transformer_encoder_all(output, mask = mask_var_c,src_key_padding_mask=mask_a)  #cross attention
            else:
                output = self.transformer_encoder_all(output, mask =mask_var_c)  #cross attention

            output = output.view(-1,self.tranfromer_input_dim)    
            output = self.decoder_layer(output)

            output = output.view(self.action_dim,-1)  
            output = output.transpose(0,1) #batch*ation_dim
            output = F.relu(output)    
            return output           


