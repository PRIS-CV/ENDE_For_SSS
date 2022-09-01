import torch
import torch.nn as nn
from blocks.conv import *
import torch.nn.functional as F
import numpy as np
from blocks.basic import MLPLinear, MultiSeq
import torch_geometric.nn as tgnn
# import torchsnooper


def init_net(opt, is_Creativity=False):
    if opt.net_name == 'ENDE_GNN':
        net = ENDE_GNN(opt)
    else:
        raise NotImplementedError('net {} is not implemented. Please check.\n'.format(opt.net_name))
    
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    return net


class ENDE_GNN_feature(nn.Module):
    def __init__(self, opt):
        super(ENDE_GNN_feature, self).__init__()
        self.opt = opt
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.pool_channels = opt.pool_channels

        ####################### point feature #######################
        opt.kernel_size = opt.local_k
        opt.dilation = opt.local_dilation
        opt.stochastic = opt.local_stochastic
        opt.epsilon = opt.local_epsilon
        dilations = [1, 4, 8] + [opt.local_dilation] * (self.n_blocks-2)   
        
        # head
        if self.opt.local_adj_type == 'static':
            self.local_head = GraphConv(opt.in_feature, self.channels, opt)
        else:
            self.local_head = DynConv(opt.in_feature, self.channels, dilations[0], opt) 
        
        # local backbone
        self.local_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.local_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])

        ####################### stroke & sketch feature #######################
        opt.kernel_size = opt.global_k
        opt.dilation = opt.global_dilation
        opt.stochastic = opt.global_stochastic
        opt.epsilon = opt.global_epsilon
        dilations = [1, opt.global_dilation//4, opt.global_dilation//2] + [opt.global_dilation] * (self.n_blocks-2)   
        
        # head
        if self.opt.global_adj_type == 'static':
            self.global_head = GraphConv(opt.in_feature, self.channels, opt)
        else:
            self.global_head = DynConv(opt.in_feature, self.channels, dilations[0], opt)    
        
        # global backbone
        self.global_backbone = MultiSeq(*[ResGcnBlock(self.channels, opt.global_adj_type, dilations[i+1], opt) for i in range(self.n_blocks)])
        
    # @torchsnooper.snoop()
    def forward(self, x, edge_index, data):
        """
        x: (BxN) x F
        """
        BN = x.shape[0]
        ####################### local line #######################
        x_l = self.local_head(x, edge_index, data).unsqueeze(-1)
        x_l = torch.cat((x_l, x_l), 2)
        x_l = self.local_backbone(x_l, edge_index, data)[0][:,:,1:].contiguous().view(BN, -1)

        ####################### global line #######################
        x_g = self.global_head(x, edge_index, data).unsqueeze(-1)
        x_g = torch.cat((x_g, x_g), 2)
        x_g = self.global_backbone(x_g, edge_index, data)[0][:,:,1:].contiguous().view(BN, -1)

        return x_l, x_g


class ENDE_GNN_decoder(nn.Module):
    def __init__(self, opt):
        super(ENDE_GNN_decoder, self).__init__()
        self.opt = opt
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.pool_channels = opt.pool_channels

        ####################### point feature #######################
        opt.kernel_size = opt.local_k
        opt.dilation = opt.local_dilation
        opt.stochastic = opt.local_stochastic
        opt.epsilon = opt.local_epsilon
        dilations = [1, 4, 8] + [opt.local_dilation] * (self.n_blocks-2)   

        ####################### stroke & sketch feature #######################
        opt.kernel_size = opt.global_k
        opt.dilation = opt.global_dilation
        opt.stochastic = opt.global_stochastic
        opt.epsilon = opt.global_epsilon
        dilations = [1, opt.global_dilation//4, opt.global_dilation//2] + [opt.global_dilation] * (self.n_blocks-2)   

        # decoder
        self.fc_hc = nn.Linear(self.channels*(self.n_blocks+1)*2, 2*opt.dec_hidden_size)
        self.lstm = nn.LSTM(self.channels*(self.n_blocks+1), opt.dec_hidden_size, dropout=opt.dropout)
        self.fc_params = nn.Sequential(
            nn.Linear(opt.dec_hidden_size,opt.dec_hidden_size//4),
            nn.BatchNorm1d(opt.dec_hidden_size//4),
            nn.ReLU(inplace=True),
            nn.Linear(opt.dec_hidden_size//4,2),
        )
        
    # @torchsnooper.snoop()
    def forward(self, x, edge_index, data):
        """
        x: (BxN) x F
        """
        batch_size = x.size(0)//self.opt.points_num
        x_max = tgnn.global_max_pool(x, data['batch'])
        x_avg = tgnn.global_mean_pool(x, data['batch'])
        hidden_cell = torch.cat([x_max,x_avg],dim=1)
        hidden,cell = torch.split(F.tanh(self.fc_hc(hidden_cell)),self.opt.dec_hidden_size,1)
        hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs,(hidden,cell) = self.lstm(x.view(batch_size,self.opt.points_num,-1).permute(1, 0, 2), hidden_cell)
        params = self.fc_params(outputs.permute(1, 0, 2).contiguous().view(-1, self.opt.dec_hidden_size))
        
        return params


class ENDE_GNN_segmenter(nn.Module):
    def __init__(self, opt):
        super(ENDE_GNN_segmenter, self).__init__()
        self.opt = opt
        self.n_blocks = opt.n_blocks
        self.channels = opt.channels
        self.pool_channels = opt.pool_channels

        ####################### point feature #######################
        opt.kernel_size = opt.local_k
        opt.dilation = opt.local_dilation
        opt.stochastic = opt.local_stochastic
        opt.epsilon = opt.local_epsilon
        dilations = [1, 4, 8] + [opt.local_dilation] * (self.n_blocks-2)   

        ####################### stroke & sketch feature #######################
        opt.kernel_size = opt.global_k
        opt.dilation = opt.global_dilation
        opt.stochastic = opt.global_stochastic
        opt.epsilon = opt.global_epsilon
        dilations = [1, opt.global_dilation//4, opt.global_dilation//2] + [opt.global_dilation] * (self.n_blocks-2)   
        
        if opt.fusion_type == 'mix':
            self.pool = MixPool(opt.channels*(opt.n_blocks+1), opt.pool_channels // 2)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        elif opt.fusion_type == 'max':
            self.pool = MaxPool(opt.channels*(opt.n_blocks+1), opt.pool_channels)
            mlpSegment = [self.channels*(self.n_blocks+1) + self.pool_channels] + opt.mlp_segment
        else:
            raise NotImplementedError('fusion_type {} is not implemented. Please check.\n'.format(opt.fusion_type))
        self.segment = MultiSeq(*[MLPLinear(mlpSegment, norm_type='batch', act_type='relu'),
                                  MLPLinear([mlpSegment[-1], opt.out_segment], norm_type='batch', act_type=None)])
        # softmax        
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
    # @torchsnooper.snoop()
    def forward(self, x_l, x_g, edge_index, data):
        """
        x: (BxN) x F
        """
        x_g = self.pool(x_g, data['stroke_idx'], data['batch'])
        ####################### cat #######################
        x = torch.cat([x_l, x_g], dim=1)

        ####################### segment #######################
        x = self.segment(x)

        return self.LogSoftmax(x)


class ENDE_GNN(nn.Module):
    def __init__(self, opt):
        super(ENDE_GNN, self).__init__()
        self.opt = opt
        self.backbone = ENDE_GNN_feature(opt)
        self.decoder = ENDE_GNN_decoder(opt)
        self.segmenter = ENDE_GNN_segmenter(opt)
        self.mseloss = nn.MSELoss()
        
    # @torchsnooper.snoop()
    def forward(self, x, edge_index, data):
        """
        x: (BxN) x F
        """
        x_l, x_g = self.backbone(x, edge_index, data)

        ####################### decoder #######################
        if self.training:
            batch_size = x_g.size(0)//self.opt.points_num
            # identify mixture params:
            params = self.decoder(x_g, edge_index, data)
            
            # prepare targets:
            rela_pos = data['pos'].detach().view(batch_size,self.opt.points_num,self.opt.in_feature)
            rela_pos = torch.cat((rela_pos[:,0,:].unsqueeze(1), rela_pos[:,1:,:]-rela_pos[:,:-1,:]), dim=1)
            self.r_cost = self.mseloss(params, rela_pos.view(-1,self.opt.in_feature))

            ####################### segment #######################
            x = self.segmenter(x_l, x_g, edge_index, data)

            return x, self.r_cost
        
        else:
            ####################### segment #######################
            x = self.segmenter(x_l, x_g, edge_index, data)

            return x


