import os
import torch
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, adjusted_rand_score
import net
import utils

class SketchModel:
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.dataset, opt.class_name, opt.timestamp)
        self.pretrain_dir = os.path.join(opt.checkpoints_dir, opt.dataset, opt.class_name, opt.pretrain)
        self.loss = None

        self.net_name = opt.net_name
        self.net = net.init_net(opt)
        self.net.train(self.is_train)
        self.loss_func = torch.nn.NLLLoss().to(self.device)

        if self.is_train:
            self.opt_theta = torch.optim.Adam(self.net.module.backbone.parameters(), 
                                            lr=opt.lr, 
                                            betas=(opt.beta1, 0.999),
                                            weight_decay=opt.weight_decay)
            self.opt_phi = torch.optim.Adam(self.net.module.segmenter.parameters(), 
                                            lr=opt.lr, 
                                            betas=(opt.beta1, 0.999),
                                            weight_decay=opt.weight_decay)
            self.opt_pi = torch.optim.Adam(self.net.module.decoder.parameters(), 
                                            lr=opt.lr, 
                                            betas=(opt.beta1, 0.999),
                                            weight_decay=opt.weight_decay)
            self.scheduler_theta = utils.get_scheduler(self.opt_theta, opt)
            self.scheduler_phi = utils.get_scheduler(self.opt_phi, opt)
            self.scheduler_pi = utils.get_scheduler(self.opt_pi, opt)
        
        if not self.is_train: #or opt.continue_train:
            self.load_network(opt.which_epoch, mode='test')
            
        if self.is_train and opt.pretrain != '-':
            self.load_network(opt.which_epoch, mode='pretrain')
    
    def forward(self, x, edge_index, data):
        out = self.net(x, edge_index, data)
        return out

    def backward(self, out, label):
        """
        out: (B*N, C)
        label: (B*N, )
        """
        self.loss = self.loss_func(out, label)
        self.loss.backward()

    def step(self, data):
        """
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x

        self.opt_theta.zero_grad()
        self.opt_phi.zero_grad()
        self.opt_pi.zero_grad()
        out, r_cost = self.forward(x, edge_index, stroke_data)
        r_cost.backward()
        self.opt_pi.step()

        self.opt_theta.zero_grad()
        self.opt_phi.zero_grad()
        self.opt_pi.zero_grad()
        out, r_cost = self.forward(x, edge_index, stroke_data)
        self.loss = self.loss_func(out, label) + self.opt.r_weight * r_cost
        self.loss.backward()
        self.opt_theta.step()
        self.opt_phi.step()
    
    def test(self, data, if_eval=False):
        """
        x: (B*N, F)
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x

        out = self.forward(x, edge_index, stroke_data)
        predict = torch.argmax(out, dim=1).cpu().numpy()

        self.loss = self.loss_func(out, label)

        return self.loss, predict
        
    
    def print_detail(self):
        print(self.net)

    def update_learning_rate(self):
        """
        update learning rate (called once every epoch)
        """
        self.scheduler_theta.step()
        self.scheduler_phi.step()
        self.scheduler_pi.step()
        lr = self.opt_theta.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_network(self, epoch):
        """
        save model to disk
        """
        path = os.path.join(self.save_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), path)
    
    def load_network(self, epoch, mode='test'):
        """
        load model from disk
        """
        path = os.path.join(self.save_dir if mode =='test' else self.pretrain_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from {}'.format(path))
        state_dict = torch.load(path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)
    
