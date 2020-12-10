import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from .distangle_model import DistangleModel
from PIL import ImageOps,Image
from .loss_function import smooth_loss

class SIDPAMIWModel(DistangleModel):
    def name(self):
        return 'Shadow Image Decomposition model PAMI19 weighted boundary loss'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.add_argument('--wdataroot',default='None',  help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--use_our_mask', action='store_true')
        parser.add_argument('--mask_train',type=str,default=None)
        parser.add_argument('--mask_test',type=str,default=None)
        parser.add_argument('--lambda_bd',type=float,default=100)
        parser.add_argument('--lambda_res',type=float,default=100)
        parser.add_argument('--lambda_param',type=float,default=100)
#        parser.add_argument('--lambda_smooth',type=float,default=100)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G_param','alpha','rescontruction','bd','smooth']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['input_img', 'alpha_pred','out','final','masked_fake']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G','M']
        # load/define networks
        opt.output_nc= 3 
        if self.opt.netG =='vgg':
            self.netG = networks.define_vgg(4,6, gpu_ids = self.gpu_ids)
        if self.opt.netG =='RESNEXT':
            self.netG = networks.define_G(4, 6, opt.ngf, 'RESNEXT', opt.norm,
                                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netM = networks.define_G(7, 3, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
        self.netG.to(self.device)
        self.netM.to(self.device)
        print(self.netG)
        print(self.netM) 
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # initialize optimizers
            self.optimizers = []
            #self.optimizer_G = torch.optim.SGD(self.netG.parameters(),
            #                                    lr=0.002, momentum=0.9)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-5)
            self.optimizer_M = torch.optim.Adam(self.netM.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-5)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_M)
   
    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.penumbra = input['penumbra'].to(self.device).type(torch.float)
        self.penumbra = (self.penumbra>0).type(torch.float)
        self.penumbra =  self.penumbra.expand(self.input_img.shape)
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        if 'isreal' in input:
            self.isreal = input['isreal']




    
    def forward(self):
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        self.shadow_param_pred = torch.squeeze(self.netG(inputG))

        n = self.shadow_param_pred.shape[0]
        #m = self.shadow_param_pred.shape[1]
        w = inputG.shape[2]
        h = inputG.shape[3]
        
        #self.shadow_param_pred = torch.mean(self.shadow_param_pred.view([n,m,-1]),dim=2)
        add = self.shadow_param_pred[:,[0,2,4]] / 2
        mul = self.shadow_param_pred[:,[1,3,5]] + 2
        
        #mul = (mul +2) * 5/3          
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        
        
        addgt = self.shadow_param[:,[0,2,4]]
        mulgt = self.shadow_param[:,[1,3,5]]

        addgt = addgt.view(n,3,1,1).expand((n,3,w,h))
        mulgt = mulgt.view(n,3,1,1).expand((n,3,w,h))
        
        self.litgt = self.input_img.clone()/2+0.5
        self.lit = self.input_img.clone()/2+0.5
        self.lit = self.lit*mul + add
        self.litgt = (self.litgt*mulgt+addgt)*2-1
        
        self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
        self.out = self.out*2-1

        

        
        #self.outgt = (self.input_img/2+0.5)*(1-self.alpha_3d) + self.lit*(self.alpha_3d)
        #self.outgt = self.outgt*2-1
        
        #lit.detach if no final loss for paramnet 
        inputM = torch.cat([self.input_img,self.lit,self.shadow_mask],1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred +1) /2        
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = self.final*2-1

        #GAN input:
        #self.masked_fake = self.final*self.penumbra
        #self.masked_real = self.shadowfree_img*self.penumbra
        

    def backward(self):
        criterion = self.criterionL1 
        self.shadow_param[:,[1,3,5]] = self.shadow_param[:,[1,3,5]] - 2
        self.shadow_param[:,[0,2,4]] = self.shadow_param[:,[0,2,4]] * 2
        self.loss_G_param = criterion(self.shadow_param_pred, self.shadow_param) * self.opt.lambda_param 
        self.loss_rescontruction = criterion(self.final,self.shadowfree_img) * self.opt.lambda_res

        self.loss_bd = criterion(self.final[self.penumbra>0],self.shadowfree_img[self.penumbra>0]) * self.opt.lambda_bd

        self.loss_smooth = smooth_loss(self.alpha_pred) * self.opt.lambda_smooth
        self.loss = self.loss_rescontruction + self.loss_G_param + self.loss_bd + self.loss_smooth
        self.loss.backward()
    


    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_M.zero_grad()
        self.backward()
        self.optimizer_G.step()
        self.optimizer_M.step()
    
    def get_current_visuals(self):
        t= time.time()
        nim = self.input_img.shape[0]
        visual_ret = OrderedDict()
        all =[]
        for i in range(0,min(nim-1,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:])
                        row.append(im)
            
            row=tuple(row)
            row = np.hstack(row)
            if hasattr(self,'isreal'):
                if self.isreal[i] == 0:
                    row = ImageOps.crop(Image.fromarray(row),border =5)
                    row = ImageOps.expand(row,border=5,fill=(0,200,0))
                    row = np.asarray(row)
            all.append(row)
        all = tuple(all)
        
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])
    
    
    
    def get_prediction(self,input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        
        inputG = F.upsample(inputG,size=(self.opt.fineSize,self.opt.fineSize))
        self.shadow_param_pred = self.netG(inputG)
        w = self.input_img.shape[2]
        h = self.input_img.shape[3]
        n = self.input_img.shape[0]
        m = self.input_img.shape[1]
        self.shadow_param_pred = self.shadow_param_pred.view([n,6,-1])
        self.shadow_param_pred = torch.mean(self.shadow_param_pred,dim=2)

        self.shadow_param_pred[:,[0,2,4]] = self.shadow_param_pred[:,[0,2,4]] /2 
        self.shadow_param_pred[:,[1,3,5]] = self.shadow_param_pred[:,[1,3,5]] +2 


        self.lit = self.input_img.clone()/2+0.5
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = self.shadow_param_pred[:,[1,3,5]]
        #mul = (mul +2) * 5/3          
        n = self.shadow_param_pred.shape[0]
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        self.lit = self.lit*mul + add
        self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
        self.out = self.out*2-1
        
        
        inputM = torch.cat([self.input_img,self.lit,self.shadow_mask],1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred +1) /2        
        #self.alpha_pred_3d=  self.alpha_pred.repeat(1,3,1,1)
        
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*self.alpha_pred
        self.final = self.final*2-1

        RES = dict()
        RES['final']= util.tensor2im(self.final,scale =0)
        #RES['phase1'] = util.tensor2im(self.out,scale =0)
        #RES['param']= self.shadow_param_pred.detach().cpu() 
        RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)

        ###EVAL on original size
        input_img_ori = input['A_ori'].to(self.device)
        input_img_ori = input_img_ori/2+0.5
        lit_ori = input_img_ori
        w = input_img_ori.shape[2]
        h = input_img_ori.shape[3]
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = self.shadow_param_pred[:,[1,3,5]]
        #mul = (mul +2) * 5/3          
        n = self.shadow_param_pred.shape[0]
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        lit_ori = lit_ori*mul + add
        alpha_pred = F.upsample(self.alpha_pred,(w,h),mode='bilinear',align_corners=True)
        final  = input_img_ori * (1-alpha_pred) + lit_ori*(alpha_pred)
        final = final*2 -1 
        RES['ori_Size'] = util.tensor2im(final.detach().cpu())
        return  RES
