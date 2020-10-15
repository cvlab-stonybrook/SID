import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image,ImageChops
from PIL import ImageFilter
import torch
from pdb import set_trace as st
import random
import numpy as np
import time
class WeaklyShadowParamDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.all =[]
        self.real_only = []
        self.fake_only = []

        dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        dir_param = os.path.join(opt.dataroot, opt.phase + 'param')
        dir_matte = os.path.join(opt.dataroot, 'matte')
        for root,_,fnames in sorted(os.walk(dir_A)):
            for fname in fnames:
                if fname.endswith('.png'):
                    X=dict()
                    X['im_path'] = os.path.join(root,fname)
                    X['mask_path'] = os.path.join(dir_B,fname)
                    X['GT_path'] = os.path.join(dir_C,fname)
                    X['param_path'] = os.path.join(dir_param,fname+'.txt')
                    X['isreal'] = True
                    X['imname'] = fname
                    self.real_only.append(X)
                    self.all.append(X)
       

        dir_A = os.path.join(opt.wdataroot, opt.phase + 'A')
        dir_B = os.path.join(opt.wdataroot, opt.phase + 'B')
        dir_C = os.path.join(opt.wdataroot, opt.phase + 'C')
        dir_param = os.path.join(opt.wdataroot, opt.phase + 'param')
        dir_matte = os.path.join(opt.wdataroot, 'matte')
        for root,_,fnames in sorted(os.walk(dir_A)):
            for fname in fnames:
                if fname.endswith('.png'):
                    X=dict()
                    X['im_path'] = os.path.join(root,fname)
                    X['mask_path'] = os.path.join(dir_B,fname)
                    X['GT_path'] = os.path.join(dir_C,fname)
                    X['param_path'] = os.path.join(dir_param,fname+'.txt')
                    X['isreal'] = False
                    X['imname'] = fname
                    self.fake_only.append(X)
                    self.all.append(X)


        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=opt.norm_mean,
                                               std = opt.norm_std)]

        self.transformA = transforms.Compose(transform_list)
        self.transformB = transforms.Compose([transforms.ToTensor()])
        self.stats()
    def stats(self):
        print("Dataset type: %s "%(self.name()))
        print("Total Image: %d "%(len(self.all)))
        print("Real / FAke : %d / %d "%(len(self.real_only),len(self.all)-len(self.real_only)))
    def __getitem__(self,index):
        birdy = {}
            
        if not hasattr(self.opt,'s_real'):
            self.opt.s_real = 0.5
        
        if random.random() < self.opt.s_real:
            choosen_set= 'real_only'
        else:
            choosen_set = 'fake_only'
        index = random.randint(0,len(getattr(self,choosen_set))-1)
        choosen = getattr(self,choosen_set)[index]
        A_path = choosen['im_path']
        imname = choosen['imname']
        B_path = choosen['mask_path']
        A_img = Image.open(A_path).convert('RGB')
        sparam = open(choosen['param_path'])
        line = sparam.read()
        shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()])
        shadow_param = shadow_param[0:6]
        
        ow = A_img.size[0]
        oh = A_img.size[1]
        w = np.float(A_img.size[0])
        h = np.float(A_img.size[1])
        if os.path.isfile(B_path): 
            B_img = Image.open(B_path)
        else:
            print('MASK NOT FOUND : %s'%(B_path))
            B_img = Image.fromarray(np.zeros((int(w),int(h)),dtype = np.float),mode='L')
        
        birdy['C'] = Image.open(choosen['GT_path']).convert('RGB')
       
        loadSize = self.opt.loadSize
        if self.opt.randomSize:
            loadSize = np.random.randint(loadSize + 1,loadSize * 1.3 ,1)[0]

#        if self.opt.randomSize:
#            loadSize = np.random.randint(loadSize + 1,loadSize * 1.3 ,1)[0]
        
        if self.opt.keep_ratio:
            if w>h:
                ratio = np.float(loadSize)/np.float(h)
                neww = np.int(w*ratio)
                newh = loadSize
            else:
                ratio = np.float(loadSize)/np.float(w)
                neww = loadSize
                newh = np.int(h*ratio)
        else:
            neww = loadSize
            newh = loadSize
        
        birdy['A'] = A_img
        birdy['B'] = B_img
        t =[Image.FLIP_LEFT_RIGHT,Image.ROTATE_90]
        for i in range(0,4):
            c = np.random.randint(0,3,1,dtype=np.int)[0]
            if c==2: continue
            for i in ['A','B','C']:
                if i in birdy:
                    birdy[i]=birdy[i].transpose(t[c])
                

        degree=np.random.randint(-20,20,1)[0]
        for i in ['A','B','C']:
            birdy[i]=birdy[i].rotate(degree)
        

        for k,im in birdy.items():
            birdy[k] = im.resize((neww, newh),Image.NEAREST)
        
        w = birdy['A'].size[0]
        h = birdy['A'].size[1]
        #rescale A to the range [1; log_scale] then take log
        #birdy['B_over'] = B_img.filter(ImageFilter.MaxFilter(7)) 
        if self.opt.lambda_bd>0:
            birdy['bd_img'] =  ImageChops.subtract(B_img.filter(ImageFilter.MaxFilter(3)),B_img.filter(ImageFilter.MinFilter(3))) 
                    
        for k,im in birdy.items():
            birdy[k] = self.transformB(im)
        #aug_lit = (np.random.rand()-0.5)+1
        #for i in ['A','C']:
        #    birdy[i]=birdy[i]*aug_lit        
        #shadow_param[[0,2,4]] = shadow_param[[0,2,4]]*aug_lit
        
        for i in ['A','C','B']:
            if i in birdy:
                birdy[i] = (birdy[i] - 0.5)*2         
        
        if not self.opt.no_crop:        
            w_offset = random.randint(0,max(0,w-self.opt.fineSize-1))
            h_offset = random.randint(0,max(0,h-self.opt.fineSize-1))
            for k,im in birdy.items():   
                birdy[k] = im[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(birdy['A'].size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            for k,im in birdy.items():
                birdy[k] = im.index_select(2, idx)
        for k,im in birdy.items():
            birdy[k] = im.type(torch.FloatTensor)
        birdy['imname'] = imname
        birdy['w'] = ow
        birdy['h'] = oh
        birdy['A_paths'] = A_path
        birdy['B_baths'] = B_path
        
        #if the shadow area is too small, let's not change anything:
        if torch.sum(birdy['B']>0) < 10 :
            shadow_param=[0,1,0,1,0,1]
            
        birdy['isreal'] = 1 if choosen['isreal'] else 0
        birdy['param'] = torch.FloatTensor(np.array(shadow_param))
        #print(birdy['param'])
        return birdy 
    def __len__(self):
        return len(self.real_only)*2 

    def name(self):
        return 'ShadowParamDataset'
