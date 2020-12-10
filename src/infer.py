import sys
import time
import torch
import os
from options.test_options import TestOptions
from models import create_model
from data.single_dataset import SingleDataset
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import util.util as util
import numpy as np
from PIL import Image
from util.util import sdmkdir
class Predict:
    def __init__(self,opt):
        opt.gpu_ids=[3]
        opt.checkpoints_dir ='../checkpoints/'  
        opt.netG = 'RESNEXT'
        opt.fineSize = 256
        opt.loadSize = 256
        opt.isTrain = False
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        self.model = model
        self.opt = opt
    
    def ISTD_test(self):
        opt = self.opt
        opt.mask_test = '/nfs/bigneuron/add_disk0/hieule/data/datasets/ISTD_Dataset/Mean_Teacher_SD/ISTD_crf'
        dataset = SingleDataset('/nfs/bigneuron/add_disk0/hieule/data/datasets/ISTD_Dataset/test/test_A',opt)
        opt.results_dir ='./ISTD_b/' 
        self.eval_backend_output_only(dataset,opt)
    
    
    def eval_backend_output_only(self,dataset,opt): 
        opt.fresdir =opt.results_dir
        if not os.path.exists(opt.fresdir):
            os.makedirs(opt.fresdir)
        print("%s: ; %d" %(dataset.name,len(dataset)))
        st = time.time()
        for i, data in enumerate(dataset):
            sd = self.model.get_prediction(data)
            resname = (data['imname'])
            if isinstance(sd,dict):
                for k in sd:
                    if k == 'final':
                        im = Image.fromarray(sd[k])
                        im.save(os.path.join(opt.fresdir,resname))

    def eval_backend(self,dataset,opt):
        opt.resdir = '//'+opt.name+'/'+opt.epoch +'/'+str(opt.loadSize)+'/'
        opt.fresdir =opt.results_dir+ '/' + opt.resdir
        evaldir = opt.fresdir+'/final/'
        if not os.path.exists(opt.fresdir):
            os.makedirs(opt.fresdir)
        print("%s: ; %d" %(dataset.name,len(dataset)))
        st = time.time()
        for i, data in enumerate(dataset):
            sd = self.model.get_prediction(data)
            resname = (data['imname'])
            if isinstance(sd,dict):
                evaldir = opt.fresdir+'/final/'
                for k in sd:
                    if k == 'param':
                        sdmkdir(os.path.join(opt.fresdir,k))
                        np.savetxt(os.path.join(opt.fresdir,k,resname+'.txt'), sd[k], delimiter=' ') 
                    else:
                        sdmkdir(os.path.join(opt.fresdir,k))
                        im = Image.fromarray(sd[k])
                        im.save(os.path.join(opt.fresdir,k,resname))
            else:
                evaldir = opt.fresdir
                sd = Image.fromarray(sd)
                sd.save(os.path.join(opt.fresdir,resname))


if __name__=='__main__':
    opt = TestOptions().parse()    #args.parse()
    a= Predict(opt)
    a.ISTD_test()
    
