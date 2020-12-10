# Shadow Removal via Shadow Image Decomposition
Pytorch implementation for ICCV19 "Shadow Removal via Shadow Image Decomposition" 

[Project Page](https://www3.cs.stonybrook.edu/~cvl/projects/SID/index.html)

[Paper](https://arxiv.org/abs/1908.08628)

<img src='./training.png' align="center">

**New**: Please check out [Weakly Supervised Shadow Removal](https://github.com/lmhieu612/FSS2SR), our new unparied patch-to-patch translation model for shadow removal.

This pytorch implementation is heavily based on the pix2pix framework written by [Jun-Yan Zhu](https://github.com/junyanz). Many thanks!

**Pretrained-model**:

--This version:

https://drive.google.com/drive/folders/17G_lf1k2CNt9wt4X2hWxeT8-7bOXpWQe?usp=sharing

--ICCV19 version: To be updated soon.


**Note**: We have made several technical improvements over the original implementation and this code might generate slightly better results than what reported in the original paper, getting around 7.0 RMSE on shadow area. We will include the pre-trained models of both the old version and this version soon.


##Testing
1. Download the pretrained-model above and but them into ./checkpoint_path/model_name/..pth
2. Set the path to the shadow-mask of the test set
3. Run by ```python infer.py --model SIDPAMIw --name model_name --epoch best```



To generate "train_params": please run the ipython notebook included in "data_processing".

If you are using this code for research, please cite:

```
Shadow Removal via Shadow Image Decomposition 
Hieu Le and Dimitris Samaras

@InProceedings{Le_2019_ICCV,
	author = {Le, Hieu and Samaras, Dimitris},
	title = {Shadow Removal via Shadow Image Decomposition},
	booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	month = {October},
	year = {2019}
}
```

And also take a look at our other shadow papers:
```
A+D-Net: Shadow Detection with Adversarial Shadow Attenuation
Hieu Le, Tomas F. Yago Vicente, Vu Nguyen, Minh Hoai, Dimitris Samaras

@inproceedings{m_Le-etal-ECCV18,
Author = {Hieu Le and Tomas F. Yago Vicente and Vu Nguyen and Minh Hoai and Dimitris Samaras},
Booktitle = {Proceedings of European Conference on Computer Vision},
Title = {{A+D Net}: Training a Shadow Detector with Adversarial Shadow Attenuation},
Year = {2018}}


From Shadow Segmentation to Shadow Removal
Hieu Le and Dimitris Samaras

@InProceedings{Le_2020_ECCV,
	author = {Le, Hieu and Samaras, Dimitris},
	title = {From Shadow Segmentation to Shadow Removal},
	booktitle = {The IEEE European Conference on Computer Vision (ECCV)},
	month = {August},
	year = {2020}
}
```
