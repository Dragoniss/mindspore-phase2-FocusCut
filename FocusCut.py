#[General module]
import os
import time
import random
import shutil
import numpy as np
from tqdm import tqdm

#[Other module]
import mindspore
import inference
from PIL import Image

#[Personal module]
import utils
import helpers
import my_custom_transforms as mtr
from dataloader_cut import GeneralCutDataset
from model.general.sync_batchnorm import patch_replication_callback

class Trainer(object):
    def __init__(self,p):
        self.p=p

    def key_code(self):
        for i, sample in enumerate(tqdm(val_robot_set)):
            id = sample['meta']['id']
            gt = np.array(Image.open(sample['meta']['gt_path']))
            pred = np.zeros_like(gt) 
            seq_points=np.empty([0,3],dtype=np.int64)
            id_preds,id_ious=[helpers.encode_mask(pred)],[0.0]
            id_other_metrics= {metric: [0.0] for metric in self.p['other_metric']}
            hr_points=[]
            sample['pre_pred']=pred
            if self.p['zoom_in']==0:
                inference.predict_wo(self.p,self.model_src,sample,np.array([helpers.get_next_anno_point(np.zeros_like(gt), gt)],dtype=np.int64)) #add -wo
            for point_num in range(1, self.p['max_point_num']+1):
                pt_next = helpers.get_next_anno_point(pred, gt, seq_points)
                seq_points=np.append(seq_points,[pt_next],axis=0)
                pred_tmp,result_tmp = inference.predict_wo(self.p,self.model_src,sample,seq_points)
                if point_num>1 and self.p['model'].startswith('hrcnet') and if_hrv and p['hr_val_setting']['pfs']!=0:
                    expand_r,if_hr=inference.cal_expand_r_new_final(pt_next,pred,pred_tmp)
                    if if_hr: 
                        hr_point={'point_num':point_num,'pt_hr':pt_next,'expand_r':expand_r,'pre_pred_hr':None,'seq_points_hr':None,'hr_result_src':None,'hr_result_count_src':None,'img_hr':None,'pred_hr':None,'gt_hr':None}
                        hr_points.append(hr_point)
                pred= inference.predict_hr_new_final(self.p,self.model_src,sample,seq_points,hr_points,pred=pred_tmp,result=result_tmp) if len(hr_points)>0 else pred_tmp
                for metric in id_other_metrics: id_other_metrics[metric].append(helpers.get_metric(pred,gt,metric))
                miou = ((pred==1)&(gt==1)).sum()/(((pred==1)|(gt==1))&(gt!=255)).sum()
                id_ious.append(miou)
                id_preds.append(helpers.encode_mask(pred))
                if (np.array(id_ious)>=max_miou_target).any() and point_num>=self.p['record_point_num']:break