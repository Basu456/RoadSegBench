import numpy as np
def binary_metrics(pred,target,threshold=.5,eps=1e-9):
    p=np.asarray(pred)>=threshold; t=np.asarray(target)>=.5
    tp=np.logical_and(p,t).sum(); fp=np.logical_and(p,~t).sum(); fn=np.logical_and(~p,t).sum()
    return {"road_iou":float(tp/(tp+fp+fn+eps)),
            "dice":float(2*tp/(2*tp+fp+fn+eps)),
            "precision":float(tp/(tp+fp+eps)),
            "recall":float(tp/(tp+fn+eps))}
def boundary_iou(pred,target,threshold=.5,radius=2,eps=1e-9):
    import cv2
    p=(np.asarray(pred)>=threshold).astype(np.uint8); t=(np.asarray(target)>=.5).astype(np.uint8)
    k=np.ones((2*radius+1,2*radius+1),np.uint8)
    pb=p-cv2.erode(p,k); tb=t-cv2.erode(t,k)
    return float(np.logical_and(pb>0,tb>0).sum()/(np.logical_or(pb>0,tb>0).sum()+eps))
