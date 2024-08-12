import numpy as np
import torch
import cv2

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

        # tracker maintain a list contains(self.tracks) for each Track object
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        # bbox_xywh (#obj,4), [xc,yc, w, h]     bounding box for each person
        # conf (#obj,1)

        self.height, self.width = ori_img.shape[:2]

        # get appearance feature with neural network (Deep) *********************************************************
        features = self._get_features(bbox_xywh, ori_img)

        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)   # # [cx,cy,w,h] -> [x1,y1,w,h]   top left

        #  generate detections class object for each person *********************************************************
        # filter object with less confidence
        # each Detection obj maintain the location(bbox_tlwh), confidence(conf), and appearance feature
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression (useless) *******************************************************************
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)  # Here, nms_max_overlap is 1
        detections = [detections[i] for i in indices]

        # update tracker ********************************************************************************************
        self.tracker.predict()      # predict based on t-1 info
        # for first frame, this function do nothing

        # detections is the measurement results as time T
        self.tracker.update(detections)

        # output bbox identities ************************************************************************************
        outputs = []
        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()       # (xc,yc,a,h) to (x1,y1,w,h)
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)  # (#obj, 5) (x1,y1,x2,y2,ID)
        return outputs

    def update_modified(self, bbox_xywh, keypts, confidences, ori_img):
        # bbox_xywh (#obj,4), [xc,yc, w, h]     bounding box for each person
        # conf (#obj,1)

        self.height, self.width = ori_img.shape[:2]

        # get appearance feature with neural network (Deep) *********************************************************
        features = self._get_features_modified(bbox_xywh, keypts, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)   # # [cx,cy,w,h] -> [x1,y1,w,h]   top left

        #  generate detections class object for each person *********************************************************
        # filter object with less confidence
        # each Detection obj maintain the location(bbox_tlwh), confidence(conf), and appearance feature
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression (useless) *******************************************************************
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)  # Here, nms_max_overlap is 1
        detections = [detections[i] for i in indices]

        # update tracker ********************************************************************************************
        self.tracker.predict()      # predict based on t-1 info
        # for first frame, this function do nothing

        # detections is the measurement results as time T
        self.tracker.update(detections)

        # output bbox identities ************************************************************************************
        outputs = []
        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()       # (xc,yc,a,h) to (x1,y1,w,h)
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)  # (#obj, 5) (x1,y1,x2,y2,ID)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _get_features_modified(self, bbox_xywh, keypts, ori_img):
        im_crops = []
        for hu in range(keypts.shape[0]):
            crop_img = self.human_seg(bbox_xywh[hu], keypts[hu], ori_img)
            im_crops.append(crop_img)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
        
    def human_seg(self, box_in, keypts, ori_img):
        left_parts = [1,3,5,11,13,15]
        right_parts = [2,4,6,12,14,16]

        x1,y1,x2,y2 = self._xywh_to_xyxy(box_in)
        # check if the left or right points are not on the left or right
        new_left_parts = []
        new_right_parts = []
        for cnt_pts in range(0,len(left_parts)):
            if keypts[left_parts[cnt_pts]][0] > keypts[right_parts[cnt_pts]][0]:
                new_left_parts.append(right_parts[cnt_pts])
                new_right_parts.append(left_parts[cnt_pts])
            else:
                new_left_parts.append(left_parts[cnt_pts])
                new_right_parts.append(right_parts[cnt_pts])
        body_polypts = np.concatenate((new_left_parts,new_right_parts[::-1]))

        msk = np.zeros(ori_img.shape[:2],np.uint8)
        # Draw keypoints
        vis_thres = 0.5
        for n in range(keypts.shape[0]):
            if keypts[n][2] <= vis_thres:
                if n in body_polypts:
                    body_polypts = np.delete(body_polypts,np.argwhere(body_polypts==n))
                continue

        cv2.fillPoly(msk, [np.array(keypts[body_polypts][:,0:2],np.int32)], color=255)

        # draw_hands 
        arm_pairs = [(5, 7), (7, 9), (6, 8), (8, 10)]
        for i, (start_p, end_p) in enumerate(arm_pairs):
            if keypts[start_p][2] <= vis_thres or keypts[end_p][2] <= vis_thres:
                continue
            else:
                start_xy = (int(keypts[start_p][0]),int(keypts[start_p][1]))
                end_xy = (int(keypts[end_p][0]),int(keypts[end_p][1]))
                cv2.line(msk, start_xy, end_xy, 255, 5)

        crop_img = ori_img[y1:y2,x1:x2]
        crop_msk = msk[y1:y2,x1:x2]

        kernel = np.ones((7,7), np.uint8)
        crop_msk_dilation = cv2.dilate(crop_msk, kernel, iterations=1)
        crop_msk_dilation = np.array(crop_msk_dilation/255.).astype('uint8')
        crop_img = crop_img*crop_msk_dilation[:,:,np.newaxis]

        return crop_img

            
    


