import argparse
import numpy as np
from tqdm import tqdm
import torch
from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils
import cv2
from pprint import pprint
import json
import math

class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]

def draw_bbox(img, bbox, thickness=3, color=(255, 0, 0)):
    canvas = img.copy()
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return canvas

def draw_mask(img, mask, thickness=3, color=(255, 0, 0)):
    def _get_edge(mask, thickness=3):
        dtype = mask.dtype
        x=cv2.Sobel(np.float32(mask),cv2.CV_16S,1,0, ksize=thickness) 
        y=cv2.Sobel(np.float32(mask),cv2.CV_16S,0,1, ksize=thickness)
        absX=cv2.convertScaleAbs(x)
        absY=cv2.convertScaleAbs(y)  
        edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return edge.astype(dtype)
    
    img = img.copy()
    canvas = np.zeros(img.shape, img.dtype) + color
    img[mask > 0] = img[mask > 0] * 0.8 + canvas[mask > 0] * 0.2
    edge = _get_edge(mask, thickness)
    img[edge > 0] = img[edge > 0] * 0.2 + canvas[edge > 0] * 0.8
    return img

def draw_skeleton(img, kpt, connection=None, colors=None, bbox=None):
    kpt = np.array(kpt, dtype=np.int32).reshape(-1, 3)
    npart = kpt.shape[0]
    canvas = img.copy()

    if npart==17: # coco
        part_names = ['nose', 
                      'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                      'left_knee', 'right_knee', 'left_ankle', 'right_ankle'] 
        #visible_map = {1: 'vis', 
        #               2: 'not_vis', 
        #               3: 'missing'}
        visible_map = {2: 'vis', 
                       1: 'not_vis', 
                       0: 'missing'}
        map_visible = {value: key for key, value in visible_map.items()}
        if connection is None:
            connection = [[16, 14], [14, 12], [17, 15], 
                          [15, 13], [12, 13], [6, 12], 
                          [7, 13], [6, 7], [6, 8], 
                          [7, 9], [8, 10], [9, 11], 
                          [2, 3], [1, 2], [1, 3], 
                          [2, 4], [3, 5], [4, 6], [5, 7]]
    elif npart==19: # ochuman
        part_names = ["right_shoulder", "right_elbow", "right_wrist",
                     "left_shoulder", "left_elbow", "left_wrist",
                     "right_hip", "right_knee", "right_ankle",
                     "left_hip", "left_knee", "left_ankle",
                     "head", "neck"] + \
                     ['right_ear', 'left_ear', 'nose', 'right_eye', 'left_eye']
        visible_map = {0: 'missing', 
                       1: 'vis', 
                       2: 'self_occluded', 
                       3: 'others_occluded'}
        map_visible = {value: key for key, value in visible_map.items()}
        if connection is None:
            connection = [[16, 19], [13, 17], [4, 5],
                         [19, 17], [17, 14], [5, 6],
                         [17, 18], [14, 4], [1, 2],
                         [18, 15], [14, 1], [2, 3],
                         [4, 10], [1, 7], [10, 7],
                         [10, 11], [7, 8], [11, 12], [8, 9],
                         [16, 4], [15, 1]] # TODO
            
    
    if colors is None:
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], 
                 [255, 255, 0], [170, 255, 0], [85, 255, 0], 
                 [0, 255, 0], [0, 255, 85], [0, 255, 170], 
                 [0, 255, 255], [0, 170, 255], [0, 85, 255], 
                 [0, 0, 255], [85, 0, 255], [170, 0, 255],
                 [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    elif type(colors[0]) not in [list, tuple]:
        colors = [colors]
    
    idxs_draw = np.where(kpt[:, 2] != map_visible['missing'])[0]
    if len(idxs_draw)==0:
        return img
    
    if bbox is None:
        bbox = [np.min(kpt[idxs_draw, 0]), np.min(kpt[idxs_draw, 1]),
                np.max(kpt[idxs_draw, 0]), np.max(kpt[idxs_draw, 1])] # xyxy
    
    Rfactor = math.sqrt((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / math.sqrt(img.shape[0] * img.shape[1])
    Rpoint = int(min(10, max(Rfactor*10, 4)))
    Rline = int(min(10, max(Rfactor*5, 2)))
    #print (Rfactor, Rpoint, Rline)
    
    for idx in idxs_draw:
        x, y, v = kpt[idx, :]
        cv2.circle(canvas, (x, y), Rpoint, colors[idx%len(colors)], thickness=-1)
        
        if v==2:
            cv2.rectangle(canvas, (x-Rpoint-1, y-Rpoint-1), (x+Rpoint+1, y+Rpoint+1), 
                          colors[idx%len(colors)], 1)
        elif v==3:
            cv2.circle(canvas, (x, y), Rpoint+2, colors[idx%len(colors)], thickness=1)

    for idx in range(len(connection)):
        idx1, idx2 = connection[idx]
        y1, x1, v1 = kpt[idx1-1]
        y2, x2, v2 = kpt[idx2-1]
        if v1 == map_visible['missing'] or v2 == map_visible['missing']:
            continue
        mX = (x1+x2)/2.0
        mY = (y1+y2)/2.0
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), Rline), int(angle), 0, 360, 1)
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, colors[idx%len(colors)])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
    return canvas

# Namespace(OCHuman=True, coco=True, weights='last.pkl')
cfg = AttrDict()
cfg.OCHuman=True
cfg.coco=True
cfg.weights='/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Pose2Seg/log/pose2seg_release.pkl'

device = torch.device('cuda', 1)

model = Pose2Seg(device).to(device)
model.init(cfg.weights)

model.eval()

dir_save = '/nethome/hkwon64/Research/imuTube/repos_v2/human_parsing/Pose2Seg/demo'

file_im = dir_save + '/demo.jpg'
file_ap = dir_save + '/alphapose-results.json'
ap_result = json.load(open(file_ap, 'r'))
gt_bbox = []
gt_kpts = []
for result in ap_result:
    x1, y1, w, h = result['box']
    bbox = int(x1), int(y1), int(x1+w), int(y1+h)
    gt_bbox.append(bbox)

    kps = np.array(result['keypoints']).reshape((-1, 3))
    gt_kpts.append(kps)
gt_kpts=np.array(gt_kpts)

img = cv2.imread(file_im)
height, width = img.shape[0:2]
    
output = model([img], [gt_kpts])
pprint (output)
assert False

# visualize & save results
colors = [[255, 0, 0], 
         [255, 255, 0],
         [0, 255, 0],
         [0, 255, 255], 
         [0, 0, 255], 
         [255, 0, 255]]

file_save = dir_save + '/demo_mask.jpg'
for i, mask in enumerate(output[0]):
    kpt = gt_kpts[i]
    bbox = gt_bbox[i]

    img = draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
    img = draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)
    img = draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])
cv2.imwrite(file_save, img)
print ('save in ...', file_save)

#--------------
assert False

def test(model, dataset='cocoVal', logger=print):    
    if dataset == 'OCHumanVal':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json'
    elif dataset == 'OCHumanTest':
        ImageRoot = './data/OCHuman/images'
        AnnoFile = './data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json'
    elif dataset == 'cocoVal':
        ImageRoot = './data/coco2017/val2017'
        AnnoFile = './data/coco2017/annotations/person_keypoints_val2017_pose2seg.json'
    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    model.eval()
    
    results_segm = []
    imgIds = []
    for i in tqdm(range(len(datainfos))):
        rawdata = datainfos[i]
        img = rawdata['data']
        image_id = rawdata['id']
        
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata['segms']
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
            
        output = model([img], [gt_kpts], [gt_masks])
    
        for mask in output[0]:
            maskencode = maskUtils.encode(np.asfortranarray(mask))
            maskencode['counts'] = maskencode['counts'].decode('ascii')
            results_segm.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "score": 1.0,
                    "segmentation": maskencode
                })
        imgIds.append(image_id)
    
    
    def do_eval_coco(image_ids, coco, results, flag):
        from pycocotools.cocoeval import COCOeval
        assert flag in ['bbox', 'segm', 'keypoints']
        # Evaluate
        coco_results = coco.loadRes(results)
        cocoEval = COCOeval(coco, coco_results, flag)
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() 
        return cocoEval
    
    cocoEval = do_eval_coco(imgIds, datainfos.COCO, results_segm, 'segm')
    logger('[POSE2SEG]          AP|.5|.75| S| M| L|    AR|.5|.75| S| M| L|')
    _str = '[segm_score] %s '%dataset
    for value in cocoEval.stats.tolist():
        _str += '%.3f '%value
    logger(_str)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg Testing")
    parser.add_argument(
        "--weights",
        help="path to .pkl model weight",
        type=str,
    )
    parser.add_argument(
        "--coco",
        help="Do test on COCOPersons val set",
        action="store_true",
    )
    parser.add_argument(
        "--OCHuman",
        help="Do test on OCHuman val&test set",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(args.weights)
            
    print('===========>   testing    <===========')
    if args.coco:
        test(model, dataset='cocoVal') 
    if args.OCHuman:
        test(model, dataset='OCHumanVal')
        test(model, dataset='OCHumanTest') 
