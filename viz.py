# -*- coding: utf-8 -*-
# File: viz.py

from six.moves import zip
import numpy as np

from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB

from utils.np_box_ops import iou as np_iou
from config import config as cfg

from scipy import misc,ndimage
import random

def draw_annotation(img, boxes, klass, is_crowd=None):
    labels = []
    assert len(boxes) == len(klass)
    if is_crowd is not None:
        assert len(boxes) == len(is_crowd)
        for cls, crd in zip(klass, is_crowd):
            clsname = cfg.DATA.CLASS_NAMES[cls]
            if crd == 1:
                clsname += ';Crowd'
            labels.append(clsname)
    else:
        for cls in klass:
            labels.append(cfg.DATA.CLASS_NAMES[cls])
    img = viz.draw_boxes(img, boxes, labels)
    return img


def draw_proposal_recall(img, proposals, proposal_scores, gt_boxes):
    """
    Draw top3 proposals for each gt.
    Args:
        proposals: NPx4
        proposal_scores: NP
        gt_boxes: NG
    """
    box_ious = np_iou(gt_boxes, proposals)    # ng x np
    box_ious_argsort = np.argsort(-box_ious, axis=1)
    good_proposals_ind = box_ious_argsort[:, :3]   # for each gt, find 3 best proposals
    good_proposals_ind = np.unique(good_proposals_ind.ravel())

    proposals = proposals[good_proposals_ind, :]
    tags = list(map(str, proposal_scores[good_proposals_ind]))
    img = viz.draw_boxes(img, proposals, tags)
    return img, good_proposals_ind


def draw_predictions(img, boxes, scores):
    """
    Args:
        boxes: kx4
        scores: kxC
    """
    if len(boxes) == 0:
        return img
    labels = scores.argmax(axis=1)
    scores = scores.max(axis=1)
    tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[lb], score) for lb, score in zip(labels, scores)]
    return viz.draw_boxes(img, boxes, tags)


def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    tags = []
    for r in results:
        tags.append(
            "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    boxes = np.asarray([r.box for r in results])
    ret = viz.draw_boxes(img, boxes, tags)

    for r in results:
        if r.mask is not None:
            ret = draw_mask(ret, r.mask)
    return ret


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im


def trimap_outputs(img, results, path, img_name):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return 

    trimaps = []
    boxes = [r.box for r in results]
    # copy from tensorpack.utils.viz.draw_boxes
    if isinstance(boxes, list):
        arr = np.zeros((len(boxes), 4), dtype='int32')
        for idx, b in enumerate(boxes):
            #assert isinstance(b, BoxBase), b
            arr[idx, :] = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
        boxes = arr
    else:
        boxes = boxes.astype('int32')
        
    trimap_count = 0
    for r,box in zip(results, boxes):
        tag = "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score)
        #boxes = np.asarray([r.box for r in results])
        #ret = []
        #for box in boxes:
        crop = img[box[0]: box[2], box[1]: box[3]]

        #for r, crop in zip(results, ret):
        if r.mask is not None:
            trimap = gen_trimap(crop, r.mask)
            misc.imsave(path+img_name+tag+str(trimap_count)+'.png', trimap)
            trimaps.append(trimap)
            trimap_count += 1
    return trimaps


def gen_trimap(im, mask):
    """
    generate trimap based on mask.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
    """
    trimap_kernel = [val for val in range(15,30)]

    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  225, 0)
    im = im.astype('uint8')
    k_size = random.choice(trimap_kernel)
    dilate = ndimage.grey_dilation(im[:,:,0],size=(k_size,k_size))
    erode = ndimage.grey_erosion(im[:,:,0],size=(k_size,k_size))
    # trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - ndimage.grey_erosion(alpha[:,:,0],size=(k_size,k_size)))!=0)] = 128
    im[np.where(dilate - erode>10)] = 128
    
    return im
