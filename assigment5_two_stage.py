import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import coutils
from coutils import extract_drive_file_id, register_colab_notebooks, \
                    fix_random_seed, rel_error
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import time
import shutil
import os

# for plotting
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# data type and device for torch.tensor
to_float = {'dtype': torch.float, 'device': 'cpu'}
to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
to_double = {'dtype': torch.double, 'device': 'cpu'}
to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
to_long = {'dtype': torch.long, 'device': 'cpu'}
to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}

if torch.cuda.is_available:
  print('Good to go!')
else:
  print('Please set GPU via Edit -> Notebook Settings.')
  
  YOLO_NOTEBOOK_LINK = ""

fcn_id = extract_drive_file_id(YOLO_NOTEBOOK_LINK)
print('Google Drive file id: "%s"' % fcn_id)
register_colab_notebooks({'single_stage_detector_yolo': fcn_id})

from single_stage_detector_yolo import data_visualizer, FeatureExtractor
from single_stage_detector_yolo import get_pascal_voc2007_data, pascal_voc2007_loader
from single_stage_detector_yolo import coord_trans, GenerateGrid, GenerateAnchor, GenerateProposal
from single_stage_detector_yolo import IoU, ReferenceOnActivatedAnchors   
from single_stage_detector_yolo import DetectionSolver, DetectionInference

print('Import successful!')

# uncomment below to use the mirror link if the original link is broken
# !wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
train_dataset = get_pascal_voc2007_data('/content', 'train')
val_dataset = get_pascal_voc2007_data('/content', 'val')

class_to_idx = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
                'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
                'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
}
idx_to_class = {i:c for c, i in class_to_idx.items()}

train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(0, 2500)) # use 2500 samples for training
train_loader = pascal_voc2007_loader(train_dataset, 10)
val_loader = pascal_voc2007_loader(val_dataset, 10)

train_loader_iter = iter(train_loader)
img, ann, _, _, _ = train_loader_iter.next()

print('Resized train images shape: ', img[0].shape)
print('Padded annotation tensor shape: ', ann[0].shape)
print(ann[0])
print('Each row in the annotation tensor indicates (x_tl, y_tl, x_br, y_br, class).')
print('Padded with bounding boxes (-1, -1, -1, -1, -1) to enable batch loading. (You may need to run a few times to see the paddings)')

# default examples for visualization
fix_random_seed(0)
batch_size = 3
sampled_idx = torch.linspace(0, len(train_dataset)-1, steps=batch_size).long()

# get the size of each image first
h_list = []
w_list = []
img_list = [] # list of images
MAX_NUM_BBOX = 40
box_list = torch.LongTensor(batch_size, MAX_NUM_BBOX, 5).fill_(-1) # PADDED GT boxes

for idx, i in enumerate(sampled_idx):
  # hack to get the original image so we don't have to load from local again...
  img, ann = train_dataset.__getitem__(i)
  img_list.append(img)

  all_bbox = ann['annotation']['object']
  if type(all_bbox) == dict:
    all_bbox = [all_bbox]
  for bbox_idx, one_bbox in enumerate(all_bbox):
    bbox = one_bbox['bndbox']
    obj_cls = one_bbox['name']
    box_list[idx][bbox_idx] = torch.LongTensor([int(bbox['xmin']), int(bbox['ymin']),
      int(bbox['xmax']), int(bbox['ymax']), class_to_idx[obj_cls]])

  # get sizes
  img = np.array(img)
  w_list.append(img.shape[1])
  h_list.append(img.shape[0])

w_list = torch.tensor(w_list, **to_float_cuda)
h_list = torch.tensor(h_list, **to_float_cuda)
box_list = torch.tensor(box_list, **to_float_cuda)
resized_box_list = coord_trans(box_list, w_list, h_list, mode='p2a')

# visualize GT boxes
for i in range(len(img_list)):
  valid_box = sum([1 if j != -1 else 0 for j in box_list[i][:, 0]])
  data_visualizer(img_list[i], idx_to_class, box_list[i][:valid_box])
  
  # Declare variables for anchor priors, a Ax2 Tensor where A is the number of anchors.
# Hand-picked, same as our two-stage detector.
anchor_list = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]], **to_float_cuda)
print(anchor_list.shape)

fix_random_seed(0)

grid_list = GenerateGrid(w_list.shape[0])
anc_list = GenerateAnchor(anchor_list, grid_list)
iou_mat = IoU(anc_list, resized_box_list)
activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
  activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anc_list, resized_box_list, grid_list, iou_mat)

expected_GT_conf_scores = torch.tensor([0.74538743, 0.72793430, 0.71128041, 0.70029843,
                                        0.75670898, 0.76044953, 0.37116671, 0.37116671], **to_float_cuda)
expected_GT_offsets = torch.tensor([[ 0.01633334,  0.11911901, -0.09431065,  0.19244696],
                                    [-0.03675002,  0.09324861, -0.00250307,  0.25213102],
                                    [-0.03675002, -0.15675139, -0.00250307,  0.25213102],
                                    [-0.02940002,  0.07459889, -0.22564663,  0.02898745],
                                    [ 0.11879997,  0.03208542,  0.20863886, -0.07974572],
                                    [-0.08120003,  0.03208542,  0.20863886, -0.07974572],
                                    [ 0.07699990,  0.28533328, -0.03459148, -0.86750042],
                                    [ 0.07699990, -0.21466672, -0.03459148, -0.86750042]], **to_float_cuda)
expected_GT_class = torch.tensor([ 6,  7,  7,  7, 19, 19,  6,  6], **to_long_cuda)
print('conf scores error: ', rel_error(GT_conf_scores, expected_GT_conf_scores))
print('offsets error: ', rel_error(GT_offsets, expected_GT_offsets))
print('class prob error: ', rel_error(GT_class, expected_GT_class))

# visualize the activated anchors
anc_per_img = torch.prod(torch.tensor(anc_list.shape[1:-1]))

print('*'*80)
print('Activated (positive) anchors:')
for img, bbox, idx in zip(img_list, box_list, torch.arange(box_list.shape[0])):
  anc_ind_in_img = (activated_anc_ind >= idx * anc_per_img) & (activated_anc_ind < (idx+1) * anc_per_img)
  print('{} activated anchors!'.format(torch.sum(anc_ind_in_img)))
  data_visualizer(img, idx_to_class, bbox[:, :4], coord_trans(activated_anc_coord[anc_ind_in_img], w_list[idx], h_list[idx]))

print('*'*80)
print('Negative anchors:')
for img, bbox, idx in zip(img_list, box_list, torch.arange(box_list.shape[0])):
  anc_ind_in_img = (negative_anc_ind >= idx * anc_per_img) & (negative_anc_ind < (idx+1) * anc_per_img)
  print('{} negative anchors!'.format(torch.sum(anc_ind_in_img)))
  data_visualizer(img, idx_to_class, bbox[:, :4], coord_trans(negative_anc_coord[anc_ind_in_img], w_list[idx], h_list[idx]))
  
  class ProposalModule(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_anchors=9, drop_ratio=0.3):
        super().__init__()

        assert num_anchors != 0
        self.num_anchors = num_anchors

        ##############################################################################
        # TODO: Define the region proposal layer - a sequential module with a 3x3    #
        # conv layer, followed by a Dropout (p=drop_ratio), a Leaky ReLU and         #
        # a 1x1 conv.                                                                #
        # HINT: The output should be of shape Bx(Ax6)x7x7, where A=self.num_anchors. #
        #       Determine the padding of the 3x3 conv layer given the output dim.    #
        ##############################################################################

        self.proposal_head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Dropout(p=drop_ratio),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, self.num_anchors * 6, kernel_size=1)
        )

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def _extract_anchor_data(self, anchor_data, anchor_idx):
        B, A, D, H, W = anchor_data.shape
        anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
        extracted_anchors = anchor_data[anchor_idx]
        return extracted_anchors

    def forward(self, features, pos_anchor_coord=None, pos_anchor_idx=None, neg_anchor_idx=None):
        if pos_anchor_coord is None or pos_anchor_idx is None or neg_anchor_idx is None:
            mode = 'eval'
        else:
            mode = 'train'

        conf_scores, offsets, proposals = None, None, None

        ##############################################################################
        # TODO: Predict classification scores (object vs background) and transforms #
        # for all anchors. During inference, simply output predictions for all     #
        # anchors. During training, extract the predictions for only the positive  #
        # and negative anchors as described above, and also apply the transforms   #
        # to the positive anchors to compute the coordinates of the region         #
        # proposals.                                                               #
        #                                                                          #
        # HINT: You can extract information about specific proposals using the     #
        # provided helper function self._extract_anchor_data.                      #
        # HINT: You can compute proposal coordinates using the GenerateProposal    #
        # function from the previous notebook.                                     #
        ##############################################################################

        output = self.proposal_head(features)

        B, _, H, W = output.shape
        A = self.num_anchors

        output = output.view(B, A, 6, H, W)

        conf_scores = output[:, :, :2, :, :]

        offsets = output[:, :, 2:, :, :]

        if mode == 'train':

            pos_conf_scores = self._extract_anchor_data(conf_scores, pos_anchor_idx)
            neg_conf_scores = self._extract_anchor_data(conf_scores, neg_anchor_idx)
            conf_scores = torch.cat([pos_conf_scores, neg_conf_scores], dim=0)


            offsets = self._extract_anchor_data(offsets, pos_anchor_idx)

            proposals = GenerateProposal(pos_anchor_coord.unsqueeze(0), offsets.unsqueeze(0)).squeeze(0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        if mode == 'train':
            return conf_scores, offsets, proposals
        elif mode == 'eval':
            return conf_scores, offsets

  
  # sanity check
fix_random_seed(0)
prop_module = ProposalModule(1280, drop_ratio=0).to(**to_float_cuda)
features = torch.linspace(-10., 10., steps=3*1280*7*7, **to_float_cuda).view(3, 1280, 7, 7)
conf_scores, offsets, proposals = prop_module(features, activated_anc_coord, \
              pos_anchor_idx=activated_anc_ind, neg_anchor_idx=negative_anc_ind)

expected_conf_scores = torch.tensor([[-0.50843990,  2.62025023],
                                     [-0.55775326, -0.29983672],
                                     [-0.55796617, -0.30000290],
                                     [ 0.17819080, -0.42211828],
                                     [-0.51439995, -0.47708601],
                                     [-0.51439744, -0.47703803],
                                     [ 0.63225138,  2.71269488],
                                     [ 0.63224381,  2.71290708]], **to_float_cuda)
expected_offsets = torch.tensor([[ 1.62754285,  1.35253453, -1.85451591, -1.77882397],
                                 [-0.33651856, -0.14402901, -0.07458937, -0.27201492],
                                 [-0.33671042, -0.14398587, -0.07479107, -0.27199429],
                                 [ 0.06847382,  0.21062726,  0.09334904, -0.02446130],
                                 [ 0.16506940, -0.30296192,  0.29626080,  0.32173073],
                                 [ 0.16507357, -0.30302414,  0.29625297,  0.32169008],
                                 [ 1.59992146, -0.75236654,  1.66449440,  2.05138564],
                                 [ 1.60008609, -0.75249159,  1.66474164,  2.05162382]], **to_float_cuda)

print('conf scores error: ', rel_error(conf_scores[:8], expected_conf_scores))
print('offsets error: ', rel_error(offsets, expected_offsets))

def ConfScoreRegression(conf_scores, batch_size):
  """
  Binary cross-entropy loss

  Inputs:
  - conf_scores: Predicted confidence scores, of shape (2M, 2). Assume that the
    first M are positive samples, and the last M are negative samples.

  Outputs:
  - conf_score_loss: Torch scalar
  """
  # the target conf_scores for positive samples are ones and negative are zeros
  M = conf_scores.shape[0] // 2
  GT_conf_scores = torch.zeros_like(conf_scores)
  GT_conf_scores[:M, 0] = 1.
  GT_conf_scores[M:, 1] = 1.

  conf_score_loss = F.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, \
                                     reduction='sum') * 1. / batch_size
  return conf_score_loss

def BboxRegression(offsets, GT_offsets, batch_size):
  """"
  Use SmoothL1 loss as in Faster R-CNN

  Inputs:
  - offsets: Predicted box offsets, of shape (M, 4)
  - GT_offsets: GT box offsets, of shape (M, 4)
  
  Outputs:
  - bbox_reg_loss: Torch scalar
  """
  bbox_reg_loss = F.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
  return bbox_reg_loss

conf_loss = ConfScoreRegression(conf_scores, features.shape[0])
reg_loss = BboxRegression(offsets, GT_offsets, features.shape[0])
print('conf loss: {:.4f}, reg loss: {:.4f}'.format(conf_loss, reg_loss))

loss_all = torch.tensor([conf_loss.data, reg_loss.data], **to_float_cuda)
expected_loss = torch.tensor([8.55673981, 5.10593748], **to_float_cuda)

print('loss error: ', rel_error(loss_all, expected_loss))

class RPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.anchor_list = torch.tensor([
            [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], 
            [2, 3], [3, 2], [3, 5], [5, 3]
        ])
        self.feat_extractor = FeatureExtractor()
        self.prop_module = ProposalModule(1280, num_anchors=self.anchor_list.shape[0])

    def forward(self, images, bboxes, output_mode='loss'):
        """
        Training-time forward pass for the Region Proposal Network.
        """
        # Weights for loss terms
        w_conf = 1  # for confidence scores
        w_reg = 5    # for offsets

        assert output_mode in ('loss', 'all'), 'invalid output mode!'
        total_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = \
            None, None, None, None, None, None

        ##############################################################################
        # TODO: Implement the forward pass of RPN.                                   #
        # i) Extract image features                                                  #
        # ii) Generate grid and anchor boxes                                         #
        # iii) Compute IoU between anchors and GT boxes and determine activations    #
        # iv) Compute conf_scores, offsets, proposals through the proposal module    #
        # v) Compute total_loss for RPN                                              #
        ##############################################################################
        features = self.feat_extractor(images)

        grid = GenerateGrid(features)
        anchors = GenerateAnchor(self.anchor_list, grid)

        iou_mat = IoU(anchors, bboxes)
        pos_anchor_idx, neg_anchor_idx, GT_conf_scores, GT_offsets, GT_class = \
            ReferenceOnActivatedAnchors(anchors, bboxes, iou_mat, neg_thresh=0.2)

        conf_scores, offsets, proposals = self.prop_module(
            features, anchors.view(-1, 4)[pos_anchor_idx], pos_anchor_idx, neg_anchor_idx
        )

        conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
        reg_loss = BboxRegression(offsets, GT_offsets)

        total_loss = w_conf * conf_loss + w_reg * reg_loss

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        if output_mode == 'loss':
            return total_loss
        else:
            return total_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, anchors.numel()

    def inference(self, images, thresh=0.5, nms_thresh=0.7, mode='RPN'):
        """
        Inference-time forward pass for the Region Proposal Network.
        """
        assert mode in ('RPN', 'FasterRCNN'), 'invalid inference mode!'

        features, final_conf_probs, final_proposals = None, [], []

        ##############################################################################
        # TODO: Predicting the RPN proposal coordinates `final_proposals` and        #
        # confidence scores `final_conf_probs`.                                      #
        # i) Extract image features                                                  #
        # ii) Generate grid and anchors                                              #
        # iii) Compute conf_scores, offsets from proposal module                     #
        # iv) Convert offsets to proposals                                           #
        # v) Apply thresholding and NMS                                              #
        ##############################################################################

        features = self.feat_extractor(images)

        grid = GenerateGrid(features)
        anchors = GenerateAnchor(self.anchor_list, grid)

        conf_scores, offsets = self.prop_module(features)

        proposals = GenerateProposal(anchors, offsets)

        B = conf_scores.shape[0] 

        for i in range(B):
            conf_probs = torch.sigmoid(conf_scores[i])
            keep = conf_probs > thresh
            filtered_proposals = proposals[i][keep]
            filtered_conf_probs = conf_probs[keep]

            if filtered_proposals.numel() == 0:
                final_proposals.append(torch.empty((0, 4), device=images.device))
                final_conf_probs.append(torch.empty((0,), device=images.device))
                continue

            keep_idx = torchvision.ops.nms(filtered_proposals, filtered_conf_probs, nms_thresh)

            final_proposals.append(filtered_proposals[keep_idx])
            final_conf_probs.append(filtered_conf_probs[keep_idx])

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        if mode == 'RPN':
            features = [torch.zeros_like(i) for i in final_conf_probs]  # dummy class
        return final_proposals, final_conf_probs, features

RPNSolver = DetectionSolver # the same solver as in YOLO

# monitor the training loss
num_sample = 10
small_dataset = torch.utils.data.Subset(train_dataset, torch.linspace(0, len(train_dataset)-1, steps=num_sample).long())
small_train_loader = pascal_voc2007_loader(small_dataset, 10) # a new loader

for lr in [1e-3]:
  print('lr: ', lr)
  rpn = RPN()
  RPNSolver(rpn, small_train_loader, learning_rate=lr, num_epochs=200)
  
  RPNInference = DetectionInference
  
  # visualize the output from the overfitted model on small dataset
# the bounding boxes should be really accurate
# ignore the dummy object class (in blue) as RPN does not output class!
RPNInference(rpn, small_train_loader, small_dataset, idx_to_class, thresh=0.8, nms_thresh=0.3)

class TwoStageDetector(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20,
                 roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
        super().__init__()

        assert num_classes != 0
        self.num_classes = num_classes
        self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h

        ##############################################################################
        # TODO: Declare your RPN and the region classification layer (in Fast R-CNN).#
        # The region classification layer is a sequential module with a Linear layer,#
        # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
        # Linear layer that predicts classification scores for each proposal.        #
        # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
        # hidden_dim -> num_classes.                                                 #
        ##############################################################################

        self.rpn = RPN()

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(p=drop_ratio),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, images, bboxes):
        """
        Training-time forward pass for our two-stage Faster R-CNN detector.
        """
        total_loss = None

        ##############################################################################
        # TODO: Implement the forward pass of TwoStageDetector.                      #
        # i) RPN: extract image features, generate grid/anchors, proposals.          #
        # ii) Perform RoI Align on proposals and meanpool the feature spatially.     #
        # iii) Pass the RoI feature through the classifier to obtain class scores.   #
        # iv) Compute cross entropy loss (cls_loss) between predicted class_prob     #
        #     and reference GT_class. Hint: Use F.cross_entropy loss.                #
        # v) Compute total_loss = rpn_loss + cls_loss.                               #
        ##############################################################################
        rpn_loss, conf_scores, proposals, features, GT_class, pos_anchor_idx, _ = \
            self.rpn(images, bboxes, output_mode='all')

        roi_features = torchvision.ops.roi_align(features, proposals, 
                                                 (self.roi_output_w, self.roi_output_h))
        roi_features = roi_features.mean(dim=(2, 3))  # Mean pool spatial dimensions

        class_scores = self.classifier(roi_features)

        cls_loss = F.cross_entropy(class_scores, GT_class)

        total_loss = rpn_loss + cls_loss

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return total_loss

    def inference(self, images, thresh=0.5, nms_thresh=0.7):
        """
        Inference-time forward pass for our two-stage Faster R-CNN detector.
        """
        final_proposals, final_conf_probs, final_class = [], [], []

        ##############################################################################
        # TODO: Predicting the final proposal coordinates `final_proposals`,        #
        # confidence scores `final_conf_probs`, and the class index `final_class`.  #
        # The overall steps are similar to the forward pass but now you do not need #
        # to decide the activated nor negative anchors.                             #
        # HINT: Use the RPN inference function to perform thresholding and NMS, and #
        # to compute final_proposals and final_conf_probs. Use the predicted class  #
        # probabilities from the second-stage network to compute final_class.       #
        ##############################################################################

        proposals, conf_probs, features = self.rpn.inference(images, thresh, nms_thresh, mode='FasterRCNN')

        B = len(proposals)  

        for i in range(B):
            if proposals[i].numel() == 0:
                final_proposals.append(torch.empty((0, 4), device=images.device))
                final_conf_probs.append(torch.empty((0,), device=images.device))
                final_class.append(torch.empty((0,), dtype=torch.int64, device=images.device))
                continue

            roi_features = torchvision.ops.roi_align(features, [proposals[i]], 
                                                     (self.roi_output_w, self.roi_output_h))
            roi_features = roi_features.mean(dim=(2, 3)) 

            class_scores = self.classifier(roi_features)

            class_prob, class_idx = class_scores.softmax(dim=-1).max(dim=-1)

            final_proposals.append(proposals[i])
            final_conf_probs.append(class_prob)
            final_class.append(class_idx)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return final_proposals, final_conf_probs, final_class

# monitor the training loss

lr = 1e-3
detector = TwoStageDetector()
DetectionSolver(detector, small_train_loader, learning_rate=lr, num_epochs=200)

# visualize the output from the overfitted model on small dataset
# the bounding boxes should be really accurate
DetectionInference(detector, small_train_loader, small_dataset, idx_to_class, thresh=0.8, nms_thresh=0.3)

# monitor the training loss
train_loader = pascal_voc2007_loader(train_dataset, 100) # a new loader

num_epochs = 50
lr = 5e-3
frcnn_detector = TwoStageDetector()
DetectionSolver(frcnn_detector, train_loader, learning_rate=lr, num_epochs=num_epochs)

# (optional) load/save checkpoint
# torch.save(frcnn_detector.state_dict(), 'frcnn_detector.pt') # uncomment to save your checkpoint
# frcnn_detector.load_state_dict(torch.load('frcnn_detector.pt')) # uncomment to load your previous checkpoint

# visualize the same output from the model trained on the entire training set
# some bounding boxes might not make sense
DetectionInference(frcnn_detector, small_train_loader, small_dataset, idx_to_class)