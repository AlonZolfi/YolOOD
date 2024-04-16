# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
from scipy.stats import truncnorm

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLossOOD:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, resp_cell_offset=0):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        self.model = model
        m = de_parallel(self.model).model[-1]  # Detect() module
        self.balance = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device
        self.resp_cell_offset = torch.tensor(resp_cell_offset, device=device)

    def __call__(self, predictions, targets):  # predictions, targets
        lcls, lobj = [], []  # class loss, object loss
        tcls, indices = self.build_targets(predictions, targets)  # targets
        # Losses
        for i, pi in enumerate(predictions):  # layer index, layer predictions
            # Classification
            pobj, pcls = pi.split((1, self.nc), -1)  # obj, cls predictions

            t = torch.full(pi[..., 1:].shape, self.cn, device=self.device)

            b, gj, gi = indices[i]  # image, gridy, gridx [0, 23, 24] * N
            c = tcls[i]
            t[b, gj, gi, c] = self.cp

            pred_mask = t.sum(dim=-1) > 0
            t_masked = t[pred_mask]
            pcls_masked = pcls[pred_mask]
            lcls.append(self.BCEcls(pcls_masked, t_masked))

            # Objectness
            tobj = torch.zeros(pi.shape[:3], dtype=pi.dtype, device=self.device)  # target obj
            tobj[pred_mask] = 1.

            lobj.append(self.BCEobj(pobj.squeeze(-1), tobj))  # obj loss

        lobj = torch.matmul(torch.stack(lobj), self.balance)
        lcls = torch.matmul(torch.stack(lcls), self.balance)
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        return (lobj + lcls), torch.cat((lobj[None], lcls[None])).detach()

    def build_targets1(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        tcls, indices = [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        for i in range(self.nl):
            # anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[2, 1, 2, 1]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(n,6)
            # if nt:
            #     # Offsets
            #     # gxy = t[:, 2:4]  # grid xy
            #     # gxi = gain[[2, 3]] - gxy  # inverse
            #     # j, k = ((gxy % 1 < g) & (gxy > 1)).T
            #     # l, m = ((gxi % 1 < g) & (gxi > 1)).T
            #     # j = torch.stack((torch.ones_like(j), j, k, l, m))
            #     j = torch.ones((off.shape[0], nt), dtype=torch.bool, device=self.device)
            #     # if not self.center_only:
            #     #     j[1:] = 0
            #     t = t.repeat((off.shape[0], 1, 1))[j]
            #     offsets = (torch.zeros((nt, 2), device=self.device)[None] + off[:, None])[j]
            #     # offsets = torch.zeros_like(t[..., 2:4]) + off[:, None]
            # else:
            #     offsets = 0

            # Define
            bc, gxy, gwh = t.chunk(3, 1)  # (image, class), grid xy, grid wh
            (b, c) = bc.long().T  # image, class
            # gij = (gxy - offsets).long()
            gij = gxy.long()
            # gij = gxy.long()
            gi, gj = gij.T  # grid indices

            # Append
            # indices.append((b, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, grid indices
            # tcls.append(c)  # class
            # if self.full_obj:

            # if self.resp_cell_offset == -1:
            # gw_half, gh_half = (gwh / 2).long().T
            # half_grid_size = p[i].shape[1] // 2
            # cell_offset = torch.tensor(half_grid_size, device=self.device, dtype=torch.long) * self.resp_cell_offset
            # gw_half_min, gh_half_min = torch.min(gw_half, self.resp_cell_offset[i]), torch.min(gh_half, self.resp_cell_offset[i])
            # else:
            #     cell_offset = torch.tensor(self.resp_cell_offset, device=self.device, dtype=torch.long)
            #     gw_half, gh_half = (gwh / 2).T
            #     max_length_half = torch.max(gw_half, gh_half)
            #     gw_half_min = ((gw_half / max_length_half) * torch.min(gw_half, cell_offset)).long()
            #     gh_half_min = ((gh_half / max_length_half) * torch.min(gh_half, cell_offset)).long()
            # gw_half_min, gh_half_min = ((gwh / 2) * self.model.resp_cell_offset).long().T
            gw_half_min, gh_half_min = ((gwh / 2) * self.model.resp_cell_offset).T
            # gw_half, gh_half = (gwh / 2).T
            # gw_half_min, gh_half_min = torch.min(gw_half, self.resp_cell_offset * p[i].shape[2] / 2).long(), torch.min(gh_half, self.resp_cell_offset * p[i].shape[1] / 2).long()
            # bboxes = torch.stack([(gi - gw_half_min).data.clamp_(min=0),
            #                       (gj - gh_half_min).data.clamp_(min=0),
            #                       (gi + gw_half_min).data.clamp_(max=p[i].shape[2]),
            #                       (gj + gh_half_min).data.clamp_(max=p[i].shape[1])], dim=1)
            bboxes = torch.stack([(gi - gw_half_min),
                                  (gj - gh_half_min),
                                  (gi + gw_half_min),
                                  (gj + gh_half_min)], dim=1)
            # points = torch.cartesian_prod(torch.arange(0, p[i].shape[2], device=self.device),
            #                               torch.arange(0, p[i].shape[1], device=self.device))
            points = torch.cartesian_prod(torch.arange(0, p[i].shape[2], device=self.device, dtype=torch.float32, requires_grad=True),
                                          torch.arange(0, p[i].shape[1], device=self.device, dtype=torch.float32, requires_grad=True))
            points = points.unsqueeze(1).repeat(1, bboxes.shape[0], 1)
            # c11 = (points[:, :, 0] >= bboxes[:, 0])
            c1 = torch.nn.functional.relu(points[:, :, 0] - bboxes[:, 0])  # & (points[:, :, 0] + self.resp_cell_offset >= gi)  # x_i >= x_left
            # c22 = (points[:, :, 0] <= bboxes[:, 2])  # & (points[:, :, 0] - self.resp_cell_offset <= gi)  # x_i <= x_right
            c2 = torch.nn.functional.relu(-(points[:, :, 0] - bboxes[:, 2]))  # & (points[:, :, 0] + self.resp_cell_offset >= gi)  # x_i >= x_left
            # c33 = (points[:, :, 1] >= bboxes[:, 1])  # & (points[:, :, 0] + self.resp_cell_offset >= gj)  # y_i >= y_left
            c3 = torch.nn.functional.relu(points[:, :, 1] - bboxes[:, 1])  # & (points[:, :, 0] + self.resp_cell_offset >= gi)  # x_i >= x_left
            # c44 = (points[:, :, 1] <= bboxes[:, 3])  # & (points[:, :, 0] - self.resp_cell_offset <= gj)  # y_i <= y_right
            c4 = torch.nn.functional.relu(-(points[:, :, 1] - bboxes[:, 3]))  # & (points[:, :, 0] + self.resp_cell_offset >= gi)  # x_i >= x_left
            mask = torch.nn.functional.relu(c1.clamp_(max=1.).ceil() + c2.clamp_(max=1.).ceil() + c3.clamp_(max=1.).ceil() + c4.clamp_(max=1.).ceil() - 3)
            mask = (c1 & c2 & c3 & c4).transpose(0, 1).view(-1, p[i].shape[2], p[i].shape[1])
            bbox_inner_indices = mask.nonzero()

            b = torch.index_select(b, 0, bbox_inner_indices[:, 0])
            c = torch.index_select(c, 0, bbox_inner_indices[:, 0])

            gi, gj = bbox_inner_indices[:, 1], bbox_inner_indices[:, 2]

            gt = torch.stack([b, gj.data.clamp_(0, gain[3] - 1), gi.data.clamp_(0, gain[2] - 1), c], 1)
            gt_unique = torch.unique(gt, dim=0)
            b, gj, gi, c = gt_unique.chunk(4, 1)
            indices.append((b[:, 0], gj[:, 0], gi[:, 0]))  # image, grid indices
            tcls.append(c[:, 0])  # class
        return tcls, indices

    def build_targets(self, p, targets):  # YOLOOD ICCV
        tcls, indices = [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain
        for i in range(self.nl):
            gain[2:6] = torch.tensor(p[i].shape)[[2, 1, 2, 1]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(n,6)

            # Define
            bc, gxy, gwh = t.chunk(3, 1)  # (image, class), grid xy, grid wh
            (b, c) = bc.long().T  # image, class
            gij = gxy.long()
            gi, gj = gij.T  # grid indices

            gw_half_min, gh_half_min = ((gwh / 2) * self.resp_cell_offset[i]).long().T
            bboxes = torch.stack([(gi - gw_half_min).data.clamp_(min=0),
                                  (gj - gh_half_min).data.clamp_(min=0),
                                  (gi + gw_half_min).data.clamp_(max=p[i].shape[2]),
                                  (gj + gh_half_min).data.clamp_(max=p[i].shape[1])], dim=1)

            points = torch.cartesian_prod(torch.arange(0, p[i].shape[2], device=self.device),
                                          torch.arange(0, p[i].shape[1], device=self.device))
            points = points.unsqueeze(1).repeat(1, bboxes.shape[0], 1)
            c1 = (points[:, :, 0] >= bboxes[:, 0])
            c2 = (points[:, :, 0] <= bboxes[:, 2])
            c3 = (points[:, :, 1] >= bboxes[:, 1])
            c4 = (points[:, :, 1] <= bboxes[:, 3])
            mask = (c1 & c2 & c3 & c4).transpose(0, 1).view(-1, p[i].shape[2], p[i].shape[1])
            bbox_inner_indices = mask.nonzero()

            b = torch.index_select(b, 0, bbox_inner_indices[:, 0])
            c = torch.index_select(c, 0, bbox_inner_indices[:, 0])

            gi, gj = bbox_inner_indices[:, 1], bbox_inner_indices[:, 2]

            gt = torch.stack([b, gj.data.clamp_(0, gain[3] - 1), gi.data.clamp_(0, gain[2] - 1), c], 1)
            gt_unique = torch.unique(gt, dim=0)
            b, gj, gi, c = gt_unique.chunk(4, 1)
            indices.append((b[:, 0], gj[:, 0], gi[:, 0]))  # image, grid indices
            tcls.append(c[:, 0])  # class

        return tcls, indices

    def build_targets2(self, p, targets): # assigning object to a single head only
        tcls, indices = [None] * self.nl, [None] * self.nl
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain
        # resp_cell_offset = torch.rand(1, device=self.device)
        for i in reversed(range(self.nl)):
            if targets.shape[0] == 0:
                continue

            gain[2:6] = torch.tensor(p[i].shape)[[2, 1, 2, 1]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(n,6)

            # Define
            bc, gxy, gwh = t.chunk(3, 1)  # (image, class), grid xy, grid wh
            (b, c) = bc.long().T  # image, class
            gij = gxy.long()
            gi, gj = gij.T  # grid indices

            gw_half_min, gh_half_min = (gwh / 2).long().T
            bboxes = torch.stack([(gi - gw_half_min).data.clamp_(min=0),
                                  (gj - gh_half_min).data.clamp_(min=0),
                                  (gi + gw_half_min).data.clamp_(max=p[i].shape[2]),
                                  (gj + gh_half_min).data.clamp_(max=p[i].shape[1])], dim=1)
            bboxes_size = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
            bigger_than_one = bboxes_size > 1
            gw_half_min, gh_half_min = ((gwh / 2) * self.resp_cell_offset[i]).long().T
            # resp_cell_offset = torch.rand(1, device=self.device)
            # resp_cell_offset = torch.tensor(truncnorm.rvs(0, 1, size=1, loc=self.loc, scale=self.scale), device=self.device)
            # gw_half_min, gh_half_min = ((gwh / 2) * resp_cell_offset).long().T
            bboxes = torch.stack([(gi - gw_half_min).data.clamp_(min=0),
                                  (gj - gh_half_min).data.clamp_(min=0),
                                  (gi + gw_half_min).data.clamp_(max=p[i].shape[2]),
                                  (gj + gh_half_min).data.clamp_(max=p[i].shape[1])], dim=1)
            bboxes = bboxes[bigger_than_one]

            if bboxes.shape[0] == 0:
                continue

            targets = targets[~bigger_than_one]
            points = torch.cartesian_prod(torch.arange(0, p[i].shape[2], device=self.device),
                                          torch.arange(0, p[i].shape[1], device=self.device))
            points = points.unsqueeze(1).repeat(1, bboxes.shape[0], 1)
            c1 = (points[:, :, 0] >= bboxes[:, 0])
            c2 = (points[:, :, 0] <= bboxes[:, 2])
            c3 = (points[:, :, 1] >= bboxes[:, 1])
            c4 = (points[:, :, 1] <= bboxes[:, 3])
            mask = (c1 & c2 & c3 & c4).transpose(0, 1).view(-1, p[i].shape[2], p[i].shape[1])
            bbox_inner_indices = mask.nonzero()

            b = torch.index_select(b, 0, bbox_inner_indices[:, 0])
            c = torch.index_select(c, 0, bbox_inner_indices[:, 0])

            gi, gj = bbox_inner_indices[:, 1], bbox_inner_indices[:, 2]

            gt = torch.stack([b, gj.data.clamp_(0, gain[3] - 1), gi.data.clamp_(0, gain[2] - 1), c], 1)
            gt_unique = torch.unique(gt, dim=0)
            b, gj, gi, c = gt_unique.chunk(4, 1)
            indices[i] = (b[:, 0], gj[:, 0], gi[:, 0])  # image, grid indices
            tcls[i] = c[:, 0]  # class

        return tcls, indices


class ComputeLoss2:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        pos_weights_file = '/dt/shabtaia/dt-fujitsu/datasets/multi_label/pascal_voc/'
        pos_weight = torch.load(pos_weights_file + 'pos_weights.pt', map_location=device)
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Define criteria
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)  # object loss
        p_obj, p_cls = p[0], p[1]
        tcls, indices = self.build_targets(p_obj, targets)  # targets

        t = torch.full_like(p_cls, self.cn)
        t_indices = torch.stack([indices[0][0], tcls[0]], 1)
        t[t_indices[:, 0], t_indices[:, 1]] = self.cp
        lcls = self.BCEcls(p_cls, t).unsqueeze(-1)

        # Losses
        for i, pi in enumerate(p_obj):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, gridy, gridx
            tobj = torch.zeros(pi.shape[:3], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                tobj[b, gj, gi] = 1.

            obji = self.BCEobj(pi[..., 0], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lobj + lcls) * bs, torch.cat((lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = 1, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            # anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[2, 1, 2, 1]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                # r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[0]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            # anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, indices