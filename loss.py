import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
torch.pi=math.pi

class FocalIoULoss(nn.Module):
    def __init__(self):
        super(FocalIoULoss, self).__init__()

    def forward(self, inputs, targets):

        [b, c, h, w] = inputs.size()

        inputs = torch.nn.Sigmoid()(inputs)
        inputs = 0.999 * (inputs - 0.5) + 0.5
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        intersection = torch.mul(inputs, targets)
        smooth = 1

        IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)

        alpha = 0.75
        gamma = 2
        num_classes = 2
        # alpha_f = torch.tensor([alpha, 1 - alpha]).cuda()
        # alpha_f = torch.tensor([alpha, 1 - alpha])
        gamma = gamma
        size_average = True

        pt = torch.exp(-BCE_loss)

        F_loss = torch.mul(((1 - pt) ** gamma), BCE_loss)

        at = targets * alpha + (1 - targets) * (1 - alpha)

        F_loss = (1 - IoU) * (F_loss) ** (IoU * 0.5 + 0.5)

        F_loss_map = at * F_loss

        F_loss_sum = F_loss_map.sum()

        #return F_loss_map, F_loss_sum
        return F_loss_sum


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            mp= nn.MaxPool2d(2, 2)
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                if i>5:
                    gt_masks=mp(gt_masks)
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss


class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            mp = nn.MaxPool2d(2, 2)
            for i in range(len(preds)):
                pred = preds[i]
                if i >5:
                    gt_masks = mp(gt_masks)
                # 确保形状一致
                if pred.size() != gt_masks.size():
                    gt_masks = gt_masks.repeat(pred.size(0), 1, 1, 1)  # 扩展批次大小

                pred =  pred.float()
                gt_masks = gt_masks.float()
                # print(pred.shape, gt_masks.shape)
                # print(pred.min().item(), pred.max().item())
                # print(gt_masks.min().item(), gt_masks.max().item())
                loss = self.bce_loss(pred, gt_masks)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            # 确保形状一致
            if pred.size() != gt_masks.size():
                gt_masks = gt_masks.repeat(pred.size(0), 1, 1, 1)  # 扩展批次大小
            loss = self.bce_loss(pred, gt_masks)
            return loss

class FindCoarseEdge(nn.Module):
    def __init__(self):
        super(FindCoarseEdge, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return torch.sigmoid(x0)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.soft_iou_loss = SoftIoULoss()
        self.edge_extractor = FindCoarseEdge()
        self.bce_loss = BCELoss()

    def forward(self, preds, gt_masks):
        # 计算 SoftIoU 损失
        iou_loss = self.soft_iou_loss(preds, gt_masks)

        # 提取预测和标签的边缘
        pred_edges = [self.edge_extractor(pred) for pred in preds] if isinstance(preds, list) or isinstance(preds, tuple) else self.edge_extractor(preds)
        gt_edges = self.edge_extractor(gt_masks)

        bce_loss = self.bce_loss(pred_edges, gt_edges)

        # 最终损失是 SoftIoU 和 BCE 损失的和
        total_loss = iou_loss * 0.5 + bce_loss
        return total_loss

class Structure_loss(nn.Module):
    def __init__(self):
        super(Structure_loss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self,pred, mask):
        # weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        # wbce = self.bce_loss(pred, mask)
        # wbce =  (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
        # inter = ((pred * mask) * weit).sum(dim=(2, 3))
        # union = ((pred + mask) * weit).sum(dim=(2, 3))
        # wiou = 1 - (inter + 1) / (union - inter + 1)
        # return (wbce + wiou).mean()
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = (inter + 1) / (union - inter + 1)
        return 1 - wiou.mean()


def structure_loss1(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = nn.BCELoss(reduction='none')(pred, mask)
    wbce =  (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()