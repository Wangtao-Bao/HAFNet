import threading

import numpy as np
import torch
from skimage import measure



def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    predict = (output > score_thresh).float()
    if len(target.shape) == 3:
        print('？？？？')  # 加一个维度 使得target与 output的size一致
        target = target.unsqueeze(dim=0)
        # target = np.expand_dims(target.float(), axis=1)
        target.to('cuda', torch.float)

    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    # 现在predict中高于阈值的部分为全1矩阵   target是GT

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()  # 对的预测为对的
    fp = (predict * ((predict != target).float())).sum()  # 错的预测为对的 虚警像素数
    tn = ((1 - predict) * ((predict == target).float())).sum()  # 错的预测为错的
    fn = (((predict != target).float()) * (1 - predict)).sum()  # 对的预测为错的
    pos = tp + fn  # 标签中 阳性的个数
    neg = fp + tn  # 标签中 阴性的个数
    class_pos = tp + fp  # 检测出的个数

    return tp, pos, fp, neg, class_pos


class SamplewiseSigmoidMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.lock = threading.Lock()
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """

        def evaluate_worker(self, label, pred):
            inter_arr, union_arr = batch_intersection_union_n(
                pred, label, self.nclass, self.score_thresh)
            with self.lock:
                self.total_inter = np.append(self.total_inter, inter_arr)
                self.total_union = np.append(self.total_union, union_arr)

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        nIoU = IoU.mean()
        return nIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])


def batch_intersection_union_n(output, target, nclass, score_thresh):
    """nIoU"""
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    outputnp = output.detach().cpu().numpy()
    # outputsig = F.sigmoid(output).detach().cpu().numpy()
    # outputsig = nd.sigmoid(output).asnumpy()
    predict = (outputnp > 0.5).astype('int64')
    # predict = predict.detach().cpu().numpy()
    # predict = (output.asnumpy() > 0).astype('int64') # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64')  # T
    elif len(target.shape) == 4:
        target = target.cpu().numpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target)  # TP  交集

    num_sample = intersection.shape[0]
    area_inter_arr = np.zeros(num_sample)
    area_pred_arr = np.zeros(num_sample)
    area_lab_arr = np.zeros(num_sample)
    area_union_arr = np.zeros(num_sample)
    for b in range(num_sample):
        # areas of intersection and union
        area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
        area_inter_arr[b] = area_inter

        area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
        area_pred_arr[b] = area_pred

        area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
        area_lab_arr[b] = area_lab

        area_union = area_pred + area_lab - area_inter
        area_union_arr[b] = area_union

        assert (area_inter <= area_union).all(), \
            "Intersection area should be smaller than Union area"

    return area_inter_arr, area_union_arr


class ROCMetric05():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):
        # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        # nclass :有几个类别 红外弱小目标检测只有一个类别
        super(ROCMetric05, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    # 网络输入的结果和标签 计算两者之前的东西
    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            # score_thresh = (iBin + 0.0) / self.bins
            score_thresh = (0.0 + iBin) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp  # 虚警像素数
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)  # tp_rates = recall = TP/(TP+FN)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)  # fp_rates =  FP/(FP+TN)
        FP = self.fp_arr / (self.neg_arr + self.pos_arr+ 0.00001)
        recall = self.tp_arr / (self.pos_arr + 0.001)  # recall = TP/(TP+FN)
        precision = self.tp_arr / (self.class_pos + 0.001)  # precision = TP/(TP+FP)
        f1_score = (2.0 * recall[5] * precision[5]) / (recall[5] + precision[5] + 0.00001)

        return tp_rates, fp_rates, recall, precision, FP, f1_score

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


class mIoU():

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)  # labeled: GT中目标的像素数目   correct:预测正确的像素数
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)  # 目标总数  直接就搞GT的连通域个数
        self.image_area_total = []  # 图像中预测的区域列表
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):  # image 与 label 之间 根据中心点 进行连通域的确定
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]  # 匹配上一个之后就 清除一个
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]  # 在image里面 但是不在label里面

        self.dismatch_pixel += np.sum(self.dismatch)  # Fa 虚警个数 像素的虚警
        # print(self.dismatch_pixel)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)  # 如果中心点之间距离在3一下 就算Pd  所以Pd 是匹配上了的目标的个数

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy())

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])