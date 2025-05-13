import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import *
from dataset import *
import time
from collections import OrderedDict
import numpy as np
import torch
from skimage import measure

from model.HAFNet import HAFNet as HAFNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument('--ROC_thr', type=int, default=10, help='num')
parser.add_argument("--model_names", default=['HAFNet'], type=list,
                    help="model_name: 'ACM', 'SCTransNet', 'DNANet', 'ISNet', 'ACMNet', 'RDIAN', 'ISTDU-Net', 'UIUNet', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['NUAA-SIRST/HAFNet_605_best.pth.tar'], type=list)
parser.add_argument("--patchSize", type=int, default=256, help="Testing patch size")
parser.add_argument("--dataset_dir", default=r'dataset', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'IRSTD-Air', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default=r'result/',
                    help="path of saved image")
parser.add_argument("--save_log", type=str, default=r'log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()


def test():
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    IOU = mIoU()
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
    eval_05 = PD_FA()
    ROC_05 = ROCMetric05(nclass=1, bins=10)

    net = HAFNet(Train=False).cuda()
    state_dict = torch.load(opt.pth_dir)
    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[6:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(tbar):
            img=img.cuda()
            gt_mask=gt_mask.cuda()
            pred = net.forward(img)
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]

            IOU.update((pred > 0.5), gt_mask)
            nIoU_metric.update(pred, gt_mask)
            ROC_05.update(pred, gt_mask)
            eval_05.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)  # 目标

            # save img
            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')


        pixAcc, mIOU = IOU.get()
        nIoU = nIoU_metric.get()
        results2 = eval_05.get()
        ture_positive_rate, false_positive_rate, recall, precision, FP, F1_score = ROC_05.get()
        print('pixAcc: %.4f| mIoU: %.4f | nIoU: %.4f | Pd: %.4f| Fa: %.4f |F1: %.4f'
              % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F1_score * 100))




if __name__ == '__main__':
    for model_name in opt.model_names:
        for dataset_name in opt.dataset_names:
            for pth_dir in opt.pth_dirs:
                # if dataset_name in pth_dir and model_name in pth_dir:
                opt.test_dataset_name = dataset_name
                opt.model_name = model_name
                opt.train_dataset_name = pth_dir.split('/')[0]
                print(pth_dir)
                print(opt.test_dataset_name)
                opt.pth_dir = opt.save_log + pth_dir
                test()
                print('\n')

