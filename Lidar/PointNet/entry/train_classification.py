import os
import sys
import argparse
import random
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from Dataset.modelnet40.data_provider import ModelNetDataset, get_dataset_path
from Lidar.PointNet.model.PointNet import PointNetCls, feature_transform_regularizer

CURRENT_PATH = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=10)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
opt.dataset_type = "modelnet40"
opt.dataset = get_dataset_path()
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print("输入参数：", opt)



if __name__ == '__main__':
    # 确认输入数据集
    if opt.dataset_type == 'shapenet':
        pass
        # train_dataset = ShapeNetDataset(root=opt.dataset, classification=True, npoints=opt.num_points)
        # test_dataset = ShapeNetDataset(root=opt.dataset, classification=True, split='test', npoints=opt.num_points, data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        train_dataset = ModelNetDataset(root=opt.dataset, npoints=opt.num_points, split='train')
        test_dataset = ModelNetDataset(root=opt.dataset, split='test', npoints=opt.num_points, data_augmentation=False)
    else:
        exit('wrong dataset type')

    tradataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    print("训练集数据量：", len(train_dataset))
    print("测试集数据量：", len(test_dataset))
    num_classes = len(train_dataset.classes)
    print('classes：', num_classes)
    num_batch = len(train_dataset) / opt.batchSize
    print('batch size：', num_batch)

    # 加载分类网络
    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
    # 可以手动指定
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
    classifier.cuda()

    # 设置优化算法
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 逐epoch训练
    for epoch in range(opt.nepoch):
        # 优化器学习率步进
        scheduler.step()
        for i, data in enumerate(tradataloader, 0):
            # 获取当前batch数据，cp到cuda上
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            # 清空优化器缓存
            optimizer.zero_grad()
            # 前向传播一次， 训练模式
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            # 反向传播一次
            loss.backward()
            # 优化参数步进
            optimizer.step()
            # 计算正确率
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            # 没训练10次就测试一次
            if i % 10 == 0:
                # 准备测试数据
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                # 测试模式
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                # 计算正确率
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))
        # 每个epoch save一次
        torch.save(classifier.state_dict(), 'cls_model_%d.pth' % (epoch))

    # epoch训练完输出一次总的测试结果
    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
