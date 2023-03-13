import torch
from collections import OrderedDict
from model.IANet import Model
import os
import numpy as np
import cv2
import IOU
import datetime

EPSILON = 1e-8
p = OrderedDict()

p['lr_bone'] = 5e-4  # Learning rate
p['lr_branch'] = 5e-3
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [14]
showEvery = 50

CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)


def structure_loss(pred, mask):
    bce = CE(pred, mask)

    iou = IOU(torch.nn.Sigmoid()(pred), mask)

    return bce + iou


class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):

        self.optimizer_bone = None
        self.net_bone = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold

        self.build_model()

        self.cuda = True

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # 返回一个tensor变量内所有元素个数
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        print('mode: {}'.format(self.config.mode))
        print('------------------------------------------')
        self.net_bone = Model(3, mode=self.config.mode)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()

        if self.config.mode == 'train':
            if self.config.model_path != '':
                print('load model……')
                assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
                self.net_bone.load_pretrain_model(self.config.model_path)
        else:
            assert (self.config.model_path != ''), ('Test mode, please import pretrained model path!')
            assert (os.path.exists(self.config.model_path)), ('please import correct pretrained model path!')
            print('load model……')
            self.net_bone.load_pretrain_model(self.config.model_path)

        base, head = [], []
        for name, param in self.net_bone.named_parameters():
            if 'ImageBone' in name or 'FlowBone' in name:
                base.append(param)
            else:
                head.append(param)

        self.optimizer_bone = torch.optim.SGD([{'params': base}, {'params': head}],
                                              lr=p['lr_bone'], momentum=p['momentum'],
                                              weight_decay=p['wd'], nesterov=True)
        print('------------------------------------------')
        self.print_network(self.net_bone, 'IANet')
        print('------------------------------------------')

    def test(self):

        self.net_bone.eval()
        if not os.path.exists(self.save_fold):
            os.makedirs(self.save_fold)
        for i, data_batch in enumerate(self.test_loader):
            print("progress {}/{}\n".format(i + 1, len(self.test_loader)))
            image, flow, name, split, size = data_batch['image'], data_batch['flow'], data_batch['name'], data_batch[
                'split'], data_batch['size']
            dataset = data_batch['dataset']

            if self.config.cuda:
                image, flow = image.cuda(), flow.cuda()
            with torch.no_grad():

                pre = self.net_bone(image, flow)

                for i in range(self.config.test_batch_size):
                    presavefold = os.path.join(self.save_fold, dataset[i], split[i])
                    if not os.path.exists(presavefold):
                        os.makedirs(presavefold)
                    pre1 = torch.nn.Sigmoid()(pre[0][i])
                    pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1) + 1e-8)
                    pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
                    pre1 = cv2.resize(pre1, (size[0][1], size[0][0]))
                    cv2.imwrite(presavefold + '/' + name[i], pre1)

    def train(self, train_loader):

        # 一个epoch中训练iter_num个batch
        iter_num = len(train_loader.dataset) // self.config.batch_size
        for epoch in range(self.config.epoch):
            self.optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
            self.optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(train_loader):
                image, label, flow = data_batch['image'], data_batch['label'], data_batch['flow']
                if image.size()[2:] != label.size()[2:]:
                    print("Skip this batch")
                    continue
                if self.config.cuda:
                    image, label, flow = image.cuda(), label.cuda(), flow.cuda()

                pred, course_img, course_flo, out1r, out2r, out3r, out4r = self.net_bone(
                    image, flow)

                pre_loss = structure_loss(pred, label)

                img_loss = structure_loss(course_img, label)
                flo_loss = structure_loss(course_flo, label)

                loss1r = structure_loss(out1r, label)
                loss2r = structure_loss(out2r, label)
                loss3r = structure_loss(out3r, label)
                loss4r = structure_loss(out4r, label)
                # loss = pre_loss + img_loss + flo_loss + 0.4 * (loss1r + loss2r) + 0.8 * (loss3r + loss4r)
                loss = pre_loss + img_loss + flo_loss + loss1r + loss2r + loss3r + loss4r

                self.optimizer_bone.zero_grad()
                loss.backward()
                self.optimizer_bone.step()

                if i % showEvery == 0:
                    print(
                        '%s || epoch: [%2d/%2d], iter: [%5d/%5d] || pre_loss : %10.4f || sum : %10.4f' % (
                            datetime.datetime.now(), epoch, self.config.epoch, i, iter_num,
                            pre_loss.data, loss.data))

                    print('Learning rate: ' + str(self.optimizer_bone.param_groups[0]['lr']))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           '%s/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if epoch in lr_decay_epoch:
                p['lr_bone'] = p['lr_bone'] * 0.1
                p['lr_branch'] = p['lr_branch'] * 0.1

        torch.save(self.net_bone.state_dict(), '%s/final_bone.pth' % self.config.save_fold)
