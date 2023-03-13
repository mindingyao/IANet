import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_aspp import ResNet_ASPP_FLOW, ResNet_ASPP_Image

class GWG(nn.Module):
    def __init__(self):
        super(GWG, self).__init__()
        self.FC = nn.Sequential(nn.Conv2d(640, 640, 1), nn.BatchNorm2d(640), nn.ReLU(inplace=True))
        self.att_1 = nn.Sequential(nn.Conv2d(640, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())
        self.att_2 = nn.Sequential(nn.Conv2d(640, 128, 1), nn.BatchNorm2d(128), nn.Sigmoid())
        self.att_3 = nn.Sequential(nn.Conv2d(640, 256, 1), nn.BatchNorm2d(256), nn.Sigmoid())
        self.att_4 = nn.Sequential(nn.Conv2d(640, 512, 1), nn.BatchNorm2d(512), nn.Sigmoid())
        self.att_5 = nn.Sequential(nn.Conv2d(640, 256, 1), nn.BatchNorm2d(256), nn.Sigmoid())

    def forward(self, a, b):
        N, _, _, _ = a.shape
        a = torch.mean(a.view(N, 320, -1), dim=2).view(N, 320, 1, 1)  # [N, 320, 1, 1]
        b = torch.mean(b.view(N, 320, -1), dim=2).view(N, 320, 1, 1)  # [N, 320, 1, 1]
        feat = torch.cat([a, b], dim=1)  # [N, 640, 1, 1]
        feat = self.FC(feat)
        return [self.att_1(feat), self.att_2(feat), self.att_3(feat), self.att_4(feat), self.att_5(feat)]



class LME(nn.Module):
    def __init__(self, in_channel=64):
        super(LME, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channel)

        self.conv3 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.conv4 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(in_channel)

        self.conv5 = nn.Conv2d(in_channel, in_channel, 1)
        self.bn5 = nn.BatchNorm2d(in_channel)
        self.conv6 = nn.Conv2d(in_channel, in_channel, 1)
        self.bn6 = nn.BatchNorm2d(in_channel)

    def forward(self, rgb, flow):
        fuse = F.relu(self.bn1(self.conv1(rgb)), inplace=True) * F.relu(self.bn2(self.conv2(flow)), inplace=True)
        sub1 = F.relu(self.bn3(self.conv3(flow - rgb)), inplace=True)
        sub2 = F.relu(self.bn4(self.conv4(rgb - flow)), inplace=True)

        rgb_ = F.relu(self.bn5(self.conv5(fuse + sub1)), inplace=True)
        flow_ = F.relu(self.bn6(self.conv6(fuse + sub2)), inplace=True)

        rgb = rgb + rgb_
        flow = flow + flow_

        return rgb, flow


class PA(nn.Module):
    def __init__(self, out_dim):
        super(PA, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_dim), act_fn, )
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_dim), act_fn, )
        self.layer_31 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_dim), act_fn, )
        self.layer_41 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_dim), act_fn, )

        self.layer_final_rgb = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(out_dim), act_fn, )
        self.layer_final_flow = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                             nn.BatchNorm2d(out_dim), act_fn, )

        self.cs = ChannelScaling(out_dim)

    def forward(self, rgb, flow, up):
        ####

        x_rgb = self.layer_11(rgb)
        x_flow = self.layer_21(flow)

        x_up_rgb = F.interpolate(self.layer_31(up), rgb.shape[2:], mode='bilinear', align_corners=True)
        x_up_flow = F.interpolate(self.layer_41(up), rgb.shape[2:], mode='bilinear', align_corners=True)

        rgb_mul = torch.mul(x_rgb, x_up_rgb)

        x_rgb_in1 = torch.reshape(x_rgb, [x_rgb.shape[0], 1, x_rgb.shape[1], x_rgb.shape[2], x_rgb.shape[3]])
        x_rgb_in2 = torch.reshape(x_up_rgb, [x_up_rgb.shape[0], 1, x_up_rgb.shape[1], x_up_rgb.shape[2], x_up_rgb.shape[3]])

        x_rgb_cat = torch.cat((x_rgb_in1, x_rgb_in2), dim=1)
        rgb_max = x_rgb_cat.max(dim=1)[0]
        rgb_out = self.layer_final_rgb(torch.cat((rgb_mul, rgb_max), dim=1))

        ####

        flow_mul = torch.mul(x_flow, x_up_flow)

        x_flow_in1 = torch.reshape(x_flow, [x_flow.shape[0], 1, x_flow.shape[1], x_flow.shape[2], x_flow.shape[3]])
        x_flow_in2 = torch.reshape(x_up_flow, [x_up_flow.shape[0], 1, x_up_flow.shape[1], x_up_flow.shape[2], x_up_flow.shape[3]])

        x_flow_cat = torch.cat((x_flow_in1, x_flow_in2), dim=1)
        flow_max = x_flow_cat.max(dim=1)[0]
        flow_out = self.layer_final_flow(torch.cat((flow_mul, flow_max), dim=1))

        return self.cs(rgb_out, flow_out, F.interpolate(up, rgb_out.shape[2:], mode='bilinear', align_corners=True))


class PA_last(nn.Module):
    def __init__(self, out_dim):
        super(PA_last, self).__init__()
        self.cs = ChannelScaling_last(out_dim)

    def forward(self, rgb, flow):
        return self.cs(rgb, flow)


class ChannelScaling(nn.Module):
    def __init__(self, in_planes):
        super(ChannelScaling, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes * 3, in_planes, 1)
        self.att1 = nn.Conv2d(in_planes, in_planes, 1)
        self.att2 = nn.Conv2d(in_planes, in_planes, 1)
        self.att3 = nn.Conv2d(in_planes, in_planes, 1)
        self.in_planes = in_planes

    def forward(self, img, flow, pre):
        pre = F.interpolate(pre, img.shape[2:], mode='bilinear', align_corners=True)
        f = self.fc1(self.avg_pool(torch.cat([img, flow, pre], 1)))  # B 3C 1 1
        att1 = self.att1(f)
        att2 = self.att2(f)
        att3 = self.att3(f)
        att = self.softmax(torch.cat([att1, att2, att3], dim=2))
        return img * att[:, :, :1, :] + flow * att[:, :, 1:2, :, ] + pre * att[:, :, 2:, :, ]


# last layer only has two input features
class ChannelScaling_last(nn.Module):
    def __init__(self, in_planes):
        super(ChannelScaling_last, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes * 2, in_planes * 2, 1)
        self.att1 = nn.Conv2d(in_planes * 2, in_planes, 1)
        self.att2 = nn.Conv2d(in_planes * 2, in_planes, 1)
        self.in_planes = in_planes

    def forward(self, img, flow):
        f = self.fc1(self.avg_pool(torch.cat([img, flow], 1)))  # B 2C 1 1
        att1 = self.att1(f)
        att2 = self.att2(f)
        att = self.softmax(torch.cat([att1, att2], dim=2))
        return img * att[:, :, :1, :] + flow * att[:, :, 1:, :]


class Model(nn.Module):
    def __init__(self, input_channel, mode):
        super(Model, self).__init__()
        self.ImageBone = ResNet_ASPP_Image(input_channel, 1, 16, 'resnet34')
        self.FlowBone = ResNet_ASPP_FLOW(input_channel, 1, 16, 'resnet34')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.GWG = GWG()

        self.conv1_i = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2_i = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3_i = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4_i = nn.Sequential(nn.Conv2d(512, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.convaspp_i = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.convp_i = nn.Sequential(nn.Conv2d(320, 5, 3, 1, 1), nn.BatchNorm2d(5), nn.ReLU())

        self.conv1_f = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2_f = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3_f = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4_f = nn.Sequential(nn.Conv2d(512, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.convaspp_f = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.convp_f = nn.Sequential(nn.Conv2d(320, 5, 3, 1, 1), nn.BatchNorm2d(5), nn.ReLU())

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.blockaspp = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.lme1 = LME(in_channel=64)
        self.lme2 = LME(in_channel=128)
        self.lme3 = LME(in_channel=256)
        self.lme4 = LME(in_channel=512)
        self.lme5 = LME(in_channel=256)

        self.pa5 = PA_last(256)
        self.pa4 = PA(512)
        self.pa3 = PA(256)
        self.pa2 = PA(128)
        self.pa1 = PA(64)

        self.pred_head1 = nn.Conv2d(64, 1, 1)
        self.pred_head2 = nn.Conv2d(128, 1, 1)
        self.pred_head3 = nn.Conv2d(256, 1, 1)
        self.pred_head4 = nn.Conv2d(512, 1, 1)
        self.pred_head5 = nn.Conv2d(256, 1, 1)

        self.last_conv_rgb = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if mode == 'train':
            self.ImageBone.backbone_features._load_pretrained_model('./model/resnet/pre_train/resnet34-333f7ec4.pth')
            self.FlowBone.backbone_features._load_pretrained_model('./model/resnet/pre_train/resnet34-333f7ec4.pth')

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def decoder_attention_module(self, img_feat, flow_map):
        final_feat = img_feat * flow_map + img_feat
        return final_feat, flow_map

    def forward(self, image, flow):
        img_layer4_feat, img_layer1_feat, img_conv1_feat, img_layer2_feat, img_layer3_feat, img_aspp_feat, course_img = self.ImageBone(
            image)

        flow_layer4_feat, flow_layer1_feat, flow_conv1_feat, flow_layer2_feat, flow_layer3_feat, flow_aspp_feat, course_flowmap = self.FlowBone(
            flow)

        img_layer1_feat, flow_layer1_feat = self.lme1(img_layer1_feat, flow_layer1_feat)
        img_layer2_feat, flow_layer2_feat = self.lme2(img_layer2_feat, flow_layer2_feat)
        img_layer3_feat, flow_layer3_feat = self.lme3(img_layer3_feat, flow_layer3_feat)
        img_layer4_feat, flow_layer4_feat = self.lme4(img_layer4_feat, flow_layer4_feat)
        img_aspp_feat, flow_aspp_feat = self.lme5(img_aspp_feat, flow_aspp_feat)

        i1 = self.conv1_i(img_layer1_feat)
        i2 = self.conv2_i(img_layer2_feat)
        i2 = F.interpolate(i2, i1.shape[2:], mode='bilinear', align_corners=True)
        i3 = self.conv3_i(img_layer3_feat)
        i3 = F.interpolate(i3, i1.shape[2:], mode='bilinear', align_corners=True)
        i4 = self.conv4_i(img_layer4_feat)
        i4 = F.interpolate(i4, i1.shape[2:], mode='bilinear', align_corners=True)
        iaspp = self.convaspp_i(img_aspp_feat)
        iaspp = F.interpolate(iaspp, i1.shape[2:], mode='bilinear', align_corners=True)
        i = torch.cat([i1, i2, i3, i4, iaspp], dim=1)

        f1 = self.conv1_f(flow_layer1_feat)
        f2 = self.conv2_f(flow_layer2_feat)
        f2 = F.interpolate(f2, f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = self.conv3_f(flow_layer3_feat)
        f3 = F.interpolate(f3, f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = self.conv4_f(flow_layer4_feat)
        f4 = F.interpolate(f4, f1.shape[2:], mode='bilinear', align_corners=True)
        faspp = self.convaspp_f(flow_aspp_feat)
        faspp = F.interpolate(faspp, f1.shape[2:], mode='bilinear', align_corners=True)
        f = torch.cat([f1, f2, f3, f4, faspp], dim=1)

        atts = self.GWG(i, f)

        course_flowmap = F.interpolate(course_flowmap, img_aspp_feat.size()[2:], mode='bilinear', align_corners=True)
        course_img, course_flowmap = self.decoder_attention_module(course_img, course_flowmap)
        course_imagemap = self.last_conv_rgb(course_img)

        feaaspp = self.pa5(atts[4] * img_aspp_feat, (1 - atts[4]) * flow_aspp_feat)
        feaaspp = feaaspp + feaaspp * F.interpolate(nn.Sigmoid()(course_imagemap), feaaspp.size()[2:], mode='bilinear',
                                                    align_corners=True)

        smap4 = self.pred_head5(feaaspp)
        feaaspp = self.blockaspp(feaaspp)

        fea4 = self.pa4(atts[3] * img_layer4_feat, (1 - atts[3]) * flow_layer4_feat, feaaspp)

        smap3 = self.pred_head4(fea4)
        fea4 = self.block4(fea4)

        fea3 = self.pa3(atts[2] * img_layer3_feat, (1 - atts[2]) * flow_layer3_feat, fea4)

        smap2 = self.pred_head3(fea3)
        fea3 = self.block3(fea3)

        fea2 = self.pa2(atts[1] * img_layer2_feat, (1 - atts[1]) * flow_layer2_feat, fea3)

        smap1 = self.pred_head2(fea2)
        fea2 = self.block2(fea2)

        fea1 = self.pa1(atts[0] * img_layer1_feat, (1 - atts[0]) * flow_layer1_feat, fea2)

        fea1 = self.block1(fea1)

        smap = self.pred_head1(fea1)

        shape = image.size()[2:]
        pred1 = F.interpolate(smap, size=shape, mode='bilinear', align_corners=True)

        smap4 = F.interpolate(smap4, image.shape[2:], mode='bilinear', align_corners=True)
        smap3 = F.interpolate(smap3, image.shape[2:], mode='bilinear', align_corners=True)
        smap2 = F.interpolate(smap2, image.shape[2:], mode='bilinear', align_corners=True)
        smap1 = F.interpolate(smap1, image.shape[2:], mode='bilinear', align_corners=True)
        course_imagemap = F.interpolate(course_imagemap, image.shape[2:], mode='bilinear', align_corners=True)
        course_flowmap = F.interpolate(course_flowmap, image.shape[2:], mode='bilinear', align_corners=True)
        return pred1, course_imagemap, course_flowmap, smap4, smap3, smap2, smap1


if __name__ == '__main__':
    net = Model(3, 'test').cuda()
    img = torch.randn(2, 3, 512, 512).cuda()
    flow = torch.randn(2, 3, 512, 512).cuda()
    net.load_state_dict(torch.load("../model_changed_6.pth"))
    pre = net(img, flow)
    print(pre[0].shape)
    print(sum(param.numel() for param in net.parameters()))
