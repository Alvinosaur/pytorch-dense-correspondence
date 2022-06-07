#!/usr/bin/python
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()

from dense_correspondence.network.resnet_dilated_OP import Resnet34_8s_OP
from object_pursuit.pretrain._model import get_multinet
from object_pursuit.model.deeplabv3.deeplab import *
import object_pursuit.model.coeffnet.coeffnet as coeffnet
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os
import numpy as np
import warnings
import logging


class OPMultiDenseCorrespondenceNetwork(DenseCorrespondenceNetwork):
    def __init__(self, *args, **kwargs):
        super(OPMultiDenseCorrespondenceNetwork,
              self).__init__(*args, **kwargs)

    @staticmethod
    def get_fcn(config):
        fcn = get_multinet(model_type=config["OP"]["model_type"],
                           class_num=config["OP"]["class_num"],
                           z_dim=config["OP"]["z_dim"],
                           device=config["device"],
                           use_backbone=config["OP"]["use_backbone"],
                           freeze_backbone=config["OP"]["freeze_backbone"],
                           use_batchnorm=False,  # only using batchsize=1
                           out_dim=config["descriptor_dimension"])

        fcn.to(config["device"])
        return fcn

    def forward(self, input, obj_index, ret_z=False):
        res, z = self.fcn(input, obj_index)
        if self._normalize:
            #print("normalizing descriptor norm")
            norm = torch.norm(res, 2, 1)  # [N,1,H,W]
            res = res / norm

        if ret_z:
            return res, z
        else:
            return res

    def forward_single_image_tensor(self, img_tensor, obj_index, ret_z=False):
        assert len(img_tensor.shape) == 3

        # transform to shape [1,3,H,W]
        img_tensor = img_tensor.unsqueeze(0)

        # make sure it's on the GPU
        img_tensor = torch.tensor(img_tensor, device=self.device)

        res = self.forward(img_tensor, obj_index=obj_index,
                           ret_z=ret_z)  # shape [1,D,H,W]
        # print("res.shape 1", res.shape)

        res = res.squeeze(0)  # shape [D,H,W]
        # print("res.shape 2", res.shape)

        res = res.permute(1, 2, 0)  # shape [H,W,D]
        # print("res.shape 3", res.shape)

        return res


class OPMultiDenseCorrespondenceNetworkV2(OPMultiDenseCorrespondenceNetwork):
    @staticmethod
    def get_fcn(config):
        fcn = Resnet34_8s_OP(descriptor_dim=config['descriptor_dimension'],
                             z_dim=config["OP"]["z_dim"], num_objects=config["OP"]["class_num"], freeze_backbone=config["OP"]["freeze_backbone"])

        fcn.to(config["device"])
        return fcn


class OPSingleNet(nn.Module):
    def __init__(self, z_dim, hypernet, backbone, device, descriptor_dimension):
        super(OPSingleNet, self).__init__()
        self.z_dim = z_dim
        self.z = nn.Parameter(torch.randn(z_dim))
        self.hypernet = hypernet
        self.backbone = backbone
        self.device = device
        self.descriptor_dimension = descriptor_dimension

    def process_network_output(self, image_pred, N):
        image_pred = image_pred.view(N, self.descriptor_dimension, -1)
        image_pred = image_pred.permute(0, 2, 1)
        return image_pred

    def save_z(self, file_path):
        with torch.no_grad():
            z = self.z.clone().detach()
            weights = self.hypernet(z)
            torch.save({'z': z, 'weights': weights}, file_path)

    def load_z(self, file_path):
        with torch.no_grad():
            self.z.data = torch.load(file_path, map_location=self.device)['z']

    def forward_helper(self, x, z):
        input_spatial_dim = x.size()[2:]
        x = self.backbone(x)
        final_conv_weight = self.hypernet(z)
        x = F.conv2d(x, final_conv_weight, bias=None,
                     stride=1, padding=0, dilation=1)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        return x

    def forward(self, x):
        z = self.z
        return self.forward_helper(x, z)


class OPCoeffNet(OPSingleNet):
    def __init__(self, base_dir, nn_init=True, index=None, *args, **kwargs):
        super(OPCoeffNet, self).__init__(*args, **kwargs)
        # base & coeffs
        del self.z
        self.zs, self.base_num = self._get_z_bases(base_dir, self.device)

        self.coeffs = nn.Parameter(torch.randn(self.base_num))
        # init coeffs
        if nn_init:
            if index is not None:
                self.init_value = 0.0  # 1.0/math.sqrt(self.base_num)
                torch.nn.init.constant_(self.coeffs, self.init_value)
                self.coeffs.data[index] = 1.0
            else:
                self.init_value = 1.0 / np.sqrt(self.base_num)
                torch.nn.init.constant_(self.coeffs, self.init_value)

        self.combine_func = self._linear

    def _get_z_bases(self, base_dir, device):
        if os.path.isdir(base_dir):
            base_files = [os.path.join(base_dir, file) for file in sorted(
                os.listdir(base_dir)) if file.endswith(".json")]
            num_base_files = len(base_files)
            print(str(f"Coeffnet: found {num_base_files} Base files"))
            zs = []
            for file in base_files:
                z = torch.load(file, map_location=device)['z']
                assert(z.size()[0] == self.z_dim)
                zs.append(z)
            base_num = len(zs)
            return zs, base_num
        elif os.path.isfile(base_dir):
            assert(os.path.isfile(base_dir) and base_dir.endswith(".pth"))
            zs = torch.load(base_dir, map_location=device)['z']
            print("found base num: ", len(zs))
            return zs, len(zs)

    def _linear(self, zs, coeffs):
        assert(len(zs) > 0 and len(zs) == coeffs.size()[0])
        z = zs[0] * coeffs[0]
        for i in range(1, len(zs)):
            z += zs[i] * coeffs[i]
        return z

    def forward(self, x):
        z = self.combine_func(self.zs, self.coeffs)
        return self.forward_helper(x, z)
