from pytorch_segmentation_detection.models.resnet_dilated import adjust_input_image_size_for_proper_feature_alignment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from object_pursuit.model.coeffnet.hypernet_block import HypernetConvBlock, FCBlock


class Resnet34_8s_OP(nn.Module):

    def __init__(self, descriptor_dim, z_dim, num_objects, freeze_backbone=False):

        super(Resnet34_8s_OP, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = models.resnet34(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        resnet34_8s.fc = nn.Identity()
        print("ResNeet34 OP freeze backbone: {}".format(bool(freeze_backbone)))
        if freeze_backbone:
            for param in resnet34_8s.parameters():
                param.requires_grad = False

        self.z = nn.Parameter(torch.randn(
            (num_objects, z_dim)), requires_grad=True)

        self.hypernet_weight_block = HypernetConvBlock(
            z_dim, kernel_size=1, in_size=resnet34_8s.inplanes, out_size=descriptor_dim, use_bn=False)

        self.hypernet_bias_block = FCBlock(hidden_ch=int(z_dim/2), num_hidden_layers=1,
                                           in_features=z_dim, out_features=descriptor_dim, outermost_linear=True)

        self.resnet34_8s = resnet34_8s

    def forward(self, x, index, feature_alignment=False):
        input_spatial_dim = x.size()[2:]
        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(
                x, output_stride=8)

        x = self.resnet34_8s(x)

        # final conv layer specified by object index
        z = self.z[index]

        # don't use batchnorm weights/biases
        final_conv_weight = self.hypernet_weight_block(z)
        final_conv_bias = self.hypernet_bias_block(z)
        x = F.conv2d(x, final_conv_weight, bias=final_conv_bias,
                     stride=1, padding=0, dilation=1)

        save_dict = dict(kernel=final_conv_weight, bias=final_conv_bias)
        torch.save(save_dict, f"temp_hypernet_output_z_{index}.pt")

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x, z
