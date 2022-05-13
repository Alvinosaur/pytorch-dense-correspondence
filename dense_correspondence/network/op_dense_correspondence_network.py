#!/usr/bin/python

from object_pursuit.pretrain._model import get_multinet
from object_pursuit.model.deeplabv3.deeplab import *
from object_pursuit.model.coeffnet.coeffnet import *
from dense_correspondence_network import DenseCorrespondenceNetwork
import torch.nn as nn
import torch
import sys
import os
import numpy as np
import warnings
import logging
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()


class OPMultiDenseCorrespondenceNetwork(DenseCorrespondenceNetwork):
    def __init__(self, *args, **kwargs):
        super(OPMultiDenseCorrespondenceNetwork,
              self).__init__(*args, **kwargs)

    @staticmethod
    def from_config(config, load_stored_params=True, model_param_file=None):
        """
        Same as original, but replaces typical fcn with a MultiNet() from OP
        """

        if "backbone" not in config:
            # default to CoRL 2018 backbone!
            config["backbone"] = dict()
            config["backbone"]["model_class"] = "Resnet"
            config["backbone"]["resnet_name"] = "Resnet34_8s"

        print(config.keys())

        # multinet = get_multinet(**config["OP"])
        multinet = get_multinet(model_type=config["OP"]["model_type"],
                                class_num=config["OP"]["class_num"],
                                z_dim=config["OP"]["z_dim"],
                                device=config["device"],
                                use_backbone=config["OP"]["use_backbone"],
                                freeze_backbone=config["OP"]["freeze_backbone"],
                                use_batchnorm=False,  # only using batchsize=1
                                out_dim=config["descriptor_dimension"]
                                )

        if 'normalize' in config:
            normalize = config['normalize']
        else:
            normalize = False

        dcn = OPMultiDenseCorrespondenceNetwork(multinet, config['descriptor_dimension'],
                                                image_width=config['image_width'],
                                                image_height=config['image_height'],
                                                normalize=normalize,
                                                device=config['device'])

        if load_stored_params:
            assert model_param_file is not None
            # should be an absolute path
            config['model_param_file'] = model_param_file
            try:
                dcn.load_state_dict(torch.load(model_param_file))
            except:
                logging.info(
                    "loading params with the new style failed, falling back to dcn.fcn.load_state_dict")
                dcn.fcn.load_state_dict(torch.load(model_param_file))

        dcn.train()
        dcn.config = config
        return dcn

    @staticmethod
    def from_model_folder(model_folder, load_stored_params=True, model_param_file=None,
                          iteration=None):
        """
        Loads a OPMultiDenseCorrespondenceNetwork from a model folder
        :param model_folder: the path to the folder where the model is stored. This direction contains
        files like

            - 003500.pth
            - training.yaml

        :type model_folder:
        :return: a DenseCorrespondenceNetwork objecc t
        :rtype:
        """

        from_model_folder = False
        model_folder = utils.convert_to_absolute_path(model_folder)

        if model_param_file is None:
            model_param_file, _, _ = utils.get_model_param_file_from_directory(
                model_folder, iteration=iteration)
            from_model_folder = True

        model_param_file = utils.convert_to_absolute_path(model_param_file)

        training_config_filename = os.path.join(model_folder, "training.yaml")
        training_config = utils.getDictFromYamlFilename(
            training_config_filename)
        config = training_config["dense_correspondence_network"]
        config["path_to_network_params_folder"] = model_folder
        config["model_param_filename_tail"] = os.path.split(model_param_file)[
            1]

        dcn = OPMultiDenseCorrespondenceNetwork.from_config(config,
                                                            load_stored_params=load_stored_params,
                                                            model_param_file=model_param_file)

        # whether or not network was constructed from model folder
        dcn.constructed_from_model_folder = from_model_folder
        dcn.model_folder = model_folder
        return dcn

    def forward(self, input, obj_index, ret_z=False):
        """
        Simple forward pass on the network. obj_index indexes
        into the specific object basis z feature to use.

        D = descriptor dimension
        N = batch size

        :param img_tensor: input tensor img.shape = [N, D, H , W] where
                    N is the batch size
        :type img_tensor: torch.Variable or torch.Tensor
        :return: torch.Variable with shape [N, D, H, W],
        :rtype:
        """
        res, z = self.fcn(input, obj_index)
        if self._normalize:
            #print "normalizing descriptor norm"
            norm = torch.norm(res, 2, 1)  # [N,1,H,W]
            res = res/norm

        if ret_z:
            return res, z
        else:
            return res

    def forward_single_image_tensor(self, img_tensor, obj_index, ret_z=False):
        """
        Simple forward pass on the network.

        Assumes the image has already been normalized (i.e. subtract mean, divide by std dev)

        Color channel should be RGB

        :param img_tensor: torch.FloatTensor with shape [3,H,W]
        :type img_tensor:
        :return: torch.FloatTensor with shape  [H, W, D]
        :rtype:
        """

        assert len(img_tensor.shape) == 3

        # transform to shape [1,3,H,W]
        img_tensor = img_tensor.unsqueeze(0)

        # make sure it's on the GPU
        img_tensor = torch.tensor(img_tensor, device=self.device)

        res = self.forward(img_tensor, obj_index=obj_index,
                           ret_z=ret_z)  # shape [1,D,H,W]
        # print "res.shape 1", res.shape

        res = res.squeeze(0)  # shape [D,H,W]
        # print "res.shape 2", res.shape

        res = res.permute(1, 2, 0)  # shape [H,W,D]
        # print "res.shape 3", res.shape

        return res


class OPDenseCorrespondenceNetwork(DenseCorrespondenceNetwork):
    def __init__(self, hypernet=None, backbone=None, *args, **kwargs):
        super(DenseCorrespondenceNetwork, self).__init__(*args, **kwargs)
        self.hypernet = hypernet
        self.backbone = backbone

    @staticmethod
    def from_config(config, load_stored_params=True, model_param_file=None):
        """
        Same as original, but replaces typical fcn with a MultiNet() from OP
        """

        if "backbone" not in config:
            # default to CoRL 2018 backbone!
            config["backbone"] = dict()
            config["backbone"]["model_class"] = "Resnet"
            config["backbone"]["resnet_name"] = "Resnet34_8s"

        coeff_net = Coeffnet(**config["OP"])

        if 'normalize' in config:
            normalize = config['normalize']
        else:
            normalize = False

        dcn = OPMultiDenseCorrespondenceNetwork(coeff_net, config['descriptor_dimension'],
                                                image_width=config['image_width'],
                                                image_height=config['image_height'],
                                                normalize=normalize)

        if load_stored_params:
            assert model_param_file is not None
            # should be an absolute path
            config['model_param_file'] = model_param_file
            try:
                dcn.load_state_dict(torch.load(model_param_file))
            except:
                logging.info(
                    "loading params with the new style failed, falling back to dcn.fcn.load_state_dict")
                dcn.fcn.load_state_dict(torch.load(model_param_file))

        dcn.cuda()
        dcn.train()
        dcn.config = config
        return dcn

    def forward(self, input, ident):
        """
        Simple forward pass on the network.

        Does NOT normalize the image

        D = descriptor dimension
        N = batch size

        :param img_tensor: input tensor img.shape = [N, D, H , W] where
                    N is the batch size
        :type img_tensor: torch.Variable or torch.Tensor
        :return: torch.Variable with shape [N, D, H, W],
        :rtype:
        """

        res = self.fcn(input, ident, hypernet=self.hypernet,
                       backbone=self.backbone)
        if self._normalize:
            #print "normalizing descriptor norm"
            norm = torch.norm(res, 2, 1)  # [N,1,H,W]
            res = res/norm

        return res
