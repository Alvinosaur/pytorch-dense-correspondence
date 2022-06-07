from email.mime import base
from object_pursuit.utils.gen_bases import genBases
from object_pursuit.utils.util import *
from object_pursuit.model.coeffnet.coeffnet_simple import *
from object_pursuit.object_pursuit.pursuit import *
from object_pursuit.loss.memory_loss import MemoryLoss
import itertools
from training import *

from dense_correspondence.network.op_dense_correspondence_network import OPCoeffNet, OPSingleNet, OPMultiDenseCorrespondenceNetwork
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.evaluation.evaluation import OPMultiDenseCorrespondenceEvaluation
from dense_correspondence.dataset.spartan_dataset_masked import OPDataSelector

utils.add_object_pursuit_to_python_path()

seed = 123
np.random.seed(123)
torch.manual_seed(123)


def f(x):
    return x


def set_eval(primary_net, hypernet, backbone=None):
    primary_net.eval()
    hypernet.eval()
    if backbone is not None:
        backbone.eval()


def set_train(primary_net, hypernet, backbone=None):
    primary_net.train()
    hypernet.train()
    if backbone is not None:
        backbone.train()


class OPMultiDenseCorrespondenceTraining(DenseCorrespondenceTraining):
    def __init__(self, *args, **kwargs):
        super(OPMultiDenseCorrespondenceTraining,
              self).__init__(*args, **kwargs)

        # Overwrite evaluator
        self._dce = OPMultiDenseCorrespondenceEvaluation(self._config)

    def run_from_pretrained(self, model_folder, iteration=None, learning_rate=None):
        raise NotImplementedError("OPMulti only meant to train from scratch!")

    def calc_losses(self, dcn, sample):
        # batch size always only 1, metadata entries are lists/tensors with size 1
        (match_type,
         img_a, img_b,
         matches_a, matches_b,
         masked_non_matches_a, masked_non_matches_b,
         background_non_matches_a, background_non_matches_b,
         blind_non_matches_a, blind_non_matches_b,
         metadata) = sample
        batch_size = img_a.shape[0]

        if (match_type == -1).all():
            print ("\n empty data, continuing \n")
            return None

        img_a = Variable(img_a.to(self.device), requires_grad=False)
        img_b = Variable(img_b.to(self.device), requires_grad=False)

        matches_a = Variable(matches_a.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)
        matches_b = Variable(matches_b.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)
        masked_non_matches_a = Variable(masked_non_matches_a.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)
        masked_non_matches_b = Variable(masked_non_matches_b.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)

        background_non_matches_a = Variable(background_non_matches_a.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)
        background_non_matches_b = Variable(background_non_matches_b.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)

        blind_non_matches_a = Variable(blind_non_matches_a.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)
        blind_non_matches_b = Variable(blind_non_matches_b.to(
            self.device).to(torch.long).squeeze(0), requires_grad=False)

        # run both images through the network
        # access specific z basis
        object_idx = metadata["object_id_int"].item()
        image_a_pred = dcn.forward(img_a, object_idx)
        image_a_pred = dcn.process_network_output(image_a_pred, batch_size)

        image_b_pred = dcn.forward(img_b, object_idx)
        image_b_pred = dcn.process_network_output(image_b_pred, batch_size)

        # get loss
        return loss_composer.get_loss(self.loss_fn, match_type,
                                      image_a_pred, image_b_pred, matches_a, matches_b,
                                      masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b)

    def save_network(self, dcn, optimizer, iteration, logging_dict=None):
        """
        Saves network parameters to logging directory
        :return:
        :rtype: None
        """

        network_param_file = os.path.join(
            self._logging_dir, utils.getPaddedString(iteration, width=6) + ".pth")
        optimizer_param_file = network_param_file + ".opt"
        torch.save(dcn.fcn.state_dict(), network_param_file)
        torch.save(optimizer.state_dict(), optimizer_param_file)

        # also save loss history stuff
        if logging_dict is not None:
            log_history_file = os.path.join(self._logging_dir, utils.getPaddedString(
                iteration, width=6) + "_log_history.yaml")
            utils.saveToYaml(logging_dict, log_history_file)

            current_loss_file = os.path.join(self._logging_dir, 'loss.yaml')
            current_loss_data = self._get_current_loss(logging_dict)

            utils.saveToYaml(current_loss_data, current_loss_file)


class OPDenseCorrespondenceTraining(OPMultiDenseCorrespondenceTraining):
    def __init__(self, DatasetClass=OPDataSelector, *args, **kwargs):
        super(OPDenseCorrespondenceTraining, self).__init__(
            DatasetClass=DatasetClass, *args, **kwargs)
        self.z_dim = self._config["dense_correspondence_network"]["OP"]["z_dim"]
        self.normalize = self._config['normalize'] if 'normalize' in self._config else False
        self.net_kwargs = {
            "descriptor_dimension": self._config["dense_correspondence_network"]["descriptor_dimension"],
        }

    def calc_losses(self, dcn, sample):
        # Don't index with any particular object index
        # either SingleNet using one z basis or CoeffNet using all z basis
        return DenseCorrespondenceTraining.calc_losses(self, dcn, sample)

    def setup_logging_dir(self, start_iteration):
        super(OPDenseCorrespondenceTraining,
              self).setup_logging_dir(start_iteration)
        # make other subfolders to store object bases, etc
        self.output_dir = self._logging_dir
        create_dir(self.output_dir)
        self.base_dir = os.path.join(self.output_dir, "Bases")
        create_dir(self.base_dir)
        self.z_dir = os.path.join(self.output_dir, "zs")
        create_dir(self.z_dir)
        create_dir(os.path.join(self.output_dir, "explored_objects"))
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        create_dir(self.checkpoint_dir)
        self.log_file = open(os.path.join(
            self.output_dir, "pursuit_log.txt"), "w")
        self.write_log = lambda s: write_log(self.log_file, s)

        return self._logging_dir

    # def _construct_optimizer(self, parameters):
    #     lr = float(self._config['training']['learning_rate'])
    #     # weight_decay = float(self._config['training']['weight_decay'])
    #     # optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    #     optimizer = optim.RMSprop(filter(
    #         lambda p: p.requires_grad, parameters), lr=lr, weight_decay=1e-7, momentum=0.9)
    #     return optimizer

    def _construct_lr_scheduler(self, optimizer):
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.7)
        return lr_scheduler

    def _batchify_sample(self, sample):
        # Convert to tensors with batch size 1
        (match_type,
         img_a, img_b,
         matches_a, matches_b,
         masked_non_matches_a, masked_non_matches_b,
         background_non_matches_a, background_non_matches_b,
         blind_non_matches_a, blind_non_matches_b,
         metadata) = sample
        return (torch.tensor(match_type), img_a.unsqueeze(0),
                img_b.unsqueeze(0), matches_a.unsqueeze(0),
                matches_b.unsqueeze(0), masked_non_matches_a.unsqueeze(0),
                masked_non_matches_b.unsqueeze(
            0), background_non_matches_a.unsqueeze(0),
            background_non_matches_b.unsqueeze(
            0), blind_non_matches_a.unsqueeze(0),
            blind_non_matches_b.unsqueeze(0), metadata)

    def train_net(self,
                  samples,
                  base_num,
                  net_type,
                  hypernet,
                  backbone,
                  zs=None,
                  save_cp_path=None,
                  max_epochs=80,
                  batch_size=16,
                  lr=0.0004,
                  wait_epochs=3,
                  l1_loss_coeff=0.2,
                  mem_loss_coeff=0.04):
        """
        Changes:
            - removed val_percent: in practice, we will fine-tune on new data
            collected by human, and this "train" distribution is identical to
            our test distrib, so no need to explicitly have a validation set.

        """
        # set logger
        log_file = open(os.path.join(save_cp_path, "log.txt"), "w")

        if net_type == "singlenet":
            primary_net = OPSingleNet(
                z_dim=self.z_dim, hypernet=hypernet, backbone=backbone, device=self.device, **self.net_kwargs)
        elif net_type == "coeffnet":
            assert zs is not None and len(zs) == base_num
            primary_net = OPCoeffNet(
                z_dim=self.z_dim, hypernet=hypernet, backbone=backbone, base_dir=self.base_dir, device=self.device, **self.net_kwargs)
        else:
            raise NotImplementedError

        primary_net.to(self.device)

        # optimize any parameters that have requires_grad=True
        optim_param = filter(lambda p: p.requires_grad, itertools.chain(
            primary_net.parameters(), hypernet.parameters(), backbone.parameters()))

        optimizer = optim.RMSprop(
            optim_param, lr=lr, weight_decay=1e-7, momentum=0.9)

        # Only use singlenet when training hypernetwork since learning new object basis... couldn't represent using existing bases
        if net_type == "singlenet":
            MemLoss = MemoryLoss(Base_dir=self.z_dir, device=self.device)
            mem_coeff = mem_loss_coeff

        global_step = 0
        max_valid_acc = 0
        max_record = None
        stop_counter = 0

        # write info
        z_dir = self.z_dir
        num_trainable_primary_net_params = sum(
            x.numel() for x in primary_net.parameters() if x.requires_grad)
        num_trainable_hypernet_params = sum(
            x.numel() for x in hypernet.parameters() if x.requires_grad)
        # info_text = f('''Starting training:
        #     net type:        {net_type}
        #     Max epochs:      {max_epochs}
        #     Batch size:      {batch_size}
        #     Learning rate:   {lr}
        #     Checkpoints:     {save_cp_path}
        #     z_dir:           {z_dir}
        #     wait epochs:     {wait_epochs}
        #     trainable parameter number of the primarynet: {num_trainable_primary_net_params}
        #     trainable parameter number of the hypernet: {num_trainable_hypernet_params}
        # ''')
        # self.write_log(info_text)

        # training process
        # try:
        for epoch in range(max_epochs):
            set_train(primary_net, hypernet, backbone)
            val_list = []
            with tqdm(total=len(samples)) as pbar:
                # TODO: how much data? do we split into multiple batches?
                loss_sum = 0.0
                for sample in samples:
                    (loss, match_loss, masked_non_match_loss,
                        background_non_match_loss, blind_non_match_loss) = self.calc_losses(primary_net, self._batchify_sample(sample))
                    loss_sum += loss
                avg_loss = loss_sum / len(samples)

                # optimize
                optimizer.zero_grad()
                avg_loss.backward()
                if net_type == "singlenet":
                    MemLoss(hypernet, mem_coeff)
                    # pass

                nn.utils.clip_grad_value_(optim_param, 0.1)
                optimizer.step()

                pbar.update(1)  # TODO: batch size?
                global_step += 1

            if save_cp_path is not None:
                if len(val_list) > 0:
                    avg_valid_acc = sum(val_list) / len(val_list)
                    if avg_valid_acc > max_valid_acc:
                        if net_type == "singlenet":
                            torch.save(primary_net.state_dict(), os.path.join(
                                save_cp_path, "Best_z.pth"))
                        elif net_type == "coeffnet":
                            torch.save(primary_net.state_dict(), os.path.join(
                                save_cp_path, "Best_coeff.pth"))
                        max_valid_acc = avg_valid_acc
                        stop_counter = 0
                        # self.write_log(
                        #     str(f"epoch {epoch} checkpoint saved! best validation acc: {max_valid_acc}"))
                    else:
                        stop_counter += 1

                    # TODO: set a threshold to stop training?
                    if stop_counter >= wait_epochs:
                        # stop procedure
                        # self.write_log(
                        #     str(f"training stopped at epoch {epoch}"))
                        # self.write_log(
                        # str(f"current record value (coeff or z): {max_record}"))
                        log_file.close()
                        return max_valid_acc, primary_net
        # except Exception as e:
        #     self.write_log(str(f"Error catch during training! info: {e}"))
        #     return 0.0, primary_net

        # stop procedure
        self.write_log(str(f"training stopped"))
        self.write_log(
            str(f"current record value (coeff or z): {max_record}"))
        log_file.close()
        return max_valid_acc, primary_net

    def have_seen(self, samples, hypernet, backbone, threshold, start_index=0):
        """
        Checks each existing basis z to see if it represents
        new object well (low segmentation loss)

        samples: List[sample]

        """
        primary_net = OPSingleNet(
            self.z_dim, hypernet=hypernet, backbone=backbone, device=self.device, **self.net_kwargs)
        primary_net.to(self.device)

        all_test_acc = []
        z_files = [os.path.join(self.z_dir, zf) for zf in sorted(
            os.listdir(self.z_dir)) if zf.endswith('.json')]
        min_loss = 1e10
        best_zf = None
        count = 0
        for zf in z_files:
            if count < start_index:
                count += 1
                continue

            primary_net.load_z(zf)

            loss_sum = 0.0
            with torch.no_grad():
                for sample in samples:
                    (loss, match_loss, masked_non_match_loss,
                     background_non_match_loss, blind_non_match_loss) = self.calc_losses(primary_net, self._batchify_sample(sample))
                    loss_sum += loss.item()

            loss_avg = loss_sum / len(samples)
            all_test_acc.append(loss_avg)
            if loss_avg < min_loss:
                min_loss = loss_avg
                best_zf = zf
            count += 1

        z_acc_pairs = [(zf, acc) for zf, acc in zip(z_files, all_test_acc)]
        if min_loss < threshold:
            return True, min_loss, best_zf, z_acc_pairs
        else:
            return False, min_loss, best_zf, z_acc_pairs

    def run(self, pretrained_model_path, loss_current_iteration=0, z_idx=None):
        """
        Runs the training
        :return:
        :rtype:
        """
        start_iteration = copy.copy(loss_current_iteration)

        self.setup(start_iteration)
        self.save_configs(start_iteration)

        z_info = []
        base_info = []

        # Unpack Object Pursuit specific arguments
        express_threshold = self._config["dense_correspondence_network"]["OP"]["express_threshold"]
        save_temp_interval = self._config["dense_correspondence_network"]["OP"]["save_temp_interval"]
        val_percent = 1.0

        # Load pretrained backbone
        self._dcn = self.build_network()
        self._dcn.fcn.load_state_dict(torch.load(pretrained_model_path))

        # make sure network is using cuda and is in train mode
        dcn = self._dcn
        dcn.to(self.device)
        dcn.train()
        try:
            backbone = dcn.fcn.backbone
        except AttributeError:
            backbone = dcn.fcn.resnet34_8s
        hypernet = dcn.fcn.hypernet_block
        num_pretrained_z = dcn.fcn.z.shape[0]

        # Take one arbitrary z to start with
        obj_counter = 0  # save the 0th basis
        if z_idx is None:
            z_idx = np.random.randint(0, num_pretrained_z)
        z0 = dcn.fcn.z[z_idx]
        torch.save({'z': z0, 'weights': hypernet(z0)},
                   os.path.join(self.base_dir, f"z_%04d.json" % obj_counter))
        obj_counter = 1  # now have 1 object so far

        # pursuit info
        pursuit_info = '''Starting pursuing:
            z_dim:                            %d
            output dir:                       %s
            device:                           %s
            pretrained_path:                  %s
            bases dir:                        %s
            express accuracy threshold:       %.2f
            save object interval:             %d (0 means don't save)
        ''' % (self.z_dim, self.output_dir, self.device, pretrained_model_path, self.base_dir, express_threshold, save_temp_interval)
        self.write_log(pursuit_info)
        if backbone is None:
            self.write_log("backbone is None !")
        batch_size = 64

        # logging
        self._logging_dict = dict()
        self._logging_dict['train'] = {"iteration": [], "loss": [], "match_loss": [],
                                       "masked_non_match_loss": [],
                                       "background_non_match_loss": [],
                                       "blind_non_match_loss": [],
                                       "learning_rate": [],
                                       "different_object_non_match_loss": []}

        self._logging_dict['test'] = {"iteration": [], "loss": [], "match_loss": [],
                                      "non_match_loss": []}

        # from training_progress_visualizer import TrainingProgressVisualizer
        # TPV = TrainingProgressVisualizer()

        new_obj_dataset_train, object_id = self._dataset.next(
            num_samples=batch_size)
        # new_obj_dataset_test = self._dataset_test.get_dataset(
        #     object_id=object_id, num_samples=2*batch_size)

        counter = 0
        while new_obj_dataset_train is not None:
            bases = get_z_bases(self.z_dim, self.base_dir, self.device)
            base_num = len(bases)

            # record checkpoints per 8 round
            counter += 1

            if save_temp_interval > 0:
                if counter % save_temp_interval == 0:
                    temp_checkpoint_dir = os.path.join(
                        self.checkpoint_dir, str(f"checkpoint_round_{counter}"))
                    create_dir(temp_checkpoint_dir)
                    torch.save(hypernet.state_dict(), os.path.join(
                        temp_checkpoint_dir, str(f"hypernet.pth")))
                    copy_zs(self.z_dir, os.path.join(
                        temp_checkpoint_dir, "zs"))
                    copy_zs(self.base_dir, os.path.join(
                        temp_checkpoint_dir, "Bases"))
                    self.write_log(
                        str(f"[checkpoint] pursuit round {counter} has been saved to {temp_checkpoint_dir}"))

            # for each new object, create a new dir
            obj_dir = os.path.join(
                self.output_dir, "explored_objects", str(f"obj_{obj_counter}"))
            create_dir(obj_dir)

            new_obj_info = str(f('''Starting new object:
                round:               {counter}
                current base num:    {base_num}
                object data dir:     {object_id}
                object index:        {obj_counter}
                output obj dir:      {obj_dir}
            '''))
            self.write_log(
                "\n=============================start new object==============================")
            self.write_log(new_obj_info)

            # ========================================================================================================
            # check if current object has been seen
            seen, loss, z_file, z_acc_pairs = self.have_seen(
                new_obj_dataset_train, hypernet, backbone, express_threshold, start_index=0)
            if seen:
                self.write_log(
                    str(f"Current object has been seen! corresponding z file: {z_file}, express loss: {loss} vs thresh: {express_threshold}"))
                new_obj_dataset_train, object_id = self._dataset.next(
                    num_samples=batch_size)
                shutil.rmtree(obj_dir)
                self.write_log(
                    "\n===============================end object==================================")
                continue
            else:
                self.write_log(
                    str(f"Current object is novel, min loss: {loss} vs thresh {express_threshold}, most similiar object: {z_file}, start object pursuit"))
            self.write_log(str(f"Z-acc pairs: {z_acc_pairs}"))

            # ========================================================================================================
            # (first check) test if a new object can be expressed by other objects
            if base_num > 0:
                self.write_log("start coefficient pursuit (first check):")
                # freeze the hypernet and backbone
                freeze(hypernet=hypernet, backbone=backbone)
                coeff_pursuit_dir = os.path.join(obj_dir, "coeff_pursuit")
                create_dir(coeff_pursuit_dir)
                self.write_log(
                    str(f"coeff pursuit result dir: {coeff_pursuit_dir}"))
                min_val_loss, coeff_net = self.train_net(base_num=base_num, samples=new_obj_dataset_train,
                                                         zs=bases,
                                                         net_type="coeffnet",  # coeffnet uses linear combo of bases
                                                         hypernet=hypernet,
                                                         backbone=backbone,
                                                         save_cp_path=coeff_pursuit_dir,
                                                         batch_size=batch_size,
                                                         lr=1e-4,
                                                         l1_loss_coeff=0.2)
                self.write_log(
                    str(f"training stop, min_val_loss: {min_val_loss}"))
            # ==========================================================================================================
            # (train as a new base) if not, train this object as a new base
            # the condition to retrain a new base
            if min_val_loss > express_threshold:
                self.write_log(
                    "can't be expressed by bases, start to train as new base:")
                # unfreeze the backbone
                unfreeze(hypernet=hypernet)
                base_update_dir = os.path.join(obj_dir, "base_update")
                create_dir(base_update_dir)
                self.write_log(
                    str(f"base update result dir: {base_update_dir}"))
                min_val_loss, z_net = self.train_net(base_num=base_num, dataset=new_obj_dataset_train,
                                                     net_type="singlenet",
                                                     hypernet=hypernet,
                                                     backbone=backbone,
                                                     save_cp_path=base_update_dir,
                                                     batch_size=batch_size,
                                                     lr=1e-4,
                                                     l1_loss_coeff=0.1,
                                                     mem_loss_coeff=0.04)
                self.write_log(
                    str(f"training stop, min_val_loss: {min_val_loss}"))

                # if the object is invalid
                # NOTE: this seems unreasonable, for what objects does this happen? Why? Would this happen if this object was the first to be trained on?
                if min_val_loss > express_threshold:
                    self.write_log(
                        str(f"[Warning] current object (data path: {object_id}) is unqualified! The validation loss should be less than {express_threshold}, current loss {min_val_loss}; All records will be removed !"))
                    # reset backbone and hypernet
                    if backbone is not None:
                        init_backbone(os.path.join(checkpoint_dir, f(
                            "backbone.pth")), backbone, device, freeze=True)
                    init_hypernet(os.path.join(checkpoint_dir, f(
                        "hypernet.pth")), hypernet, device, freeze=True)
                    new_obj_dataset_train, object_id = self._dataset.next()
                    shutil.rmtree(obj_dir)
                    self.write_log(
                        "\n===============================end object=================================")
                    continue

                # ======================================================================================================
                # (second check) check new z can now be approximated (expressed by coeffs) by current bases
                if base_num > 0:
                    self.write_log(
                        str(f"start to examine whether the object {obj_counter} can be expressed by bases now (second check):"))
                    # freeze the hypernet and backbone
                    freeze(hypernet=hypernet, backbone=backbone)
                    check_express_dir = os.path.join(obj_dir, "check_express")
                    create_dir(check_express_dir)
                    self.write_log(
                        str(f"check express result dir: {check_express_dir}"))
                    min_val_loss, examine_coeff_net = self.train_net(base_num=base_num, dataset=new_obj_dataset_train,
                                                                     zs=bases,
                                                                     net_type="coeffnet",
                                                                     hypernet=hypernet,
                                                                     backbone=backbone,
                                                                     save_cp_path=check_express_dir,
                                                                     batch_size=batch_size,
                                                                     lr=1e-4,
                                                                     acc_threshold=1.0,
                                                                     l1_loss_coeff=0.2)
                else:
                    min_val_loss = 1e10

                if min_val_loss <= express_threshold:
                    self.write_log(
                        str(f"new z can be expressed by current bases, redundant! min_val_loss: {min_val_loss}, don't add it to bases"))
                    # save object's z
                    self.write_log(
                        str(f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.json' to {self.z_dir}"))
                    examine_coeff_net.save_z(os.path.join(self.z_dir, f(
                        "z_{'%04d' % obj_counter}.json")), bases, hypernet)
                else:
                    # save z as a new base
                    # NOTE: Since hypernetwork has been updated, shouldn't z_net also be updated again?
                    self.write_log(
                        str(f"new z can't be expressed by current bases, not redundant! express min_val_loss: {min_val_loss}, add 'base_{'%04d' % base_num}.json' to bases"))
                    z_net.save_z(os.path.join(self.base_dir, f(
                        "base_{'%04d' % base_num}.json")), hypernet)
                    # record base info
                    base_info.append({
                        "index": obj_counter,
                        "data_dir": object_id,
                        "base_file": str(f"base_{'%04d' % base_num}.json"),
                        "z_file": str(f"z_{'%04d' % obj_counter}.json")
                    })
                    # save object's z
                    self.write_log(
                        str(f"object {obj_counter} pursuit complete, save object z 'z_{'%04d' % obj_counter}.json' to {self.z_dir}"))
                    z_net.save_z(os.path.join(self.z_dir, f(
                        "z_{'%04d' % obj_counter}.json")), hypernet)
                # ======================================================================================================
        self.log_file.close()
