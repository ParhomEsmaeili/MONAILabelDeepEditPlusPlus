# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import UNETR, DynUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool

####################################################################### External Validation metric imports
import csv
import shutil
import json

logger = logging.getLogger(__name__)


class DeepEditPlusPlus(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.epistemic_enabled = None
        self.epistemic_samples = None

        
        ###################### Setting the location to extract the label configs from ####################
        dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        label_config_path = os.path.join(dir_name, "monailabel", os.path.splitext(os.path.basename(__file__))[0], 'label_configs.txt')
        
        ################### Importing the label configs dictionary #####################

        with open(label_config_path) as f:
            config_dict = json.load(f)

        self.labels = config_dict["labels"]
        self.original_dataset_labels = config_dict["original_dataset_labels"]
        self.label_mapping = config_dict["label_mapping"]

        # #BRATS CONFIGURATION format #######
        
        # self.labels = {
        #     "tumor": 1,
        #     "background": 0,
        # }
        # self.original_dataset_labels = {
        #     "peritumoral edema": 1,
        #     "non-enhancing tumor": 2,
        #     "enhancing tumor": 3,
        #     "background": 0
        # }

        # self.label_mapping = {
        #     "tumor": ["peritumoral edema", "non-enhancing tumor", "enhancing tumor"],
        #     "background": ["background"]
        # }

        # Number of input channels - 4 for BRATS and 1 for spleen
        # self.number_intensity_ch = 1

        # # Channels being extracted, if using a multi-channel/modality image.
        # self.extract_channels = [3]



        ################################### #Spleen_CT Configuration
        # Binary/Single label

        # self.labels = {     
        #     "spleen": 1,
        #     "background": 0,
        # }

        # self.original_dataset_labels = {
        #     "spleen": 1,
        #     "background": 0
        # }
        # ############ Label Mapping, key = output_class, val = list of input classes. ##############
        # self.label_mapping = {
        #     "spleen": ["spleen"],
        #     "background": ["background"]
        # }

        # Number of input channels - 4 for BRATS and 1 for spleen
        self.number_intensity_ch = 1

        

        network = self.conf.get("network", "dynunet")
        num_epochs = self.conf.get("max_epochs", "50")
        dataset_name = self.conf.get("dataset_name", "Task09_Spleen")
        run_mode = self.conf.get("mode", "train")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{self.name}_{network}_num_epochs_{num_epochs}_dataset_{dataset_name}.pt"),  # published
        ]

        ##################### CHANGE IN PLACE TEMPORARILY FOR THE EXTERNAL VALIDATION METRIC SAVES #####################################################
        output_dir = os.path.join(os.path.expanduser('~'), 'external_validation', dataset_name)
        output_dir_scores = os.path.join(output_dir, 'validation_scores')
        output_dir_images = os.path.join(output_dir, 'validation_images_verif')

        if run_mode == "train":
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            os.makedirs(output_dir_scores)
            os.makedirs(output_dir_images)


            fields = ['deepgrow_dice', 'autoseg_dice', 'deepedit_autoseg_dice']    
            with open(os.path.join(output_dir_scores, 'validation.csv'),'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields) 
        
        ################################################################################################################################################
        #--conf use_pretrained_model false will disable this, or we can change it in the code.
        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_deepedit_{network}_multilabel.pt"
            download_file(url, self.path[0])

        self.target_spacing = (1.0, 1.0, 1.0)  # target space for image
        self.spatial_size = (128, 128, 128)  # train input size

        # Network
        self.network = (
            UNETR(
                spatial_dims=3,
                in_channels= 2 * len(self.labels) + self.number_intensity_ch, 
                out_channels=len(self.labels),
                img_size=self.spatial_size,
                feature_size=64,
                hidden_size=1536,
                mlp_dim=3072,
                num_heads=48,
                pos_embed="conv",
                norm_name="instance",
                res_block=True,
            )
            if network == "unetr"
            else DynUNet(
                spatial_dims=3,
                in_channels= 2 * len(self.labels) + self.number_intensity_ch, #TODO: change back # 2 * len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
            )
        )

        self.network_with_dropout = (
            UNETR(
                spatial_dims=3,
                in_channels= 2 * len(self.labels) + self.number_intensity_ch, #TODO: change back # 2 * len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                img_size=self.spatial_size,
                feature_size=64,
                hidden_size=1536,
                mlp_dim=3072,
                num_heads=48,
                pos_embed="conv",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.2,
            )
            if network == "unetr"
            else DynUNet(
                spatial_dims=3,
                in_channels= 2 * len(self.labels) + self.number_intensity_ch, #TODO: change back #2 * len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
                dropout=0.2,
            )
        )

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        print('checkpoint path is:')
        print(self.path)
        return {
            self.name: lib.infers.DeepEditPlusPlus(
                path=self.path,
                network=self.network,
                #original_dataset_labels=self.original_dataset_labels,
                #label_mapping=self.label_mapping,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                number_intensity_ch=self.number_intensity_ch,
                config={"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
            ),
            f"{self.name}_autoseg": lib.infers.DeepEditPlusPlus(
                path=self.path,
                network=self.network,
                #original_dataset_labels=self.original_dataset_labels,
                #label_mapping=self.label_mapping,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                number_intensity_ch=self.number_intensity_ch,
                type=InferType.SEGMENTATION,
            ),
            f"{self.name}_deepgrow": lib.infers.DeepEditPlusPlus(
                path=self.path,
                network=self.network,
                #original_dataset_labels=self.original_dataset_labels,
                #label_mapping=self.label_mapping,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload","false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                number_intensity_ch=self.number_intensity_ch,
                type=InferType.DEEPGROW
            )
        }

    def trainer(self) -> Optional[TrainTask]:
        #output_dir = os.path.join(self.model_dir, f"{self.name}_" + self.conf.get("network", "dynunet"))
        output_dir = os.path.join(self.model_dir, f"{self.name}_" + self.conf.get("network","dynunet") + "_num_epochs_" + self.conf.get("max_epochs", "50") + "_dataset_" + self.conf.get("dataset_name", "default"))
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.DeepEditPlusPlus(
            model_dir=output_dir,
            network=self.network,
            original_dataset_labels=self.original_dataset_labels,
            label_mapping=self.label_mapping,
            load_path=load_path,
            publish_path=self.path[1],
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            number_intensity_ch=self.number_intensity_ch,
            config={"pretrained": strtobool(self.conf.get("use_pretrained_model", "true"))},
            labels=self.labels,
            debug_mode=False, #True,
            find_unused_parameters=True,
        )
        return task

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=self.network_with_dropout,
                transforms=lib.infers.DeepEditPlusPlus(
                    type=InferType.DEEPEDIT,
                    path=self.path,
                    network=self.network,
                    labels=self.labels,
                    preload=strtobool(self.conf.get("preload", "false")),
                    spatial_size=self.spatial_size,
                ).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        return methods
