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

####################################################################### External imports
import csv
import shutil
import json
import datetime 
from os.path import dirname as up 
import copy 

logger = logging.getLogger(__name__)


class DeepEdit(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.epistemic_enabled = None
        self.epistemic_samples = None

        # # Multilabel
        # self.labels = {
        #     "spleen": 1,
        #     "right kidney": 2,
        #     "left kidney": 3,
        #     "liver": 6,
        #     "stomach": 7,
        #     "aorta": 8,
        #     "inferior vena cava": 9,
        #     "background": 0,
        # }

        # # Single label
        # # self.labels = {
        # #     "spleen": 1,
        # #     "background": 0,
        # # }


        ###################### Setting the location to extract the label configs from ####################
        dir_name = up(up(up(up(up(__file__)))))
        
        label_config_path = os.path.join(dir_name, "monailabel", os.path.splitext(os.path.basename(__file__))[0], self.conf.get('dataset_name') + '_label_configs.txt')
        
        ################### Importing the label configs dictionary #####################

        with open(label_config_path) as f:
            config_dict = json.load(f)

        self.labels = config_dict["labels"]
        self.original_dataset_labels = config_dict["original_dataset_labels"]
        self.label_mapping = config_dict["label_mapping"]



        # Number of input channels for image 
        self.number_intensity_ch = 1
        
        ########################### Extracting the parameters which are used for naming model files/checkpoints/finding checkpoint paths etc.
        network = self.conf.get("network", "dynunet")
        model_name = self.conf.pop("checkpoint", None) #If a specific checkpoint/save is being used, else use the default. 


        #Adding the datetime for saving the model weights for train, or extracting the datetime set in the inference script:
        if self.conf.get("mode") == "train":
            self.datetime_now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        else:
            self.datetime_now = self.conf.pop("datetime")

        ################ Extracting parameters which dicate how many epochs/interval size for saving checkpoints ####################
        
        self.max_epochs = self.conf.get("max_epochs", 50)
        self.save_interval = self.conf.get("save_interval", 10)

        #network = self.conf.get("network", "dynunet")

        # Model Files

        #Setting the filename automatically based off the settings provided. If not using checkpoint: DEPRECATED_WE SAVE THIS INSTEAD TO A TXT FILE. The models 
        #are kept separate through the use of the datetime!

        
        self.filename = 'models'
        # len_keys = len(list(self.conf.keys()))
        # for i, key in enumerate(self.conf.keys()):
        #     if key == "mode":
        #         continue
        #     else:
        #         self.filename += key
        #         self.filename += f'_{self.conf[key]}'
        #         if i != len_keys - 1:
        #             self.filename += '_'
        
        ###### Name of the text file which we are going to save our non-default hyperparam (non-default params) info to ##################

        hyperparams_filename = 'optional_hyperparam_settings.txt'

        if self.conf.get("mode") == "train":
            #create a duplicate dict without the mode name:
            duplicate_conf = copy.deepcopy(self.conf)
            del duplicate_conf["mode"]
            
            os.makedirs(os.path.join(self.model_dir, self.datetime_now))
            with open(os.path.join(self.model_dir, self.datetime_now, hyperparams_filename), 'w') as file:
                file.write(json.dumps(duplicate_conf))
                

        if model_name == None:
            model_weights_path = os.path.join(self.model_dir, self.datetime_now, self.filename + '.pt')
            #In this case we use the default saved weights:
        else:
            model_weights_path = os.path.join(self.model_dir, self.datetime_now, self.filename, 'train_01', model_name + '.pt')
            
        
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),  # pretrained
            model_weights_path   #f"{self.name}_{network}_num_epochs_{num_epochs}_dataset_{dataset_name}.pt"),  # published
        ]

        # # Model Files
        # self.path = [
        #     os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),  # pretrained
        #     os.path.join(self.model_dir, f"{self.name}_{network}.pt"),  # published
        # ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_deepedit_{network}_multilabel.pt"
            download_file(url, self.path[0])

        # self.target_spacing = (1.0, 1.0, 1.0)  # target space for image
        # self.spatial_size = (128, 128, 128)  # train input size

        self.target_spacing = json.loads(self.conf.get('target_spacing', '[1.0, 1.0, 1.0]'))  # target space for image
        self.spatial_size = json.loads(self.conf.get('spatial_size', '[128, 128, 128]'))  # train input size
        self.divisible_padding_factor = json.loads(self.conf.get('divisible_padding_factor', '[64, 64, 32]'))
        self.max_interactions = int(self.conf.get("max_interactions", 1))
        # Network
        self.network = (
            UNETR(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
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
                in_channels=len(self.labels) + self.number_intensity_ch,
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
                in_channels=len(self.labels) + self.number_intensity_ch,
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
                in_channels=len(self.labels) + self.number_intensity_ch,
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
            self.name: lib.infers.DeepEdit(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                divisible_padding_factor=self.divisible_padding_factor,
                number_intensity_ch=self.number_intensity_ch,
                config={"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
            ),
            f"{self.name}_autoseg": lib.infers.DeepEdit(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                divisible_padding_factor=self.divisible_padding_factor,
                number_intensity_ch=self.number_intensity_ch,
                type=InferType.SEGMENTATION,
            ),
        }

    def trainer(self) -> Optional[TrainTask]:
        # output_dir = os.path.join(self.model_dir, f"{self.name}_" + self.conf.get("network", "dynunet"))
        output_dir = os.path.join(self.model_dir, self.datetime_now, self.filename) 
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.DeepEdit(
            model_dir=output_dir,
            network=self.network,
            original_dataset_labels=self.original_dataset_labels,
            label_mapping=self.label_mapping,
            n_saved=int(self.max_epochs)//self.save_interval,
            load_path=load_path,
            publish_path=self.path[1],
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            divisible_padding_factor=self.divisible_padding_factor,
            number_intensity_ch=self.number_intensity_ch,
            config={"pretrained": strtobool(self.conf.get("use_pretrained_model", "true"))},
            labels=self.labels,
            debug_mode=False,
            max_interactions=self.max_interactions,
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
                transforms=lib.infers.DeepEdit(
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
