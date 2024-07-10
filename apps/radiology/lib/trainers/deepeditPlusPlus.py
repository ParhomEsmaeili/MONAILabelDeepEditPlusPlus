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

import torch
#from lib.transforms.transforms import NormalizeLabelsInDatasetd
from monailabel.deepeditPlusPlus.interaction import Interaction
from monailabel.deepeditPlusPlus.transforms import (
    AddGuidanceSignalDeepEditd,
    AddInitialSeedPointMissingLabelsd,
    AddRandomGuidanceDeepEditd,
    FindAllValidSlicesMissingLabelsd,
    FindDiscrepancyRegionsDeepEditd,
    SplitPredsLabeld,
    NormalizeLabelsInDatasetd,
    AddSegmentationInputChannels,
    #ExtractChannelsd,
    MappingLabelsInDatasetd,
    ExtractMeta,
    IntensityCorrection,
)
from monai.handlers import MeanDice, from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    SelectItemsd,
    ToNumpyd,
    ToTensord,
    ToDeviced,
    DivisiblePadd,
    CenterSpatialCropd
)

from monailabel.deepeditPlusPlus.handlers import TensorBoardImageHandler
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class DeepEditPlusPlus(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        original_dataset_labels,
        label_mapping,
        external_validation_dir,
        n_saved,
        #extract_channels,
        description="Train DeepEdit model for 3D Images",
        spatial_size=(128, 128, 128),
        divisible_padding_factor=[128,128,128],
        target_spacing=(1.0, 1.0, 1.0),
        number_intensity_ch=1,
        deepgrow_probability_train= 0.5,
        deepedit_probability_train= 1/3,
        deepgrow_probability_val=1.0,
        deepedit_probability_val=1.0, #TODO: Why are these probability values for the validation 1.0? This would mean that validation occurs solely on the editing mode.
        debug_mode=False,
        max_iterations=1,
        **kwargs,
    ):
        self._network = network
        self.original_dataset_labels = original_dataset_labels
        self.label_mapping = label_mapping
        self.external_validation_dir = external_validation_dir
        #self.cuda_device = cuda_device
        #self.extract_channels = extract_channels
        self.spatial_size = spatial_size
        self.divisible_padding_factor = divisible_padding_factor
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch
        self.deepgrow_probability_train = deepgrow_probability_train
        self.deepedit_probability_train = deepedit_probability_train
        self.deepgrow_probability_val = deepgrow_probability_val
        self.deepedit_probability_val = deepedit_probability_val 
        self.debug_mode = debug_mode
        self.max_iterations = max_iterations

        super().__init__(model_dir, description, n_saved=n_saved, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=0.0001)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def get_click_transforms(self, context: Context):
        return [
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            #Temporary measure, not the final resolution for the issue with this code deleting the meta dictionary.
            ExtractMeta(keys=("image")),
            ToNumpyd(keys=("image", "label", "pred")),
            # Transforms for click simulation
            FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanceDeepEditd(
                keys="NA",
                guidance="guidance",
                discrepancy="discrepancy",
                probability="probability",
            ),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch),
            #
            AddSegmentationInputChannels(keys=["image"], previous_seg_name = "pred", number_intensity_ch = self.number_intensity_ch, label_names= None, previous_seg_flag= True),
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names", "saved_meta")),
        ]

    def train_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            #ToDeviced(keys=("image", "label"), device="cuda:0"),
            EnsureChannelFirstd(keys=("image", "label")),
            MappingLabelsInDatasetd(keys="label", original_label_names=self.original_dataset_labels, label_names = self._labels, label_mapping=self.label_mapping),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels), 
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # This transform may not work well for MR images
            IntensityCorrection(keys="image", modality = context.imaging_modality),
            #Here we will pad and crop the image to fit the requirements of the backbone architecture
            DivisiblePadd(keys=("image", "label"), k=self.divisible_padding_factor),
            CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),

            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            #Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            # Transforms for click simulation (depracated)
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch),
            AddSegmentationInputChannels(keys="image", previous_seg_name= None, number_intensity_ch = self.number_intensity_ch, label_names=None, previous_seg_flag= False),

            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")), #"guidance", "label_names")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=len(self._labels),
            ),
            SplitPredsLabeld(keys="pred"),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=("image", "label")),
            MappingLabelsInDatasetd(keys="label", original_label_names=self.original_dataset_labels, label_names = self._labels, label_mapping=self.label_mapping),
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # This transform may not work well for MR images
            IntensityCorrection(keys="image", modality = context.imaging_modality),
            DivisiblePadd(keys=("image", "label"), k=self.divisible_padding_factor),
            CenterSpatialCropd(keys=("image", "label"), roi_size=self.spatial_size),
            #ScaleIntensityRanged(keys=("image"), a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            #Resized(keys=("image", "label"), spatial_size=self.spatial_size, mode=("area", "nearest")),
            # Transforms for click simulation 
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
            AddGuidanceSignalDeepEditd(keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch),
            AddSegmentationInputChannels(keys="image", previous_seg_name=None, number_intensity_ch = self.number_intensity_ch, label_names=None, previous_seg_flag= False),
            
            ToTensord(keys=("image", "label")),
            SelectItemsd(keys=("image", "label", "label_names")), #"guidance", "label_names")),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=self.deepgrow_probability_train,
            deepedit_probability=self.deepedit_probability_train,
            num_intensity_channel=self.number_intensity_ch,  
            transforms=self.get_click_transforms(context),
            click_probability_key="probability",
            train=True,
            external_validation_output_dir=self.external_validation_dir,
            max_iterations=self.max_iterations
            #label_names=self._labels,
        )

    def val_iteration_update(self, context: Context):
        return Interaction(
            deepgrow_probability=self.deepgrow_probability_val,
            deepedit_probability= self.deepedit_probability_val,
            num_intensity_channel=self.number_intensity_ch, 
            transforms=self.get_click_transforms(context),
            click_probability_key="probability",
            train=False,
            external_validation_output_dir=self.external_validation_dir,
            max_iterations=self.max_iterations 
            #label_names=self._labels,
        )

    def train_key_metric(self, context: Context):
        all_metrics = dict()
        all_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]), include_background=False)
        for key_label in self._labels:
            if key_label != "background":
                all_metrics[key_label + "_dice"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
                )
        return all_metrics

    def val_key_metric(self, context: Context):
        all_metrics = dict()
        all_metrics["val_mean_dice"] = MeanDice(
            output_transform=from_engine(["pred", "label"]), include_background=False
        )
        for key_label in self._labels:
            if key_label != "background":
                all_metrics["val_" + key_label + "_dice"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
                )
        return all_metrics

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if self.debug_mode and context.local_rank == 0:
            handlers.append(TensorBoardImageHandler(log_dir=context.events_dir))
        return handlers
