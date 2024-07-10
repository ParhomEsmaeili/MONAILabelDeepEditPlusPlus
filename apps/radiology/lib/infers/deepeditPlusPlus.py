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
from typing import Callable, Sequence, Union

from lib.transforms.transforms import GetCentroidsd
from monailabel.deepeditPlusPlus.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelDeepEditd,
    AddSegmentationInputChannels,
    IntensityCorrection,
    MappingLabelsInDatasetd, 
    MappingGuidancePointsd
)
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
    DivisiblePadd,
    CenterSpatialCropd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored
##################################################
# from monai.transforms import LoadImage
# import torch
# import numpy as np
# import nibabel as nib 
# import os
logger = logging.getLogger(__name__)


class DeepEditPlusPlus(BasicInferTask):
    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path,
        #original_dataset_labels,
        #label_mapping,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        dimension=3,
        spatial_size=(128, 128, 128),
        target_spacing=(1.0, 1.0, 1.0),
        divisible_padding_factor=[128,128,128],
        number_intensity_ch=1,
        description="A DeepEdit model for volumetric (3D) segmentation over 3D Images",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            load_strict=False,
            **kwargs,
        )

        #self.original_dataset_labels = original_dataset_labels
        #self.label_mapping = label_mapping
        
        #######
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.divisible_padding_factor = divisible_padding_factor
        self.number_intensity_ch = number_intensity_ch
        self.load_strict = False

    def pre_transforms(self, data=None):
        # #Hacky check in the UI
        # if "imaging_modality" not in data:
        #     data["imaging_modality"] = "MRI"
        # if "previous_seg" not in data:
        #     data["previous_seg"] = "/home/parhomesmaeili/Desktop/BraTS2021_01620.nii.gz"    
        if self.type == InferType.DEEPEDIT:
            t = [
                LoadImaged(keys=["image", "previous_seg"], reader="ITKReader", image_only=False), 
                #TODO: Previous_seg should ideally be the path to a temporary file that gets generated when the update button is hit?
                EnsureChannelFirstd(keys=["image", "previous_seg"]),
                Orientationd(keys=["image", "previous_seg"], axcodes="RAS"),
                #ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                IntensityCorrection(keys="image", modality=data["imaging_modality"]),
                DivisiblePadd(keys=("image", "previous_seg"), k=self.divisible_padding_factor),
                MappingGuidancePointsd(keys=("image"), original_spatial_size_key="preprocessing_original_size", label_names=self.labels), 
                #CenterSpatialCropd(keys=("image"),roi_size=self.spatial_size), #Potential issue with inverse if we crop out regions which are non-zero in the image.
                #MappingLabelsInDatasetd(keys="previous_seg",original_label_names=self.original_dataset_labels, label_names = self.labels, label_mapping=self.label_mapping),
                
                #Addguidance has been deprecated since we will no longer work with resampling images in the loop:
                
                #AddGuidanceFromPointsDeepEditd(ref_image="image", guidance="guidance", label_names=self.labels),
                #Resized(keys=["image", "previous_seg"], spatial_size=self.spatial_size, mode=["area", "nearest"]),
                #ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),
                AddGuidanceSignalDeepEditd(
                    keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch, label_names = self.labels
                ),
                AddSegmentationInputChannels(keys=["image"], previous_seg_name="previous_seg", number_intensity_ch = self.number_intensity_ch, label_names=self.labels, previous_seg_flag= True)
            ]
            
        else:
            t = [
            LoadImaged(keys="image", reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            #ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            IntensityCorrection(keys="image", modality=data["imaging_modality"]),
            DivisiblePadd(keys=("image"), k=self.divisible_padding_factor),
            ]
            if self.type == InferType.DEEPGROW:
            
                t.extend(
                    [   
                        MappingGuidancePointsd(keys="image", original_spatial_size_key="preprocessing_original_size", label_names=self.labels),
                        #Addguidancefrompoints has been deprecated since we will no longer work with resampling images in the loop:

                        #AddGuidanceFromPointsDeepEditd(ref_image="image", guidance="guidance", label_names=self.labels),
                        #Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                        #ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),

                        AddGuidanceSignalDeepEditd(
                            keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch, label_names = self.labels
                        ),
                        AddSegmentationInputChannels(keys="image", number_intensity_ch = self.number_intensity_ch, label_names=self.labels, previous_seg_flag= False),
                    ]
                )
            
            elif self.type == InferType.SEGMENTATION:
                t.extend(
                    [
                        #Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                        DiscardAddGuidanced(
                            keys="image", label_names=self.labels, number_intensity_ch=self.number_intensity_ch
                        ),
                        AddSegmentationInputChannels(keys="image", number_intensity_ch = self.number_intensity_ch, label_names=self.labels, previous_seg_flag= False),
                    ]
                )
        # t.append(DivisiblePadd(keys=("image"), k=self.divisible_padding_factor))
        # t.append(CenterSpatialCropd(keys=("image"),roi_size=self.spatial_size)) #Potential issue with inverse if we crop out regions which are non-zero in the image.
        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True), #just after this, all the dimensions get squeezed into one channel.
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"), 
            Restored(
                keys="pred",
                ref_image="image",
                config_labels=self.labels if data.get("restore_label_idx", False) else None, 
                #TODO: When running the code in our experiments we need to make sure the request has this variable to be TRUE. Need to also change the UI/front end so it works with integer codes for representing labels, saving labels, instead of the indexing methods.
            ),
            GetCentroidsd(keys="pred", centroids_key="centroids"),
        ]
