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

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import logging 
from monai.data import decollate_batch, list_data_collate
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.transforms import Compose
from monai.utils.enums import CommonKeys

##################################################
import nibabel as nib
import os 

logger = logging.getLogger(__name__)

class Interaction:
    """
    Ignite process_function used to introduce interactions (simulation of clicks) for DeepEdit Training/Evaluation.

    More details about this can be found at:

        Diaz-Pinto et al., MONAI Label: A framework for AI-assisted Interactive
        Labeling of 3D Medical Images. (2022) https://arxiv.org/abs/2203.12362

    Args:
        deepgrow_probability: probability of simulating clicks in an iteration
        transforms: execute additional transformation during every iteration (before train).
            Typically, several Tensor based transforms composed by `Compose`.
        train: True for training mode or False for evaluation mode
        click_probability_key: key to click/interaction probability
        label_names: Dict of label names
        max_interactions: maximum number of interactions per iteration
    """

    def __init__(
        self,
        deepgrow_probability: float,
        deepedit_probability: float,
        num_intensity_channel: int,
        transforms: Sequence[Callable] | Callable,
        train: bool,
        #label_names: None | dict[str, int] = None,
        click_probability_key: str = "probability",
        max_interactions: int = 1,
    ) -> None:
        self.deepgrow_probability = deepgrow_probability
        self.deepedit_probability = deepedit_probability
        self.num_intensity_channel = num_intensity_channel
        self.transforms = Compose(transforms) if not isinstance(transforms, Compose) else transforms
        self.train = train
        #self.label_names = label_names
        self.click_probability_key = click_probability_key
        self.max_interactions = max_interactions

    def __call__(self, engine: SupervisedTrainer | SupervisedEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        label_names = batchdata["label_names"]

        if np.random.choice([True, False], p=[self.deepgrow_probability, 1 - self.deepgrow_probability]): #TODO: should this be done as a subclass of the randomizable class, so that we can use fixed seeds for this also?
            #Here we run the loop for Deepgrow
            logger.info("Grow from prompts Inner Subloop")
            
        
        else:
            #Here we run the loop for generating autoseg prompt channels
            # zero out input guidance channels
            batchdata_list = decollate_batch(batchdata, detach=True)
            for i in range(self.num_intensity_channel, self.num_intensity_channel + len(label_names)):
                batchdata_list[0][CommonKeys.IMAGE][i] *= 0
            batchdata = list_data_collate(batchdata_list)
            logger.info("AutoSegmentation Inner Subloop")
            

        #Here we use the initial segmentations to generate a prediction (our new previous seg) and generate a new set of inputs with this updated previous seg.
        
        if np.random.choice([True, False], p  = [self.deepedit_probability, 1 - self.deepedit_probability]):
        
            logger.info("Editing mode Inner Subloop")
            for j in range(self.max_interactions):
                inputs, _ = engine.prepare_batch(batchdata)
                inputs = inputs.to(engine.state.device)
                
                batchdata_list = decollate_batch(batchdata, detach=True)
                for i in range(batchdata_list[0][CommonKeys.IMAGE].size(dim=0)):
                    placeholder_tensor = batchdata_list[0][CommonKeys.IMAGE]
                    placeholder = np.array(placeholder_tensor[i])
                    #print(placeholder)
                    nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/TrainingInnerLoop', f'inputs_prior_prediction_iteration_{j}_channel_' + str(i)+'.nii.gz'))
                batchdata = list_data_collate(batchdata_list)

                engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
                engine.network.eval()

                with torch.no_grad():
                    if engine.amp:
                        with torch.cuda.amp.autocast():
                            predictions = engine.inferer(inputs, engine.network)
                    else:
                        predictions = engine.inferer(inputs, engine.network)
                batchdata.update({CommonKeys.PRED: predictions})
                
                #verification check of the prediction generated by the forward pass using Autoseg/Deepgrow inputs. TODO: Delete this.

                for i in range(batchdata[CommonKeys.PRED].size(dim=1)):
                    placeholder_tensor = batchdata[CommonKeys.PRED][0].cpu()
                    placeholder = np.array(placeholder_tensor[i], dtype=np.float32)
                    #print(placeholder)
                    nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/TrainingInnerLoopPrediction', f'predictions_iteration_{j}_channel_' + str(i)+'.nii.gz'))

                # decollate/collate batchdata to execute click transforms
                batchdata_list = decollate_batch(batchdata, detach=True)
                for i in range(len(batchdata_list)):
                    batchdata_list[i][self.click_probability_key] = (
                        (1.0 - ((1.0 / self.max_interactions) * j)) if self.train else 1.0
                    )
                    batchdata_list[i] = self.transforms(batchdata_list[i])

                for i in range(batchdata_list[0][CommonKeys.IMAGE].size(dim=0)):
                    placeholder_tensor = batchdata_list[0][CommonKeys.IMAGE]
                    placeholder = np.array(placeholder_tensor[i])
                    #print(placeholder)
                    nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/TrainingInnerLoop', f'inputs_iteration_{j}_channel_' + str(i)+'.nii.gz'))

                batchdata = list_data_collate(batchdata_list)
                engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

        # first item in batch only
        engine.state.batch = batchdata

        inputs, _ = engine.prepare_batch(batchdata)

        
        for i in range(inputs.size(dim=1)):
            placeholder_tensor = inputs[0]
            placeholder = np.array(placeholder_tensor[i])
            #print(placeholder)
            nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/Pictures','final_inputs_channel_' + str(i)+'.nii.gz'))
        
        label = batchdata["label"][0]
        placeholder = np.array(label[0])
        nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/Pictures/label.nii.gz'))

        return engine._iteration(engine, batchdata)  # type: ignore[arg-type]
