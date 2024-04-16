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

#################################################### Imports for computing all of the validation modes, e.g. Dice computation etc.
import csv 
from monai.handlers import MeanDice 
from monai.metrics import DiceHelper, DiceMetric, do_metric_reduction 
from monai.transforms import Activations, AsDiscrete 
from monai.utils import MetricReduction

from datetime import datetime

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
        
        ######################## Code for creating dir for saving output files ##############################
        meta_dict_filename = batchdata["image"].meta["filename_or_obj"][0]
        studies_name = meta_dict_filename.split('/')[meta_dict_filename.split('/').index('datasets') + 1]
        
        output_dir = os.path.join(os.path.expanduser('~'), 'external_validation', studies_name)

            #TODO figure out how to get to save this within the folder that already exists for the save?
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #################################################### Implementation of validation in-the-loop so that we can validate across all modes. ##############################
        #We just do autoseg and deepgrow for now, deepedit (the deepgrow pre-transform one) is already done as part of the default validation).
        if not True: # self.train:
            #If validating, make a diff copy of the params e.g. batchdata so it does not affect the normal validation. For some reason I believe this does not do anything, it still affects the original batchdata, maybe because of the engine..
            
            batchdata_val_deepgrow = batchdata.copy()
            #First we compute the deepgrow based mode validations:
            inputs_val_deepgrow, _ = engine.prepare_batch(batchdata_val_deepgrow)
            inputs_val_deepgrow = inputs_val_deepgrow.to(engine.state.device)

            #Deepgrow eval only:

            # engine.network.eval()
            # with torch.no_grad():
            with engine.mode(engine.network):
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions_val_deepgrow = engine.inferer(inputs_val_deepgrow, engine.network)
                else:
                    predictions_val_deepgrow = engine.inferer(inputs_val_deepgrow, engine.network)
            
            #Now we do autoseg based mode validations:
            batchdata_val_autoseg = batchdata.copy()

            batchdata_list_val_autoseg = decollate_batch(batchdata_val_autoseg, detach=True)
            for i in range(self.num_intensity_channel, self.num_intensity_channel + len(label_names)):
                batchdata_list_val_autoseg[0][CommonKeys.IMAGE][i] *= 0
            batchdata_val_autoseg = list_data_collate(batchdata_list_val_autoseg)
            
            inputs_val_autoseg, _ = engine.prepare_batch(batchdata_val_autoseg)
            inputs_val_autoseg = inputs_val_autoseg.to(engine.state.device)

            
            #Autoseg only:

            # engine.network.eval()
            # with torch.no_grad():
            with engine.mode(engine.network):
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions_val_autoseg = engine.inferer(inputs_val_autoseg, engine.network)
                else:
                    predictions_val_autoseg = engine.inferer(inputs_val_autoseg, engine.network)
            
            
            
            ################# Process the predictions and label for metric computation ###########################
            discretised_label = AsDiscrete(argmax=False, to_onehot=len(label_names))(batchdata['label'][0])

            #Deepgrow
            deepgrow_activation = Activations(softmax=True)(predictions_val_deepgrow[0])
            deepgrow_discretised_pred = AsDiscrete(argmax=True, to_onehot=len(label_names))(deepgrow_activation)
            
            #We will only compute the meandice for now, no need for dice on each individual label currently.., just need to see how it stacks up comparatively to the DeepEdit one (any fluctuation between classes likely shows up on DeepEdit mode also)
            #SplitPredsLabel(keys="pred"), 


            #Autoseg
            autoseg_activation = Activations(softmax=True)(predictions_val_autoseg[0])
            autoseg_discretised_pred = AsDiscrete(argmax=True, to_onehot=len(label_names))(autoseg_activation)
            #SplitPredsLabel(keys="pred"), 



            #TODO: Add the computation of the metrics here:
            deepgrow_dice = DiceHelper(  # type: ignore
                include_background=False,
                reduction=MetricReduction.MEAN,
                get_not_nans=False,
                softmax=False,
                ignore_empty=True,
                num_classes=None,
                )(y_pred=deepgrow_discretised_pred.cpu(), y=discretised_label)
            
            autoseg_dice = DiceHelper(  # type: ignore
                include_background=False,
                reduction=MetricReduction.MEAN,
                get_not_nans=False,
                softmax=False,
                ignore_empty=True,
                num_classes=None,
                )(y_pred=autoseg_discretised_pred.cpu(), y=discretised_label)
            


            ########################### Saving outputs ##########################
            
            

            ################## Saving the metrics #############################
                
            ############# Appending the metric values to csv file ################# 
            # fields = [float(deepgrow_dice), float(autoseg_dice)]    
            # with open(os.path.join(output_dir, 'validation_scores', 'validation.csv'),'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(fields)

            ############### Saving the predictions and labels ################################
            nib.save(nib.Nifti1Image(np.array(deepgrow_discretised_pred[0].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'deepgrow_discretised_pred_channel_0.nii.gz'))
            nib.save(nib.Nifti1Image(np.array(deepgrow_discretised_pred[1].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'deepgrow_discretised_pred_channel_1.nii.gz'))
            
            nib.save(nib.Nifti1Image(np.array(autoseg_discretised_pred[0].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'autoseg_discretised_pred_channel_0.nii.gz'))
            nib.save(nib.Nifti1Image(np.array(autoseg_discretised_pred[1].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'autoseg_discretised_pred_channel_1.nii.gz'))
            
            nib.save(nib.Nifti1Image(np.array(discretised_label[0].cpu()), None), os.path.join(output_dir, 'validation_images_verif','discretised_label_channel_0.nii.gz'))
            nib.save(nib.Nifti1Image(np.array(discretised_label[1].cpu()), None), os.path.join(output_dir, 'validation_images_verif','discretised_label_channel_1.nii.gz')) 
        ######################################################### Normal implementation of DeepEdit++ #########################################################################

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
            
        #Here we print whether the input is on the cuda device from the initialisation:
        logger.info(f'The input image and label directly after the initialisation: Image is on cuda: {batchdata["image"].is_cuda}, Label is on cuda: {batchdata["label"].is_cuda}')

        #Here we use the initial segmentations to generate a prediction (our new previous seg) and generate a new set of inputs with this updated previous seg.
        
        if np.random.choice([True, False], p  = [self.deepedit_probability, 1 - self.deepedit_probability]):
        
            logger.info("Editing mode Inner Subloop")
            for j in range(self.max_interactions):
                inputs, _ = engine.prepare_batch(batchdata)
                #Next line puts the inputs on the cuda device
                inputs = inputs.to(engine.state.device)
                
                if not self.train:
                    if not os.path.exists(os.path.join(output_dir, 'TrainingInnerLoop')):
                        
                        os.makedirs(os.path.join(output_dir, 'TrainingInnerLoop'))

                    batchdata_list = decollate_batch(batchdata, detach=True)
                    for i in range(batchdata_list[0][CommonKeys.IMAGE].size(dim=0)):
                        placeholder_tensor = batchdata_list[0][CommonKeys.IMAGE]
                        placeholder = np.array(placeholder_tensor[i])
                        #print(placeholder)
                        nib.save(nib.Nifti1Image(placeholder, None), os.path.join(output_dir, 'TrainingInnerLoop', f'inputs_prior_prediction_iteration_{j}_channel_' + str(i)+'.nii.gz'))
                    batchdata = list_data_collate(batchdata_list)

                engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
                engine.network.eval()

                #Printing which device the image is on prior to the inner loop prediction.
                logger.info(f'The input image prior to the inner loop inference: Image is on cuda: {inputs.is_cuda}')

                with torch.no_grad():
                    logger.info(f'The model prior to the inner loop inference is on device {next(engine.network.parameters()).device}') 
                    if engine.amp:
                        with torch.cuda.amp.autocast():
                            #Runs the inferer on the cuda device
                            predictions = engine.inferer(inputs, engine.network)
                    else:
                        predictions = engine.inferer(inputs, engine.network)
                batchdata.update({CommonKeys.PRED: predictions})
                
                ##################################################################################################################
                #verification check of the prediction generated by the forward pass using Autoseg/Deepgrow inputs. TODO: Delete this.

                if not self.train:
                    if not os.path.exists(os.path.join(output_dir, 'TrainingInnerLoopPrediction')):
                        os.makedirs(os.path.join(output_dir, 'TrainingInnerLoopPrediction'))

                    for i in range(batchdata[CommonKeys.PRED].size(dim=1)):
                        
                        placeholder_tensor = batchdata[CommonKeys.PRED][0].cpu()
                        placeholder = np.array(placeholder_tensor[i], dtype=np.float32)
                        #print(placeholder)
                        nib.save(nib.Nifti1Image(placeholder, None), os.path.join(output_dir, 'TrainingInnerLoopPrediction', f'predictions_iteration_{j}_channel_' + str(i)+'.nii.gz'))

                    ######################################################################################################

                # decollate/collate batchdata to execute click transforms
                batchdata_list = decollate_batch(batchdata, detach=True)
                #Checking whether pred, Image metatensor and label metatensor are on cuda device here:
                logger.info(f'The pre-click transform inputs: Image is on cuda: {batchdata_list[0]["image"].is_cuda}, Label is on cuda {batchdata_list[0]["label"].is_cuda}, Prediction is on cuda {batchdata_list[0]["pred"].is_cuda}')
                for i in range(len(batchdata_list)):
                    batchdata_list[i][self.click_probability_key] = (
                        (1.0 - ((1.0 / self.max_interactions) * j)) if self.train else 1.0
                    )
                    batchdata_list[i] = self.transforms(batchdata_list[i])

                    ########################################################################################################
                if not self.train:

                    for i in range(batchdata_list[0][CommonKeys.IMAGE].size(dim=0)):
                        placeholder_tensor = batchdata_list[0][CommonKeys.IMAGE]
                        placeholder = np.array(placeholder_tensor[i])
                        #print(placeholder)
                        nib.save(nib.Nifti1Image(placeholder, None), os.path.join(output_dir, 'TrainingInnerLoop', f'inputs_iteration_{j}_channel_' + str(i)+'.nii.gz'))
                        #################################################################################################################

                batchdata = list_data_collate(batchdata_list)
                engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

        
        if not True:  #self.train:
            ######################## we will save the image predictions for the deepedit validation ###############################
            batchdata_val_deepedit = batchdata.copy()
            #First we compute the deepgrow based mode validations:
            inputs_val_deepedit, _ = engine.prepare_batch(batchdata_val_deepedit)
            inputs_val_deepedit = inputs_val_deepedit.to(engine.state.device)
            
            # engine.network.eval()
            # with torch.no_grad():
            with engine.mode(engine.network):
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions_val_deepedit = engine.inferer(inputs_val_deepedit, engine.network)
                else:
                    predictions_val_deepedit = engine.inferer(inputs_val_deepedit, engine.network)
            
            
            
            ################# Process the prediction for metric computation ###########################
            
            #Deepedit
            deepedit_activation = Activations(softmax=True)(predictions_val_deepedit[0])
            deepedit_discretised_pred = AsDiscrete(argmax=True, to_onehot=len(label_names))(deepedit_activation)
            #SplitPredsLabel(keys="pred"), 
        
            deepedit_dice = DiceHelper(  # type: ignore
                include_background=False,
                reduction=MetricReduction.MEAN, #MetricReduction.MEAN,
                get_not_nans=False,
                softmax=False,
                ignore_empty=True,
                num_classes=None,
                )(y_pred=deepedit_discretised_pred.cpu(), y=discretised_label)
            
            #deepedit_dice_reduction = do_metric_reduction(deepedit_dice, MetricReduction.MEAN)[0]

            ########################### Saving outputs ##########################
            
            

            ################## Saving the metrics #############################
                
            ############# Appending the metric values to csv file ################# 
            fields = [float(deepgrow_dice), float(autoseg_dice), float(deepedit_dice)] #[float(deepedit_dice_reduction), float(deepedit_dice_reduction), float(deepedit_dice_reduction)] #    
            with open(os.path.join(output_dir, 'validation_scores', 'validation.csv'),'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

            ############### Saving the predictions and labels ################################

            nib.save(nib.Nifti1Image(np.array(deepedit_discretised_pred[0].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'deepedit_discretised_pred_channel_0.nii.gz'))
            nib.save(nib.Nifti1Image(np.array(deepedit_discretised_pred[1].cpu()), None), os.path.join(output_dir, 'validation_images_verif', 'deepedit_discretised_pred_channel_1.nii.gz'))
            
        # first item in batch only
        # engine.state.batch = batchdata

        # inputs, _ = engine.prepare_batch(batchdata)

        
        # for i in range(inputs.size(dim=1)):
        #     placeholder_tensor = inputs[0]
        #     placeholder = np.array(placeholder_tensor[i])
        #     #print(placeholder)
        #     nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/Pictures','final_inputs_channel_' + str(i)+'.nii.gz'))
        
        # label = batchdata["label"][0]
        # placeholder = np.array(label[0])
        # nib.save(nib.Nifti1Image(placeholder, None), os.path.join('/home/parhomesmaeili/Pictures/label.nii.gz'))

        logger.info(f'For the final inputs the image is on cuda: {batchdata["image"].is_cuda}, the label is on cuda: {batchdata["label"].is_cuda}')
        logger.info(f'For the engine, amp is {engine.amp}')
        return engine._iteration(engine, batchdata)  # type: ignore[arg-type]
