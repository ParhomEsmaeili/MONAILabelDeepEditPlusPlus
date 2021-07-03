import json
import logging
import os

from lib import MyInfer, MyStrategy, MyTrain
from monai.networks.nets import DynUNet

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.network = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            kernel_size=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            upsample_kernel_size=[
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 1],
            ],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        self.pretrained_model = os.path.join(self.model_dir, "segmentation_whole_heart.pt")
        self.final_model = os.path.join(self.model_dir, "final.pt")
        self.train_stats_path = os.path.join(self.model_dir, "train_stats.json")

        path = [self.pretrained_model, self.final_model]
        infers = {
            "segmentation_whole_heart": MyInfer(path, self.network),
        }

        strategies = {
            "random": Random(),
            "first": MyStrategy(),
        }

        resources = [
            (
                self.pretrained_model,
                "https://api.ngc.nvidia.com/v2/models/nvidia/med"
                "/clara_pt_liver_and_tumor_ct_segmentation/versions/1/files/models/model.pt",
            ),
        ]

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            infers=infers,
            strategies=strategies,
            resources=resources,
        )

        # # Simple way to Add deepgrow 2D+3D models for infer tasks
        self.add_deepgrow_infer_tasks()

    def train(self, request):
        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, request.get("name", "model_01"))

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        # Use pretrained weights to start training?
        load_path = (
            load_path
            if os.path.exists(load_path)
            else self.pretrained_model
            if request.get("pretrained", True)
            else None
        )

        # Datalist for train/validation
        train_d, val_d = self.partition_datalist(self.datastore().datalist(), request.get("val_split", 0.2))

        task = MyTrain(
            output_dir=output_dir,
            train_datalist=train_d,
            val_datalist=val_d,
            network=self.network,
            load_path=load_path,
            publish_path=self.final_model,
            stats_path=self.train_stats_path,
            device=request.get("device", "cuda"),
            lr=request.get("lr", 0.0001),
            val_split=request.get("val_split", 0.2),
            max_epochs=request.get("epochs", 1),
            amp=request.get("amp", True),
            train_batch_size=request.get("train_batch_size", 1),
            val_batch_size=request.get("val_batch_size", 1),
        )
        return task()

    def train_stats(self):
        if os.path.exists(self.train_stats_path):
            with open(self.train_stats_path, "r") as fc:
                return json.load(fc)
        return super().train_stats()