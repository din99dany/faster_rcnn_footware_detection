# Before you start

Before being able to start training the model on the cloud you need to do a few things

## Import additional files
In order for `train.py` to be able to run, you will need to download some extra files from git and place them in same folder as `train.py`. This is done because we use some logic from these older files. To do this, please run the following commands in the directory where `train.py` is

```bash
git clone https://github.com/pytorch/vision.git

cd vision
git checkout v0.14.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
```

## Create the dataset
Please import your dataset in AzureML. To do that you can use the tab on the left side of the studio.

## Create the compute resource
In order to train the model in a distributed manner, you need a cluster to train on. you can create such cluster from the menu on the left of the studio. Once you create it, please update the details in `azure_ml_job_config.ipynb`