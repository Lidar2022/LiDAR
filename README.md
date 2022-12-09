##Introduction

Complex-YOLOv3 is a 3D Object Detection framework based on deep learning with the help of Convolutional Neural Networks. It takes in a birds-eye-view representation which is projected from 3D LiDAR point cloud and applies a YOLOV3 architecture. Furthermore, it utilizes a Euler Regional-Proposal Network for reliable angle regression to detect accurate multi-class oriented 3D objects. 

Real-time Asset Inventory is a practice that allows asset-intensive industries e.g. Transportation, healthcare, manufacturing etc., to instantly locate physical assets both indoors and outdoors, monitor their condition and usage and optimize asset allocation. Millions of dollars in losses are caused by the misplacement and destruction of road assets, adopting an effective 3D Object Detection model, Companies can achieve accurate and efficient real-time detection of valuable assets and thereby drastically reducing these losses. 

##Dataset

We made use of the KITTI dataset. It is a well known benchmark dataset collected for the autonomous driving platform. The sensors include a wide-angle camera and a Velodyne HDL-64E LiDAR. The dataset contains both 2D and 3D annotations of cars, pedestrians, and cyclists. It contains 7481 training images and 7518 testing images including calibration files.

Download the 3D KITTI detection dataset from here.

  https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d


##Data Preparation:

  The downloaded data includes:

    Velodyne point clouds (29 GB): input data to the Complex-YOLO model
    Training labels of object data set (5 MB): input label to the Complex-YOLO model
    Camera calibration matrices of object data set (16 MB): for visualization of  predictions
    Left color images of object data set (12 GB): for visualization of predictions
    
##How to run

  Visualize the dataset (both BEV images from LiDAR and camera images)

      cd src/data_process

   To visualize BEV maps and camera images (with 3D boxes), let's execute (the output-width param can be changed to show the images in a bigger/smaller window):
   
      python kitti_dataloader.py --output-width 608

   To visualize mosaics that are composed from 4 BEV maps (Using during training only), let's execute:

      python kitti_dataloader.py --show-train-data --mosaic --output-width 608 

   By default, there is no padding for the output mosaics, the feature could be activated by executing:

      python kitti_dataloader.py --show-train-data --mosaic --random-padding --output-width 608 

  To visualize cutout augmentation, let's execute:

      python kitti_dataloader.py --show-train-data --cutout_prob 1. --cutout_nholes 1 --cutout_fill_value 1. --cutout_ratio 0.3 --output-width 608

# Inference

    Download the trained model from here, then put it to ${ROOT}/checkpoints/ and execute:

    python test.py --gpu_idx 0 --pretrained_path../checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth --cfgfile./config/cfg/complex_yolov4.cfg --show_image

 # Evaluation

python evaluate.py --gpu_idx 0 --pretrained_path <PATH> --cfgfile <CFG> --img_size <SIZE> --conf-thresh <THRESH> --nms-thresh <THRESH> --iou-thresh <THRESH>
(The conf-thresh, nms-thresh, and iou-thresh params can be adjusted. By default, these params have been set to 0.5)

 # Training

    Single machine, single gpu

      python train.py --gpu_idx 0 --batch_size <N> --num_workers <N>...

  #Multi-processing Distributed Data Parallel Training

    We should always use the nccl backend for multi-processing distributed training since it currently provides the best distributed training performance.

    #Single machine (node), multiple GPUs

      python train.py --dist-url 'tcp://127.0.0.1:29500' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

    #Two machines (two nodes), multiple GPUs

      #First machine

        python train.py --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0

      #Second machine

        python train.py --dist-url 'tcp://IP_OF_NODE2:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1

 # To reproduce the results, you can run the bash shell script

    ./train.sh

 ## Tensorboard

    To track the training progress, go to the logs/ folder and

      cd logs/<saved_fn>/tensorboard/

      tensorboard --logdir=./

    Then go to http://localhost:6006/:
  
## Folder structure
```   
${ROOT}
└── checkpoints/    
    ├── complexer_yolo/
        └── Model_complexer_yolo_epoch_4280.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ <-- for visualization
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ <-- for visualization
        │   ├── calib/
        │   └── velodyne/ 
        └── classes_names.txt
└── src/
    ├── config/
    ├── cfg/
        │   ├── complex_yolov3.cfg
        │   ├── complex_yolov3_tiny.cfg
        │   ├── complex_yolov4.cfg
        │   ├── complex_yolov4_tiny.cfg
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_bev_utils.py
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   ├── kitti_data_utils.py
    │   ├── train_val_split.py
    │   └── transformation.py
    ├── models/
    │   ├── darknet2pytorch.py
    │   ├── darknet_utils.py
    │   ├── model_utils.py
    │   ├── yolo_layer.py
    └── utils/
    │   ├── evaluation_utils.py
    │   ├── iou_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── evaluate.py
    ├── test.py
    ├── test.sh
    ├── train.py
    └── train.sh
├── README.md 
└── requirements.txt
```
