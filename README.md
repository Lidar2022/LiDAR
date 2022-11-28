# Surveying and Real-time Asset and Inventory Management practices in Civil engineering are often associated with time consuming, labor intensive and financially demanding exercises. These exercises include manual inspections, disruption of traffic, safety concerns and the use of semi-automated equipment/technology that often results in inefficient data collection processes.
# The consequences of these inefficient processes are misallocation of resources, protracted projects, and reduction in profits. To alleviate this problem 2D detection algorithms have been utilized in the industry for data collection purposes.
# However, there are also limitations in using 2-D Object Detection Algorithms, as we are unable to capture the dimensionalities, localize the object position and understand the depth of the object. These problems are critical to object detection as cameras will only be able to depict the surroundings in 2D and they are also affected by lighting and weather conditions.
# In conclusion, surveying practices are still outdated and there is an immediate need to adopt 3D object detection algorithms to make accurate data driven decisions.​


Data Preparation:
  Download the 3D KITTI detection dataset from here.

  The downloaded data includes:

    Velodyne point clouds (29 GB): input data to the Complex-YOLO model
    Training labels of object data set (5 MB): input label to the Complex-YOLO model
    Camera calibration matrices of object data set (16 MB): for visualization of  predictions
    Left color images of object data set (12 GB): for visualization of predictions
    
How to run

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

 Inference

    Download the trained model from here, then put it to ${ROOT}/checkpoints/ and execute:

    python test.py --gpu_idx 0 --pretrained_path../checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth --cfgfile./config/cfg/complex_yolov4.cfg --show_image

  Evaluation

python evaluate.py --gpu_idx 0 --pretrained_path <PATH> --cfgfile <CFG> --img_size <SIZE> --conf-thresh <THRESH> --nms-thresh <THRESH> --iou-thresh <THRESH>
(The conf-thresh, nms-thresh, and iou-thresh params can be adjusted. By default, these params have been set to 0.5)

  Training

    Single machine, single gpu

      python train.py --gpu_idx 0 --batch_size <N> --num_workers <N>...

  Multi-processing Distributed Data Parallel Training

    We should always use the nccl backend for multi-processing distributed training since it currently provides the best distributed training performance.

    Single machine (node), multiple GPUs

      python train.py --dist-url 'tcp://127.0.0.1:29500' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

    Two machines (two nodes), multiple GPUs

      First machine

        python train.py --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0

      Second machine

        python train.py --dist-url 'tcp://IP_OF_NODE2:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1

  To reproduce the results, you can run the bash shell script

    ./train.sh

  Tensorboard

    To track the training progress, go to the logs/ folder and

      cd logs/<saved_fn>/tensorboard/

      tensorboard --logdir=./

    Then go to http://localhost:6006/:
