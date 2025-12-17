# TerraLock
A Geographic-Aware UAV Pose Estimation System

./main.py:Entry function, containing an adaptive global matching algorithm.

./location_optism.py:Relating to pose optimization, it encompasses the core logic of an elevation-constrained decoupled optimization method based on BA optimization techniques and a grid-based random sampling arbitration mechanism.

./hight_load.py:Regarding the reading of elevation data.

./globleFind.py:Regarding the satellite database that matches the global scope.

./DataLogger.py:Data logging during project operation.

./make_dataSet/make_hight_data.py:Create an elevation database.

./make_dataSet/makePicture_online.py:Create an online satellite image database for global matching applications.

./make_dataSet/makePicture_train_general.py:Creating a universal dataset for training global matching models.

./make_dataSet/makePicture_train_Task.py:Task Area Dataset Used for Training the Global Matching Model.

./make_dataSet/mapLocation.py:Load true value data for the satellite image database.

If you need to run this project, you will require the MixVPR project and the GIM project. Their GitHub links are:
  https://github.com/amaralibey/MixVPR.git
  
  https://github.com/xuelunshen/gim.git
Place its project files in the ./ directory.And enter the correct dataset and file paths into the mian.py file.

