# For Face Recognition Team
This is a slightly tiny project espacailly for face recognition,
where we seperate several parts(networks,metrics & loss functions, data loaders) into different folders accordingly. moreover, in order to be extensible, 
major training modules are written in `modules.py` which includes:
* `IOFactory` for logging, reading and saving.
* `OptimFactory` for params optimization and lr scheduling
* `Header` for pairwise metric learning or metrics with classification.
In this project ,we use colPallelLinear and mpu vocab  pallel cross entropy to build a pallel version Face train project.
1 To get training data,you can download from web link in dbfile.txt
2 we have make a scripy to convert images in folders to tfrecord files,which can help speed up data read in HDD disk.the scripy is data/create_tfrecord.py. Also if you have fast SSD disk, ignore this step. rebuild the dataloader in solver_ddp.py.
3 The overall training protocol has been written in file solver_ddp.py, while the configurations are set in file config.yaml, using shell run_face_main.sh to train a model.Make sure set right host IP for model pallel env.
4 After training ,you can use validation/vertification_owndata.py to get vertification performance of self-models on agedb_30 lfw and cfp_fp.(the validation has provided in dbfile.txt) 
5 Experiment shows a consistent result with InsightFace, some more networks will be added!

# Compatibility
The code has been tested using Pytorch 1.4.0 under Centos7.4 with python3.6.
 we reommand use docker image horovod/horovod:0.19.3-tf2.1.0-torch-mxnet1.6.0-py3.6-gpu .
