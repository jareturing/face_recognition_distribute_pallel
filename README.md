# For Fenghuo Face Recognition Team
This is a slightly tiny project espacailly for face recognition,
where we seperate several parts(networks,metrics & loss functions, data loaders) into different folders accordingly. moreover, in order to be extensible, 
major training modules are written in `modules.py` which includes:
* `IOFactory` for logging, reading and saving.
* `OptimFactory` for params optimization and lr scheduling
* `Header` for pairwise metric learning or metrics with classification

The overall training protocol has been written in file solver.py, while the configurations are set in file config.yaml, before running `python solver.py --cfg config.yaml`
to train a model, you may just feel free to change and config you own. 

To keep up with Sansuo evaluation metrics, we add the validation module which can be implemented by running `python evaluation.py --cfg validation.config.Config` in validation folder, which includes face verification(1vs1) and identification(1vsN), 
considering evaluation may take some time (50 min around), we didn't insert it into solver.py,.

Experiment shows a consistent result with InsightFace, some more networks will be added!

# Compatibility
The code has been tested using Pytorch r1.2.0 under Ubuntu16 with python3.6

