# Data Generation

This folder is meant to gather all of the scripts that will generate data files. All of 
the data generated here will be saved under the `input/data/` folder. 

### Sample data
In this folder you can find scripts to create sample data so you can test functions, classes, 
transformers and everything else you are developing. These scripts have prefix `sample_`
and can be simply run by using `$ python sample_<algorithm>.py`. 

### Train data generation
In some cases, you won't be provided a unique train data, but multiple files or 
multiple data sources that need to be joined to create a data file to be passed 
in your model training.
 
This folder has a template for the creationg of a train data file, the `template_train_data.py`. 
It has instructions on how to fill it in it. Once you have finished, save the new 
script with the name of `make_train_data.py` in this folder. 
