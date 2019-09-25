import argparse
import json
import os
from pathlib import Path
import torch
from statnlp.hypergraph.NetworkConfig import LossType

class Config:
    def __init__(self, experiment_id, mainfolder): 
        ## Checkpoint 
        self.id = experiment_id
        self.name = 'id_' + str(self.id)
        self.mainfolder = mainfolder
        self.checkpoint = os.path.join(self.mainfolder, 'ckpts', self.name+".pth.tar")
        self.best_checkpoint = os.path.join(self.mainfolder, 'ckpts', 'BEST_'+self.name+".pth.tar")
        self.saved_model_path = os.path.join(self.mainfolder, 'ckpts', 'BEST_'+self.name+".pth.tar")
        self.log_path = os.path.join(self.mainfolder, 'logs', self.name + '_tb') # store tfevents generated from tensorboard & predicted test file
        self.log_file = os.path.join(self.mainfolder, 'logs', self.name+'.log') # log files printed from console
        self.pred_out_dir = os.path.join(self.mainfolder, 'logs', self.name + '_pred')
        self.train_file = os.path.join(self.mainfolder, 'cached-data', 'train.pth') # currently not in use 
        self.dev_file = os.path.join(self.mainfolder, 'cached-data', 'dev.pth') # currently not in use
        self.test_file = os.path.join(self.mainfolder, 'cached-data', 'test.pth') # currently not in use

        ## Data parameters
        self.data = "data/pdtb-ptb-dataset"
        self.train_file =  os.path.join(self.data, "train.txt")
        self.dev_file = os.path.join(self.data, "dev.txt")
        self.test_file = os.path.join(self.data, "test.txt")
        self.num_train = len(open(self.train_file).readlines())
        self.num_dev = len(open(self.dev_file).readlines())
        self.num_test = len(open(self.test_file).readlines())
        #self.glove = 'data/glove.6B.100d.txt' # currently not in use
        self.glove = None

        ## Pretrained models

        ## Training parameters
        self.NEURAL_LEARNING_RATE = 0.015
        self.lr_decay = 0.05
        self.l2=1e-08
        self.LOSS_TYPE = "SSVM"
        self.torch_seed = 1234
        self.numpy_seed = 1234
        self.num_iter = 40
        self.batch_size = 1

        ## Model parameters
        self.DEFAULT_CAPACITY_NETWORK = [1000, 1000, 1000, 1000, 1000]
        self.BUILD_GRAPH_WITH_FULL_BATCH = True
        self.IGNORE_TRANSITION = False
        self.HYPEREDGE_ORDER = 2

        ## Addtional parameters
        self.DEBUG = False
        self.visual = False
        self.TRIAL = False
        self.check_every = self.num_train//2 # how many training instance per evaluation
        self.ECHO_TRAINING_PROGRESS = 100
        self.ECHO_TEST_RESULT_DURING_EVAL_ON_DEV = False
        self.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING = True

        ## Computational resources
        #self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.DEVICE = "cpu" 
        #self.DEVICE = torch.device("cuda:" + args.gpuid)
        self.torch_number_threads = 40
        self.num_thread = 35 # any number > 1

        ## Extra message
        self.message = " \
                "
        
    def save(self):
        json_file = os.path.join(self.mainfolder, 'configs', self.name+'.json')
        with open(json_file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        ## copy over the config file
        os.system("cp config.py " + os.path.join(mainfolder, "configs", self.name+'.py') ) 

    def load(self):
        json_file = os.path.join(self.mainfolder, 'configs', self.name+'.json')
        if not Path(json_file).exists():
            raise Exception('json file is not generated')
        with open(json_file) as f:
            self.__dict__.update(json.load(f))

if __name__ == '__main__':
    ## Parameters to change
    parameter = ''
    parameter_choice = ['']
    experiment_folder = 'constraint-tree'
    config_id = 3

    ## creating folders
    assert(os.path.isdir(os.path.join(os.getcwd(), 'experiments')))
    mainfolder = os.path.join(os.getcwd(), 'experiments/', experiment_folder)
    if not os.path.isdir(mainfolder): 
        os.mkdir(mainfolder); 
        print(f'make new dir: {mainfolder}')
    subfolders = ['ckpts', 'cached-data', 'configs', 'logs']
    for sub in subfolders:
        subfolder = os.path.join(mainfolder, sub)
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)


    
    if config_id == None:
        # change parameters
        for i, option in enumerate(parameter_choice):
            config = Config(i, mainfolder)
            setattr(config, parameter, option) 
            config.save()
    else:
        config = Config(config_id, mainfolder)
        setattr(config, parameter, parameter_choice[0])
        config.save()
