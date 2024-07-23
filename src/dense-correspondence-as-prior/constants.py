import os
import sys 
import random
import numpy as np
import argparse
import logging
import torch
import platform


########################## GET SYSTEM Information ##########################
SYSTEM_OS = platform.system()

############################ Command Line Arguments ############################################
# Create the argument parser
parser = argparse.ArgumentParser(
				prog=' Retargetting Parameteric models',
				description='This folder contains the experiments used to determine whether dense correspondence between 3D objects can be used to determine whether dense shape correspondence methods, such as [SpiderMatch](!https://paulroetzer.github.io/publications/2024-06-19-spidermatch.html), or [functional maps](!https://dl.acm.org/doi/10.1145/2185520.2185526) can be used to retarget motion.',
				epilog='')


# Source and target parametric models
parser.add_argument('--source',type=str, default="SMPL", help="Source mesh")  
parser.add_argument('--target',type=str, default="SMPL", help="Target mesh")  

# Experiment params
parser.add_argument('--exp-name',type=str, default="Exp-1", help="Experiment name")  
parser.add_argument('--out-dir',type=str, default="./save", help="Location where all experiements are saved")  
parser.add_argument('--batch-size',type=int, default=10, help="Location where all experiements are saved")  

# Experiment Variables 
parser.add_argument('--pose',type=str, default="", help="Motion to be retargetted")  
parser.add_argument('--shape', type=str, default="", help="Whether to change the shape of the target model")

parser.add_argument('--distortions', dest='distortions', nargs='?', const=None, default=None, help="Distortions allowed")
parser.add_argument('--distorion', type=str, default="", help="Whether to change the shape of the target model")

parser.add_argument('-f', '--force',action='store_true',help="forces a re-run on retargetting even if pkl file containg smpl data is already present.")  # on/off flag

parser.add_argument('--render', action='store_true', default=True, help="Render a video and save it it in RENDER_DIR. Can also be set in the utils.py")  # on/off flag

parser.add_argument('--debug', dest='debug', action='store_true',  help="Debug and run on less number of frames")
parser.add_argument('--no-debug', dest='debug', action='store_false',default=False, help="Debug and run on less number of frames")

parser.add_argument('--gpu', dest='gpu', action='store_true', default=True,help="Whether to use CUDA or CPU")
parser.add_argument('--no-gpu', dest='gpu', action='store_false',help="Whether to use CUDA or CPU")



############################# LIBRARY IMPORTS ####################################################
assert __file__[-1] != '/' , f'File:{__file__}, cannot be parsed' 
SRC_DIR,_ = os.path.split(os.path.abspath(__file__))
# HOME_DIR,_ = os.path.split(SRC_DIR)
# RABIT_DIR  = os.path.join(HOME_DIR,'RaBit')
# STYLEGAN_DIR = os.path.join(RABIT_DIR,'stylegan3')
# sys.path.extend([HOME_DIR,RABIT_DIR,STYLEGAN_DIR])

############################# FOLDER PATHS #######################################################
# if os.path.isdir(os.path.join(HOME_DIR,'MCS_DATA')): 
#     DATA_DIR = os.path.join(HOME_DIR,'MCS_DATA')
# elif os.path.isdir(os.path.join(HOME_DIR,'data')): 
#     DATA_DIR = os.path.join(HOME_DIR,'data')
# else: 
#     raise FileNotFoundError("Unable to find directory containing the MCS dataset")

# INPUT_DIR = os.path.join(DATA_DIR,'OpenSim') # Path containing all the training data (currently using xyz)
# SMPL_DIR = os.path.join(DATA_DIR,'SMPL')
# RENDER_DIR = os.path.join(DATA_DIR,'rendered_videos')
# LOG_DIR = os.path.join(DATA_DIR,'logs')
# SEGMENT_DIR = os.path.join(DATA_DIR,'segments')
# PKL_DIR = os.path.join(DATA_DIR,'pkl')
# HUMANML_DIR = os.path.join(DATA_DIR,'humanml3d')



########################## SET RANDOM SEEDS #################################
seed = 69
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############################# DATASET CONSTANTS #######################################################
smpl_joints = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand'
]

ROOT_INIT_ROTVEC = np.array([0,np.pi/2,0])

##################################### RABIT Parameters #######################################
smpl2rabit_mapping = [ 0, # Center hip 
					   3, # spine0
					   6, # spine1
					   9, # spine3 
					   14,# right chest 
					   17,# right shoulder 
					   19,# right elblow 
					   21,# right ability ? 
					   23,# right hand 
					   12,# neck1
					   15,# neck2
					   2, # r hip 
					   5, # r knee
					   1, # l hip 
					   4, # l knee
					   8, # r ankle 
					   11,# r foot 
					   13,# l chest 
					   16,# lshoudler
					   18,# elbow
					   20,# lability 
					   22,# lhand
					   7, # lankle
					   10,# lfoor
					 ]

############################# LOGGING #######################################################
class CustomFormatter(logging.Formatter):

	BLACK = '\033[0;30m'
	RED = '\033[0;31m'
	GREEN = '\033[0;32m'
	BROWN = '\033[0;33m'
	BLUE = '\033[0;34m'
	PURPLE = '\033[0;35m'
	CYAN = '\033[0;36m'
	GREY = '\033[0;37m'

	DARK_GREY = '\033[1;30m'
	LIGHT_RED = '\033[1;31m'
	LIGHT_GREEN = '\033[1;32m'
	YELLOW = '\033[1;33m'
	LIGHT_BLUE = '\033[1;34m'
	LIGHT_PURPLE = '\033[1;35m'
	LIGHT_CYAN = '\033[1;36m'
	WHITE = '\033[1;37m'

	RESET = "\033[0m"

	format = "[%(filename)s:%(lineno)d]: %(message)s (%(asctime)s) "

	FORMATS = {
		logging.DEBUG: YELLOW + format + RESET,
		logging.INFO: GREEN + format + RESET,
		logging.WARNING: LIGHT_RED + format + RESET,
		logging.ERROR: RED + format + RESET,
		logging.CRITICAL: RED + format + RESET
	}

	def format(self, record):
		log_fmt = self.FORMATS.get(record.levelno)
		formatter = logging.Formatter(log_fmt)
		return formatter.format(record)

def get_logger(out_dir=None,debug=False,return_writer=False):

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.DEBUG if debug else logging.INFO)

	if os.path.exists(out_dir):
		handler = logging.FileHandler(os.path.join(out_dir,"log.txt"))
		handler.setLevel(level=logging.DEBUG if debug else logging.INFO)
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)

	handler = logging.StreamHandler()
	handler.setLevel(level=logging.DEBUG if debug else logging.INFO)
	handler.setFormatter(CustomFormatter())
	logger.addHandler(handler)

	if not return_writer:
		return logger

	else:
		try: 
			from tensorboardX import SummaryWriter
			writer = SummaryWriter(out_dir)

		except ModuleNotFoundError:
			logger.warning("Unable to load tensorboardX to write summary.")
			writer = None

		return logger, writer