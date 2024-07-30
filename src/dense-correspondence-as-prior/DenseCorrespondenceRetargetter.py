## Code to retarget using dense correspondence
## Usage: python DesenseCorrespondenceRetargetter.py 

import os
import sys
import argparse
import numpy as np 

# Insert the local directory at the beginning of sys.path
sys.path.insert(0, 'smplx/smplx')

# Now import the package
import smplx
import torch


# Load from modules 
from constants import parser,get_logger
from renderer import Visualizer


class DenseCorrespondenceRetargetter:

    def load(): 
        pass

    def retarget(self,source,target):
        pass




def main(args):

    # Create experiement directory 
    args.out_dir  = os.path.join(args.out_dir,args.exp_name)
    os.makedirs(args.out_dir,exist_ok=True)

    # Create a logger to print and log resutlts
    logger = get_logger(out_dir=args.out_dir,debug=args.debug)
    logger.info("Creating Experiment with arguements: {}".format(args))

    if os.path.exists(os.path.join(args.out_dir,f'{args.exp_name}.pkl')) and not args.force: # If data already exists and we are not forcing to overwriting
        logger.info(f"Setup already exists at:{os.path.join(args.out_dir,f'{args.exp_name}.pkl')}. Skipping")    
        return 



    visualizer = Visualizer() if args.render else None #Create a polyscope visualizer if we are rendering

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # Create source mesh
    # Create arguments for source mesh


if __name__ == '__main__':

    cmd_line_args = parser.parse_args()
    main(cmd_line_args)

    