import sys
from main import *


if __name__ == "__main__":
    print(sys.argv)
    config = load_config(True,config_path=sys.argv[1])
    ckpt = None
    if len(sys.argv) >= 3:
        ckpt = sys.argv[2]
    train_model(config=config, ckpt=ckpt)
    
    # good luck my friend, i know you are going to change this code
    # to something better, but for now this is what we have
    # if you want to run this code, you need to have the config file in the same directory
    # and the data in the correct format
    # and the model should be defined in the main.py file