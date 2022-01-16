import numpy as np 
import random 
import json 

import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 

from pipeline_NLP import bow, tokenize, stem 
from model import NeuralNet


with open('intents.json', 'r') as f:
    intents = json.load(f)
    print(intents)