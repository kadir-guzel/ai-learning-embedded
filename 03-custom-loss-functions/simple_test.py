#!/usr/bin/env python3
"""
Simple PyTrim Test File
=======================
This file has only a few used imports and several unused ones
that PyTrim should definitely detect and remove.
"""

# USED IMPORTS
import torch
import torch.nn as nn

# UNUSED IMPORTS - PyTrim should remove these
import numpy as np
import pandas as pd
import scipy.stats
import requests
import cv2
import sklearn
from transformers import pipeline
import tensorflow as tf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def main():
    model = SimpleModel()
    x = torch.randn(5, 10)
    output = model(x)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()