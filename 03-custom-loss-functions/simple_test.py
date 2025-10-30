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
from transformers import pipeline

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