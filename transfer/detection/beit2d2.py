#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if "state_dict" in obj:
        obj = obj["state_dict"]
    elif "model" in obj:
        obj = obj["model"]
    else:
        raise Exception("Cannot find the model in the checkpoint.")
    
    new_model = {}
    for k, v in obj.items():
        old_k = k
        k = 'backbone.net.' + k
        print(old_k, "->", k)
        new_model[k] = v
    
    res = {"model": new_model}

    torch.save(res, sys.argv[2])
