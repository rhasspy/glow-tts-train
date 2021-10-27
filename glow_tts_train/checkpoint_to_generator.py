#!/usr/bin/env python3
import sys

import torch

assert len(sys.argv) == 3, "IN OUT"
c = torch.load(sys.argv[1], map_location="cpu")

model = {}

prefix = "generator."
for key, value in c["state_dict"].items():
    if key.startswith(prefix):
        new_key = key[len(prefix) :]
        model[new_key] = value
        print(key, new_key, sep=" -> ")

torch.save({"model": model}, sys.argv[2])
