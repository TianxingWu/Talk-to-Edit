import numpy as np
import copy
# import torch

np.random.seed(6)

MAX_LENGTH = 5
NUM_PARTS = 5

candidates_all = [[j for j in range(NUM_PARTS)] for i in range(MAX_LENGTH)]

edit_seq_list = []
for i in range(NUM_PARTS):
    candidates = copy.deepcopy(candidates_all)
    for j in range(MAX_LENGTH):
        edit_seq = [i]
        for k in range(j):
            k = k+1
            x = np.random.choice(list(set(candidates[k]) - set(edit_seq)))
            candidates[k].pop(candidates[k].index(x))
            edit_seq.append(x)
        edit_seq_list.append(edit_seq)
        print(edit_seq)

print(edit_seq_list)