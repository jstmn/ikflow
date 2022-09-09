import torch

# Get an arbitrary model and optimizer
class crazymodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Linear(10, 10)


nn_model = crazymodel()
optimizer = torch.optim.Adadelta(nn_model.parameters(), lr=1.0)

# convenience function
def get_lr(_optimizer) -> float:
    lrs = []
    for param_group in _optimizer.param_groups:
        lrs.append(param_group["lr"])
    assert len(set(lrs)) == 1, f"Error: Multiple learning rates found. There should only be one. lrs: '{lrs}'"
    return lrs[0]


# Setup lr scheduler
STEP_EVERY = 10
GAMMA = 0.5
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_EVERY, gamma=GAMMA, verbose=False)
print(f"STEP_EVERY: {STEP_EVERY}")
print(f"GAMMA:      {GAMMA}\n")

# Look at results
for global_step in range(50):
    lr_scheduler.step()
    print(global_step, get_lr(optimizer))


""" Example output
STEP_EVERY: 10
GAMMA:      0.5

/home/jeremysmorgan/.local/share/virtualenvs/ikflow-yYw_PNKQ/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:131: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
0 1.0
1 1.0
2 1.0
3 1.0
4 1.0
5 1.0
6 1.0
7 1.0
8 1.0
9 0.5
10 0.5
11 0.5
12 0.5
13 0.5
14 0.5
15 0.5
16 0.5
17 0.5
18 0.5
19 0.25
20 0.25
21 0.25
22 0.25
23 0.25
24 0.25
25 0.25
26 0.25
27 0.25
28 0.25
29 0.125
30 0.125
31 0.125
32 0.125
33 0.125
34 0.125
35 0.125
36 0.125
37 0.125
38 0.125
39 0.0625
40 0.0625
41 0.0625
42 0.0625
43 0.0625
44 0.0625
45 0.0625
46 0.0625
47 0.0625
48 0.0625
49 0.03125
"""
