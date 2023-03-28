## Model Performances
Generated with `scripts/evaluate.py --testset_size=50 --n_samples_for_runtime=10 --all`

|    | Robot     | Model name              |   Mean l2 error (cm) |   Mean angular error (deg) |   Joint limits exceeded % |   Self-colliding % |   Mean runtime for 10 solutions (ms) |   Runtime std (k=5) |   Number of coupling layers |
|---:|:----------|:------------------------|---------------------:|---------------------------:|--------------------------:|-------------------:|-------------------------------------:|--------------------:|----------------------------:|
|  0 | panda     | panda_full_tpm          |             1.03469  |                   0.859154 |                         0 |               1.12 |                              7.82676 |         0.00539608  |                          12 |
|  1 | panda     | panda_full_nsc_tpm      |             0.975534 |                   0.751579 |                         0 |               1.48 |                              5.74055 |         6.30143e-05 |                          12 |
|  2 | panda     | panda_lite_tpm          |             1.52715  |                   7.50247  |                         0 |               8.56 |                              2.94867 |         9.43252e-05 |                           6 |
|  3 | fetch     | fetch_full_temp_tpm     |             1.36466  |                   1.19676  |                         0 |               3.12 |                              5.26166 |         4.34817e-05 |                          12 |
|  4 | fetch     | fetch_full_temp_nsc_tpm |             1.46572  |                   0.731544 |                         0 |               2.76 |                              5.79877 |         0.000151532 |                          12 |
|  5 | fetch_arm | fetch_arm_full_temp     |             2.04461  |                   1.75114  |                         0 |               3.64 |                              5.98402 |         0.000197209 |                          12 |
|  6 | iiwa7     | iiwa7_full_temp_nsc_tpm |             1.34085  |                   8.11497  |                         0 |               0    |                              5.83868 |         7.1178e-05  |                          12 |

**03/28/2023, 09:35:32**
Generated with `scripts/evaluate.py --testset_size=500 --n_samples_for_runtime=100 --all`

|    | Robot     | Model name              |   Mean l2 error (cm) |   Mean angular error (deg) |   Joint limits exceeded % |   Self-colliding % |   Mean runtime for 100 solutions (ms) |   Runtime std (k=5) |   Number of coupling layers |
|---:|:----------|:------------------------|---------------------:|---------------------------:|--------------------------:|-------------------:|--------------------------------------:|--------------------:|----------------------------:|
|  0 | panda     | panda_full_tpm          |             1.06033  |                   4.60678  |                         0 |              5.424 |                               6.0245  |         5.52396e-05 |                          12 |
|  1 | panda     | panda_full_nsc_tpm      |             0.988406 |                   4.05426  |                         0 |              4.632 |                               6.42967 |         1.65477e-05 |                          12 |
|  2 | panda     | panda_lite_tpm          |             1.4245   |                  10.5319   |                         0 |              6.056 |                               4.13961 |         1.12562e-05 |                           6 |
|  3 | fetch     | fetch_full_temp_tpm     |             1.37692  |                   0.876734 |                         0 |              3.856 |                               8.41703 |         1.67176e-05 |                          12 |
|  4 | fetch     | fetch_full_temp_nsc_tpm |             1.4652   |                   0.709084 |                         0 |              2.98  |                               8.50849 |         1.73738e-05 |                          12 |
|  5 | fetch     | fetch_large_temp        |             1.21257  |                   3.91324  |                         0 |              2.148 |                               9.49244 |         2.08031e-05 |                          16 |
|  6 | fetch_arm | fetch_arm_full_temp     |             2.19165  |                   6.23809  |                         0 |              3.768 |                               8.51307 |         1.04182e-05 |                          12 |
|  7 | fetch_arm | fetch_arm_large_temp    |             1.9145   |                   1.82625  |                         0 |              2.092 |                               9.63984 |         0.000216909 |                          16 |
|  8 | iiwa7     | iiwa7_full_temp_nsc_tpm |             1.59026  |                   1.96689  |                         0 |              0.016 |                               6.12326 |         3.15216e-05 |                          12 |