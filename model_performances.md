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

**03/29/2023, 08:22:49**
Generated with `scripts/evaluate.py --testset_size=500 --n_solutions_for_runtime=100 --all`

|    | Robot     | Model name              |   Mean l2 error (cm) |   Mean angular error (deg) |   Joint limits exceeded % |   Self-colliding % |   Mean runtime for 100 solutions (ms) |   Runtime std (k=5) |   Number of coupling layers |
|---:|:----------|:------------------------|---------------------:|---------------------------:|--------------------------:|-------------------:|--------------------------------------:|--------------------:|----------------------------:|
|  0 | panda     | panda_full_tpm          |             1.06033  |                   4.60678  |                         0 |              5.424 |                               6.76646 |         5.49513e-05 |                          12 |
|  1 | panda     | panda_full_nsc_tpm      |             0.988406 |                   4.05426  |                         0 |              4.632 |                               6.35095 |         7.96247e-05 |                          12 |
|  2 | panda     | panda_lite_tpm          |             1.4245   |                  10.5319   |                         0 |              6.056 |                               3.45173 |         0.000180327 |                           6 |
|  3 | fetch     | fetch_full_temp_tpm     |             1.37692  |                   0.876734 |                         0 |              3.856 |                               7.54085 |         3.28529e-05 |                          12 |
|  4 | fetch     | fetch_full_temp_nsc_tpm |             1.4652   |                   0.709084 |                         0 |              2.98  |                               8.29468 |         8.56386e-05 |                          12 |
|  5 | fetch     | fetch_large_temp        |             1.21257  |                   3.91324  |                         0 |              2.148 |                               9.99708 |         0.000382957 |                          16 |
|  6 | fetch_arm | fetch_arm_full_temp     |             2.19165  |                   6.23809  |                         0 |              3.768 |                               6.78906 |         0.000222861 |                          12 |
|  7 | fetch_arm | fetch_arm_large_temp    |             1.30515  |                   0.696194 |                         0 |              2.068 |                               9.77407 |         1.93214e-05 |                          16 |
|  8 | iiwa7     | iiwa7_full_temp_nsc_tpm |             1.59026  |                   1.96689  |                         0 |              0.016 |                               6.38609 |         5.22252e-05 |                          12 |


**04/03/2023, 18:10:42** | Generated with `python scripts/evaluate.py --testset_size=500 --n_solutions_for_runtime=100 --all`

|    | Robot     | Model name                   |   Mean l2 error (cm) |   Mean angular error (deg) |   Joint limits exceeded % |   Self-colliding % |   Mean runtime for 100 solutions (ms) |   Number of coupling layers |
|---:|:----------|:-----------------------------|---------------------:|---------------------------:|--------------------------:|-------------------:|--------------------------------------:|----------------------------:|
|  0 | panda     | panda__full__lp191_5.25m     |             0.680774 |                   2.15541  |                         0 |              4.564 |                               6.36058 |                          12 |
|  1 | panda     | panda_lite_tpm               |             1.44054  |                   8.47811  |                         0 |              6.052 |                               3.33767 |                           6 |
|  2 | fetch     | fetch_full_temp_tpm          |             1.39091  |                   1.32819  |                         0 |              3.688 |                               6.08006 |                          12 |
|  3 | fetch     | fetch_full_temp_nsc_tpm      |             1.47613  |                   1.15856  |                         0 |              3.044 |                               6.31628 |                          12 |
|  4 | fetch     | fetch_large_temp             |             1.22823  |                   2.481    |                         0 |              2.08  |                               8.28385 |                          16 |
|  5 | fetch_arm | fetch_arm__large__mh186_6.5m |             0.859171 |                   0.406847 |                         0 |              1.828 |                               8.26726 |                          16 |
|  6 | iiwa7     | iiwa7_full_temp_nsc_tpm      |             1.56682  |                   4.33375  |                         0 |              0.028 |                               6.59132 |                          12 |

**04/18/2023, 15:35:13** | Generated with `python scripts/evaluate.py --testset_size=500 --n_solutions_for_runtime=100 --all`

|    | Robot     | Model name                    |   Mean l2 error (cm) |   Mean angular error (deg) |   Joint limits exceeded % |   Self-colliding % |   Mean runtime for 100 solutions (ms) |   Number of coupling layers |
|---:|:----------|:------------------------------|---------------------:|---------------------------:|--------------------------:|-------------------:|--------------------------------------:|----------------------------:|
|  0 | panda     | panda__full__lp191_5.25m      |             0.680774 |                   2.15541  |                         0 |              4.564 |                               5.86753 |                          12 |
|  1 | panda     | panda_lite_tpm                |             1.44054  |                   8.47811  |                         0 |              6.052 |                               4.87037 |                           6 |
|  2 | fetch     | fetch_full_temp_nsc_tpm       |             1.45065  |                   0.7864   |                         0 |              2.976 |                               6.1903  |                          12 |
|  3 | fetch     | fetch__large__ns183_9.75m     |             0.979202 |                   3.64909  |                         0 |              2.192 |                               7.29508 |                          16 |
|  4 | fetch_arm | fetch_arm__large__mh186_9.25m |             0.829996 |                   0.551073 |                         0 |              1.668 |                               7.85046 |                          16 |
|  5 | iiwa7     | iiwa7_full_temp_nsc_tpm       |             1.6165   |                   4.92476  |                         0 |              0.028 |                               6.04854 |                          12 |

**05/02/2023, 23:50:49** | Generated with `python scripts/evaluate.py --testset_size=500 --n_solutions_for_runtime=15000 --all`

|    | Robot     | Model name                    |   Mean l2 error (cm) |   Mean angular error (deg) |   Joint limits exceeded % |   Self-colliding % |   Mean runtime for 15000 solutions (ms) |   Number of coupling layers |
|---:|:----------|:------------------------------|---------------------:|---------------------------:|--------------------------:|-------------------:|----------------------------------------:|----------------------------:|
|  0 | panda     | panda__full__lp191_5.25m      |             0.680773 |                    2.15531 |                         0 |              4.564 |                                 52.8669 |                          12 |
|  1 | panda     | panda_lite_tpm                |             1.44735  |                   10.7018  |                         0 |              5.884 |                                 26.8883 |                           6 |
|  2 | fetch     | fetch_full_temp_nsc_tpm       |             1.41584  |                    1.02989 |                         0 |              2.916 |                                 53.0806 |                          12 |
|  3 | fetch     | fetch__large__ns183_9.75m     |             0.970093 |                    3.03681 |                         0 |              2.124 |                                 71.7826 |                          16 |
|  4 | fetch_arm | fetch_arm__large__mh186_9.25m |             0.813048 |                    0.47009 |                         0 |              1.86  |                                 70.9249 |                          16 |
|  5 | iiwa7     | iiwa7_full_temp_nsc_tpm       |             1.66783  |                    2.5039  |                         0 |              0.012 |                                 53.2426 |                          12 |