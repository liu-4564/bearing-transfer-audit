## Table 1  Within-dataset accuracy (mean over seeds where available)

| Dataset | Split | Norm | Feature + GBM | 1D-CNN | 1D-ResNet |
|---|---|---|---|---|---|
| CWRU | random | window | 1.000 | 1.000 | 1.000 |
| CWRU | random | global | 0.999 | 1.000 | 1.000 |
| CWRU | run | window | 1.000 | 1.000 | 1.000 |
| CWRU | run | global | 0.999 | 1.000 | 1.000 |
| MFPT | random | window | 1.000 | 0.996 | 1.000 |
| MFPT | random | global | 1.000 | 0.996 | 0.996 |
| MFPT | run | window | 0.854 | 1.000 | 0.993 |
| MFPT | run | global | 0.803 | 0.861 | 0.814 |
| PU | random | window | 1.000 | 1.000 | 1.000 |
| PU | random | global | 1.000 | 1.000 | 1.000 |
| PU | run | window | 1.000 | 1.000 | 1.000 |
| PU | run | global | 1.000 | 1.000 | 1.000 |

## Table 2  Cross-dataset balanced accuracy vs. target samples per class k

| Direction | Norm | Model | k=0 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|---|
| CWRU → MFPT | window | Feature + GBM | 0.067 | 0.715 ± 0.148 | 0.694 ± 0.177 | 0.753 ± 0.175 |
| CWRU → MFPT | window | 1D-CNN | 0.129 | 0.339 ± 0.153 | 0.379 ± 0.143 | 0.667 ± 0.011 |
| CWRU → MFPT | window | 1D-ResNet | 0.005 | 0.622 ± 0.163 | 0.652 ± 0.201 | 0.749 ± 0.154 |
| CWRU → MFPT | global | Feature + GBM | 0.039 | 0.811 ± 0.115 | 0.772 ± 0.171 | 0.785 ± 0.152 |
| CWRU → MFPT | global | 1D-CNN | 0.068 | 0.345 ± 0.016 | 0.388 ± 0.077 | 0.586 ± 0.058 |
| CWRU → MFPT | global | 1D-ResNet | 0.043 | 0.540 ± 0.146 | 0.544 ± 0.150 | 0.777 ± 0.111 |
| CWRU → PU | window | Feature + GBM | 0.330 | 1.000 | 0.987 ± 0.008 | 1.000 |
| CWRU → PU | window | 1D-CNN | 0.479 | 0.886 ± 0.089 | 0.866 ± 0.058 | 0.996 ± 0.003 |
| CWRU → PU | window | 1D-ResNet | 0.167 | 0.724 ± 0.038 | 0.609 ± 0.118 | 1.000 ± 0.000 |
| CWRU → PU | global | Feature + GBM | 0.332 | 1.000 | 1.000 | 1.000 |
| CWRU → PU | global | 1D-CNN | 0.457 | 0.790 ± 0.158 | 0.772 ± 0.152 | 0.998 ± 0.002 |
| CWRU → PU | global | 1D-ResNet | 0.466 | 0.977 ± 0.025 | 0.970 ± 0.020 | 1.000 |
| MFPT → CWRU | window | Feature + GBM | 0.149 | 0.641 ± 0.049 | 0.624 ± 0.081 | 0.706 ± 0.105 |
| MFPT → CWRU | window | 1D-CNN | 0.205 | 0.363 ± 0.132 | 0.394 ± 0.028 | 0.495 ± 0.090 |
| MFPT → CWRU | window | 1D-ResNet | 0.029 | 0.375 ± 0.102 | 0.450 ± 0.100 | 0.715 ± 0.113 |
| MFPT → CWRU | global | Feature + GBM | 0.193 | 0.605 ± 0.058 | 0.629 ± 0.098 | 0.763 ± 0.121 |
| MFPT → CWRU | global | 1D-CNN | 0.318 | 0.354 ± 0.068 | 0.379 ± 0.096 | 0.405 ± 0.070 |
| MFPT → CWRU | global | 1D-ResNet | 0.333 | 0.562 ± 0.044 | 0.553 ± 0.048 | 0.676 ± 0.123 |
| MFPT → PU | window | Feature + GBM | 0.579 | 0.956 ± 0.027 | 0.978 ± 0.011 | 0.997 ± 0.000 |
| MFPT → PU | window | 1D-CNN | 0.302 | 0.805 ± 0.104 | 0.818 ± 0.128 | 0.978 ± 0.015 |
| MFPT → PU | window | 1D-ResNet | 0.650 | 0.681 ± 0.013 | 0.753 ± 0.043 | 1.000 |
| MFPT → PU | global | Feature + GBM | 0.334 | 0.958 ± 0.027 | 0.976 ± 0.008 | 0.997 ± 0.000 |
| MFPT → PU | global | 1D-CNN | 0.304 | 0.551 ± 0.149 | 0.559 ± 0.152 | 0.883 ± 0.089 |
| MFPT → PU | global | 1D-ResNet | 0.333 | 0.550 ± 0.095 | 0.595 ± 0.075 | 0.835 ± 0.010 |
| PU → CWRU | window | Feature + GBM | 0.157 | 0.596 ± 0.131 | 0.688 ± 0.202 | 0.745 ± 0.194 |
| PU → CWRU | window | 1D-CNN | 0.412 | 0.612 ± 0.055 | 0.633 ± 0.083 | 0.810 ± 0.068 |
| PU → CWRU | window | 1D-ResNet | 0.391 | 0.620 ± 0.038 | 0.634 ± 0.080 | 0.774 ± 0.050 |
| PU → CWRU | global | Feature + GBM | 0.216 | 0.756 ± 0.068 | 0.724 ± 0.234 | 0.775 ± 0.215 |
| PU → CWRU | global | 1D-CNN | 0.612 | 0.639 ± 0.041 | 0.646 ± 0.031 | 0.717 ± 0.066 |
| PU → CWRU | global | 1D-ResNet | 0.448 | 0.572 ± 0.063 | 0.582 ± 0.054 | 0.645 ± 0.203 |
| PU → MFPT | window | Feature + GBM | 0.377 | 0.953 ± 0.054 | 0.988 ± 0.010 | 1.000 |
| PU → MFPT | window | 1D-CNN | 0.431 | 0.656 ± 0.137 | 0.646 ± 0.121 | 0.766 ± 0.135 |
| PU → MFPT | window | 1D-ResNet | 0.333 | 0.833 ± 0.092 | 0.877 ± 0.101 | 0.904 ± 0.092 |
| PU → MFPT | global | Feature + GBM | 0.546 | 0.964 ± 0.030 | 0.918 ± 0.068 | 1.000 |
| PU → MFPT | global | 1D-CNN | 0.333 | 0.561 ± 0.149 | 0.565 ± 0.143 | 0.667 ± 0.001 |
| PU → MFPT | global | 1D-ResNet | 0.357 | 0.444 ± 0.154 | 0.428 ± 0.134 | 0.749 ± 0.059 |

## Table 3  Feature-based baseline (XGBoost) cross-dataset accuracy

| Direction | Norm | k=0 (no adaptation) | k=20 (refit with target samples) |
|---|---|---|---|
| CWRU → MFPT | window | 0.067 | 0.753 ± 0.175 |
| CWRU → MFPT | global | 0.039 | 0.785 ± 0.152 |
| CWRU → PU | window | 0.330 | 1.000 |
| CWRU → PU | global | 0.332 | 1.000 |
| MFPT → CWRU | window | 0.149 | 0.706 ± 0.105 |
| MFPT → CWRU | global | 0.193 | 0.763 ± 0.121 |
| MFPT → PU | window | 0.579 | 0.997 ± 0.000 |
| MFPT → PU | global | 0.334 | 0.997 ± 0.000 |
| PU → CWRU | window | 0.157 | 0.745 ± 0.194 |
| PU → CWRU | global | 0.216 | 0.775 ± 0.215 |
| PU → MFPT | window | 0.377 | 1.000 |
| PU → MFPT | global | 0.546 | 1.000 |

## Table 4  Rig distance vs. transfer accuracy (no adaptation, global norm)

| Direction | Descriptor distance | Models combined | 1D-ResNet |
|---|---|---|---|
| CWRU → MFPT | 7.87 | 0.050 ± 0.013 | 0.043 |
| CWRU → PU | 5.42 | 0.418 ± 0.061 | 0.466 |
| MFPT → CWRU | 7.87 | 0.282 ± 0.063 | 0.333 |
| MFPT → PU | 6.87 | 0.324 ± 0.014 | 0.333 |
| PU → CWRU | 5.42 | 0.425 ± 0.163 | 0.448 |
| PU → MFPT | 6.87 | 0.412 ± 0.095 | 0.357 |

Spearman rho = -0.956 (p = 0.003); Pearson r = -0.763 (p = 0.077); n = 6 directions.

