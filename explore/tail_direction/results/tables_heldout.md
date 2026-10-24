### Table 1: per-arm results (5 seeds; median [min, max] over seeds unless stated)

| Target | Arm | Median grads | Tail-ESS/grad | Worst-coord tail-ESS | Max R-hat | Divergences (sum) | Tail-prob error, mean over coords and p | Mean MCSE |
|---|---|---|---|---|---|---|---|---|
| rot_t3_d10 | stock | 96,976 | 0.0169 [0.0129, 0.0191] | 1356 [1149, 1409] | 1.012 | 0 | 0.0020 [0.0013, 0.0063] | 0.0028 |
| rot_t3_d10 | stock_dense | 67,144 | 0.0293 [0.0236, 0.0336] | 1551 [1432, 1832] | 1.006 | 0 | 0.0012 [0.0010, 0.0019] | 0.0025 |
| rot_t3_d10 | coord | 90,399 | 0.0216 [0.0204, 0.0240] | 1679 [1535, 1767] | 1.004 | 0 | 0.0016 [0.0013, 0.0018] | 0.0022 |
| rot_t3_d10 | whiten | 61,130 | 0.0311 [0.0284, 0.0384] | 1508 [1223, 1737] | 1.008 | 0 | 0.0019 [0.0012, 0.0025] | 0.0029 |
| rot_t3_d10 | radial | 48,823 | 0.0689 [0.0651, 0.0707] | 2979 [2394, 3102] | 1.003 | 0 | 0.0014 [0.0010, 0.0022] | 0.0017 |
| rot_t3_d10 | composite (= radial) | 48,823 | 0.0689 [0.0651, 0.0707] | 2979 [2394, 3102] | 1.003 | 0 | 0.0014 [0.0010, 0.0022] | 0.0017 |
| rot_t3_d10 | selected (= radial, whiten) | 78,923 | 0.0322 [0.0182, 0.0651] | 1922 [1514, 2394] | 1.005 | 0 | 0.0014 [0.0012, 0.0019] | 0.0021 |
| two_blocks_d12 | stock | 304,011 | 0.0042 [0.0012, 0.0050] | 517 [56, 994] | 1.047 | 0 | 0.0038 [0.0017, 0.0082] | 0.0035 |
| two_blocks_d12 | stock_dense | 232,725 | 0.0060 [0.0047, 0.0063] | 473 [253, 942] | 1.028 | 0 | 0.0031 [0.0023, 0.0044] | 0.0029 |
| two_blocks_d12 | coord | 176,622 | 0.0091 [0.0079, 0.0096] | 744 [593, 1045] | 1.006 | 0 | 0.0020 [0.0019, 0.0027] | 0.0026 |
| two_blocks_d12 | whiten | 248,654 | 0.0063 [0.0036, 0.0077] | 559 [376, 944] | 1.019 | 0 | 0.0025 [0.0018, 0.0039] | 0.0030 |
| two_blocks_d12 | radial | 118,670 | 0.0121 [0.0090, 0.0186] | 552 [295, 897] | 1.009 | 77 | 0.0038 [0.0025, 0.0046] | 0.0033 |
| two_blocks_d12 | composite | 51,783 | 0.0561 [0.0460, 0.0579] | 2166 [1814, 2403] | 1.006 | 0 | 0.0015 [0.0012, 0.0019] | 0.0019 |
| two_blocks_d12 | selected (= composite) | 54,895 | 0.0535 [0.0460, 0.0579] | 2294 [1814, 2524] | 1.006 | 0 | 0.0014 [0.0011, 0.0019] | 0.0019 |
| rot_prod_cauchy_d10 | stock | 556,410 | 0.0003 [0.0002, 0.0005] | 127 [75, 160] | 1.048 | 0 | 0.0078 [0.0064, 0.0116] | 0.0053 |
| rot_prod_cauchy_d10 | stock_dense | 1,418,150 | 0.0002 [0.0000, 0.0003] | 102 [13, 270] | 1.099 | 0 | 0.0091 [0.0045, 0.0237] | 0.0032 |
| rot_prod_cauchy_d10 | coord | 377,946 | 0.0006 [0.0003, 0.0011] | 113 [27, 280] | 1.053 | 1 | 0.0100 [0.0096, 0.0100] | 0.0000 |
| rot_prod_cauchy_d10 | whiten | 595,909 | 0.0003 [0.0001, 0.0006] | 152 [39, 261] | 1.076 | 0 | 0.0070 [0.0068, 0.0169] | 0.0059 |
| rot_prod_cauchy_d10 | radial | 353,839 | 0.0004 [0.0000, 0.0008] | 74 [18, 159] | 1.084 | 1 | 0.0099 [0.0094, 0.0102] | 0.0005 |
| rot_prod_cauchy_d10 | composite | 149,660 | 0.0097 [0.0036, 0.0124] | 1199 [1017, 1699] | 1.007 | 0 | 0.0021 [0.0013, 0.0063] | 0.0023 |
| rot_prod_cauchy_d10 | selected (= radial) | 414,098 | 0.0004 [0.0000, 0.0019] | 105 [18, 352] | 1.084 | 2 | 0.0099 [0.0097, 0.0102] | 0.0001 |

### Table 2: paired per-seed tail-ESS/grad ratios, median [min, max] (seeds with ratio > 1)

| Target | Arm | vs stock | vs coord | Worst-coord tail-ESS/grad vs coord |
|---|---|---|---|---|
| rot_t3_d10 | stock_dense | 1.78× [1.34, 2.01] (5/5) | 1.34× [1.08, 1.56] (5/5) | 1.39× [1.12, 1.51] (5/5) |
| rot_t3_d10 | coord | 1.28× [1.14, 1.86] (5/5) | — | — |
| rot_t3_d10 | whiten | 1.88× [1.69, 2.26] (5/5) | 1.47× [1.18, 1.77] (5/5) | 1.36× [1.08, 1.65] (5/5) |
| rot_t3_d10 | radial | 3.93× [3.61, 5.36] (5/5) | 3.15× [2.88, 3.33] (5/5) | 3.25× [2.97, 3.49] (5/5) |
| rot_t3_d10 | composite | 3.93× [3.61, 5.36] (5/5) | 3.15× [2.88, 3.33] (5/5) | 3.25× [2.97, 3.49] (5/5) |
| rot_t3_d10 | selected | 1.69× [1.17, 3.93] (5/5) | 1.47× [0.76, 3.06] (3/5) | 1.36× [0.71, 2.97] (3/5) |
| two_blocks_d12 | stock_dense | 1.43× [1.07, 4.82] (5/5) | 0.63× [0.49, 0.76] (0/5) | 0.52× [0.36, 0.71] (0/5) |
| two_blocks_d12 | coord | 2.28× [1.81, 6.30] (5/5) | — | — |
| two_blocks_d12 | whiten | 1.61× [1.02, 2.87] (5/5) | 0.69× [0.46, 0.86] (0/5) | 0.57× [0.28, 0.87] (0/5) |
| two_blocks_d12 | radial | 3.70× [2.14, 10.54] (5/5) | 1.36× [0.94, 2.04] (4/5) | 1.13× [0.60, 2.02] (3/5) |
| two_blocks_d12 | composite | 13.28× [10.96, 44.97] (5/5) | 6.08× [4.82, 7.13] (5/5) | 10.85× [5.70, 11.94] (5/5) |
| two_blocks_d12 | selected | 13.28× [10.29, 43.51] (5/5) | 6.00× [4.82, 6.90] (5/5) | 10.29× [5.70, 11.94] (5/5) |
| rot_prod_cauchy_d10 | stock_dense | 0.70× [0.03, 0.99] (0/5) | 0.30× [0.02, 0.67] (0/5) | 0.35× [0.02, 1.82] (1/5) |
| rot_prod_cauchy_d10 | coord | 1.76× [0.89, 3.47] (4/5) | — | — |
| rot_prod_cauchy_d10 | whiten | 0.58× [0.37, 2.97] (2/5) | 0.37× [0.32, 0.95] (0/5) | 0.78× [0.30, 1.39] (1/5) |
| rot_prod_cauchy_d10 | radial | 1.14× [0.07, 4.15] (4/5) | 0.77× [0.04, 2.17] (2/5) | 1.36× [0.04, 6.18] (3/5) |
| rot_prod_cauchy_d10 | composite | 29.84× [7.71, 50.72] (5/5) | 15.24× [4.39, 38.99] (5/5) | 27.51× [7.80, 163.63] (5/5) |
| rot_prod_cauchy_d10 | selected | 1.47× [0.07, 5.73] (4/5) | 0.77× [0.04, 1.65] (2/5) | 1.49× [0.04, 4.60] (3/5) |

### Table 3: Phase 1 diagnostic per seed

| Target | Seed | Diagnostic label | Partition exact | Role accuracy | Best label (tail-ESS/grad) | Picked best? | Selected / best of {coord, radial, composite} |
|---|---|---|---|---|---|---|---|
| rot_t3_d10 | 0 | radial | yes | 1.00 | radial | yes | 1.00 |
| rot_t3_d10 | 1 | affine | no | 0.00 | radial | no | 0.47 |
| rot_t3_d10 | 2 | composite | no | 0.50 | radial | no | 0.26 |
| rot_t3_d10 | 3 | composite | no | 0.60 | radial | no | 0.28 |
| rot_t3_d10 | 4 | composite | no | 0.90 | radial | no | 0.56 |
| two_blocks_d12 | 0 | composite | no | 0.83 | composite | yes | 0.92 |
| two_blocks_d12 | 1 | composite | yes | 1.00 | composite | yes | 1.00 |
| two_blocks_d12 | 2 | composite | yes | 1.00 | composite | yes | 1.00 |
| two_blocks_d12 | 3 | composite | yes | 1.00 | composite | yes | 1.00 |
| two_blocks_d12 | 4 | composite | no | 0.92 | composite | yes | 0.97 |
| rot_prod_cauchy_d10 | 0 | radial | no | 0.00 | coord | no | 0.04 |
| rot_prod_cauchy_d10 | 1 | radial | no | 0.00 | coord | no | 0.77 |
| rot_prod_cauchy_d10 | 2 | composite | no | 0.10 | coord | no | 1.65 |
| rot_prod_cauchy_d10 | 3 | composite | no | 0.10 | radial | no | 0.76 |
| rot_prod_cauchy_d10 | 4 | composite | no | 0.00 | radial | no | 0.53 |

Aggregate over 15 (target, seed) runs: label = empirical best in 6/15; label = oracle label in 9/15; exact partition in 4/15; mean role accuracy 0.53; selected tail-ESS/grad ÷ best of the candidate maps: median 0.77, min 0.04, ≥ 0.9 in 7/15; the oracle label was the empirical best in 10/15.

Phase 1 draws reproduce the production calibration exactly in 60/60 Phase-1 arm records.
