# Issue 9: Hyperparameter-Experimente

## Kurzfassung

- Getestet wurden kombinierte Hyperparameter-Konfigurationen, nicht einzelne Parameter isoliert.
- Für den finalen Counting-Lauf würde ich `c05_long_default` nehmen.
- `c05_long_default` war in Stage C die beste Counting-Variante: mean_avg @20M = `-0.05259`, seed_std = `0.00052`.
- Baseline Default ist bei 20M trotzdem noch besser: mean_avg @20M = `-0.04906` bei nur ca. `280` Q-States statt ca. `108k` Q-States.
- Die Ergebnisse sprechen also für die Hyperparameter-Anpassung beim Counting Agent, aber noch nicht dafür, dass Counting bei 20M schon auskonvergiert oder besser als Baseline ist.

## Empfohlene Konfiguration

```text
config_id = c05_long_default
learning_rate = 0.01
discount_factor = 0.99
initial_epsilon = 1.0
epsilon_decay_fraction = 0.90
final_epsilon = 0.10
```

Der Unterschied zum bisherigen Counting Default ist damit vor allem: länger explorieren und `gamma` von `0.95` auf `0.99` setzen.

## Setup

- Artefakte und Rohdaten liegen lokal isoliert unter `models/issue9_experiments/`. Nicht gepushed, aber verfügbar falls Rückfragen kommen sollten. 
- Evaluation lief greedy, also ohne Exploration.
- Innerhalb eines Checkpoints wurden dieselben Eval-Seeds genutzt.
- Stage A: 16 Konfigurationen, 1 Seed, 3M Training, 150k Evaluationen pro Checkpoint.
- Stage B: 8 Konfigurationen, 2 Seeds, 10M Training, 500k Evaluationen pro Checkpoint.
- Stage C: 4 Counting-Konfigurationen plus Baseline Default, 3 Seeds, 20M Training, 1M Evaluationen pro Checkpoint.
- `mean_ci99` in Stage C ist der mittlere 99%-CI-Rand der einzelnen 1M-Evaluationen; `seed_std` ist die Streuung zwischen Trainingsseeds.

## Stage A

| Config | lr | exploration | gamma | avg_reward @3M | 99%-CI | loss | q_states | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c05_long_default | 0.01 | long_default | 0.99 | -0.06424 | +-0.00634 | 0.4890 | 91,751 | keep |
| c10_lr02_gamma099 | 0.02 | long_low | 0.99 | -0.06527 | +-0.00634 | 0.4893 | 91,483 | keep |
| c06_low_noise | 0.01 | long_low | 0.99 | -0.06838 | +-0.00634 | 0.4910 | 91,641 | keep |
| c07_high_noise | 0.005 | long_high | 0.98 | -0.06845 | +-0.00634 | 0.4913 | 91,514 | keep |
| c13_lr02_gamma1 | 0.02 | moderate_low | 1.0 | -0.06927 | +-0.00633 | 0.4906 | 91,586 | keep |
| c15_long_high_lr | 0.02 | long_default | 0.99 | -0.06945 | +-0.00634 | 0.4910 | 91,571 | keep |
| c09_lr01_gamma098 | 0.01 | long_default | 0.98 | -0.07189 | +-0.00634 | 0.4932 | 91,807 | keep |
| c03_fast | 0.02 | moderate_low | 0.98 | -0.07221 | +-0.00633 | 0.4914 | 91,931 | drop |
| c02_very_stable | 0.0025 | long_low | 0.99 | -0.07245 | +-0.00635 | 0.4954 | 91,843 | drop |
| c12_lr01_gamma1 | 0.01 | long_low | 1.0 | -0.07270 | +-0.00634 | 0.4928 | 91,663 | drop |
| c11_lr005_gamma1 | 0.005 | long_low | 1.0 | -0.07513 | +-0.00634 | 0.4953 | 91,611 | drop |
| c04_aggressive | 0.04 | fast_greedy | 0.95 | -0.07526 | +-0.00632 | 0.4924 | 91,718 | drop |
| c00_default | 0.01 | default_like | 0.95 | -0.07555 | +-0.00634 | 0.4951 | 91,423 | keep |
| c01_conservative | 0.005 | long_low | 0.99 | -0.07585 | +-0.00635 | 0.4962 | 91,818 | drop |
| c08_lr005_gamma098 | 0.005 | long_default | 0.98 | -0.07659 | +-0.00635 | 0.4967 | 91,757 | drop |
| c14_fast_low_eps | 0.01 | fast_greedy | 0.99 | -0.07957 | +-0.00634 | 0.4974 | 91,530 | drop |

## Stage B

| Config | lr | exploration | gamma | mean_avg @10M | mean_loss | seed_std | mean_q_states | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c05_long_default | 0.01 | long_default | 0.99 | -0.05622 | 0.4827 | 0.00261 | 104,229 | keep |
| c15_long_high_lr | 0.02 | long_default | 0.99 | -0.05673 | 0.4826 | 0.00138 | 104,244 | keep |
| c09_lr01_gamma098 | 0.01 | long_default | 0.98 | -0.05720 | 0.4834 | 0.00105 | 104,200 | keep |
| c07_high_noise | 0.005 | long_high | 0.98 | -0.05806 | 0.4842 | 0.00081 | 104,178 | drop |
| c13_lr02_gamma1 | 0.02 | moderate_low | 1.0 | -0.05833 | 0.4832 | 0.00143 | 104,237 | drop |
| c10_lr02_gamma099 | 0.02 | long_low | 0.99 | -0.05863 | 0.4836 | 0.00153 | 104,178 | drop |
| c06_low_noise | 0.01 | long_low | 0.99 | -0.05906 | 0.4840 | 0.00050 | 104,182 | drop |
| c00_default | 0.01 | default_like | 0.95 | -0.06067 | 0.4850 | 0.00118 | 104,265 | keep |

## Stage C

| Agent | Config | mean_avg @20M | mean_ci99 | mean_win | mean_loss | mean_push | seed_std | mean_q_states |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | c00_default | -0.04906 | +-0.00245 | 0.4290 | 0.4780 | 0.0930 | 0.00138 | 280 |
| counting | c05_long_default | -0.05259 | +-0.00245 | 0.4275 | 0.4801 | 0.0923 | 0.00052 | 108,681 |
| counting | c09_lr01_gamma098 | -0.05353 | +-0.00245 | 0.4272 | 0.4808 | 0.0920 | 0.00101 | 108,597 |
| counting | c00_default | -0.05442 | +-0.00245 | 0.4267 | 0.4811 | 0.0922 | 0.00138 | 108,590 |
| counting | c15_long_high_lr | -0.05528 | +-0.00245 | 0.4262 | 0.4814 | 0.0924 | 0.00106 | 108,479 |

## Stage-C-Trend Counting

| Config | mean_avg @10M | mean_avg @15M | mean_avg @20M |
| --- | --- | --- | --- |
| c00_default | -0.05625 | -0.05440 | -0.05442 |
| c05_long_default | -0.05533 | -0.05447 | -0.05259 |
| c09_lr01_gamma098 | -0.05619 | -0.05358 | -0.05353 |
| c15_long_high_lr | -0.05721 | -0.05694 | -0.05528 |

## Einschätzung

- `c05_long_default` ist die sinnvollste Wahl für den finalen Counting-Lauf, weil es bei 20M vorne liegt und über die Seeds am stabilsten war.
- `c09_lr01_gamma098` ist nah dran, aber bei 20M etwas schlechter und weniger stabil.
- `c15_long_high_lr` wirkt wegen Seed 123 weniger robust.
- Baseline ist in diesem Lauf weiterhin vorne; das passt zur großen State-Space-Differenz und heißt nicht automatisch, dass Counting generell schlechter ist. Anzahl der Trainingsepisoden muss jedoch deutlich nach oben geschraubt werden.
- Die Ergebnisse sollten als Empfehlung für den nächsten großen Lauf gelesen werden, nicht als endgültiger Beweis für optimale Hyperparameter, da nicht jede möglicherweise sinnvolle Kombination getestet wurde.
