# Big Training Run `20260608_170145`

## Konfiguration und Lauf

- Trainingsseeds: `1`, `42`, `123`
- Agenten: Baseline und Counting, jeweils `100,000,000` Episoden pro Seed
- Checkpoints: alle `10,000,000` Episoden
- Eval-Seeds: `1234`, `4321`, `9876`
- Greedy-Evaluation: `1,000,000` Episoden je Eval-Seed, nominal `3,000,000` je trainiertem Agent/Seed
- Training: `13:01:47`
- Greedy-Evaluation: `0:12:06`
- True-Count-Bucket-Evaluation: `0:12:12`
- Notebook-Lauf abgeschlossen: `2026-06-09 06:28:19 +02:00`

Alle `60` Checkpoints und alle sechs finalen Modelle wurden erfolgreich geschrieben und validiert. Baseline hat je finalem Modell `280` Q-States; Counting hat je nach Seed `113,744` bis `113,769` Q-States.

## Evaluationsergebnis

Aggregation über drei Trainingsseeds und drei Eval-Seeds:

| Agent | Episoden | Win Rate | Loss Rate | Push Rate | Average Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 9,000,000 | 0.427461 | 0.479026 | 0.093513 | -0.051565 |
| Counting | 9,000,000 | 0.428345 | 0.478081 | 0.093574 | -0.049736 |
| Counting minus Baseline | - | +0.000884 | -0.000945 | +0.000061 | **+0.001829** |

Average Reward je Trainingsseed, gemittelt über die drei Eval-Seeds:

| Train-Seed | Baseline | Counting | Counting minus Baseline |
| ---: | ---: | ---: | ---: |
| 1 | -0.051894 | -0.049303 | +0.002591 |
| 42 | -0.051878 | -0.049213 | +0.002665 |
| 123 | -0.050924 | -0.050691 | +0.000233 |

Counting liegt insgesamt und für jeden Trainingsseed über dem jeweiligen Baseline-Niveau. Der Vorteil ist bei Seed `123` jedoch klein, was auf relevante Trainingsseed-Varianz hinweist.

## True-Count-Kontext

Gewichteter Average-Reward-Unterschied Counting minus Baseline:

| True-Count-Bucket | Unterschied |
| --- | ---: |
| `<= -3` | -0.003590 |
| `-2 to 0` | +0.002074 |
| `1 to 2` | +0.002918 |
| `>= 3` | +0.003473 |

Counting ist besonders bei positivem True Count besser als Baseline, verliert aber im stark negativen Bucket `<= -3`.

## Artefakte

- Checkpoints: `models/checkpoints/<agent-seed>/` (erstmal nicht gepushed wegen Speicherbedarf)
- Finale Modelle: `models/20260608_170145_*_agent.pkl`
- Evaluationen und Bericht: `models/evaluations/20260608_170145/`
- Ausgefuehrtes Notebook: `models/runs/big_run_launcher_20260608_170138/`
- Plots: `images/`