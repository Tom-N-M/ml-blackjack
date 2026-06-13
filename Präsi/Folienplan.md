# Folienplan: Evaluation von Q-Learning-Agenten in Blackjack

## Gestaltungsregeln

- **Eine Folie = eine Aussage.** Keine zweite Argumentation in einer Fußnote oder Nebenbox verstecken.
- **Titel bleiben einzeilig** und bestehen möglichst aus höchstens fünf Wörtern.
- **Auf die Folie kommen nur Belege:** Definitionen, Kennzahlen, Formeln und Diagramme.
- **Erklärungen gehören in den Vortrag.** Richtwert: höchstens 40 bis 50 sichtbare Wörter je Inhaltsfolie.
- **Keine pauschalen Aussagen:** Ein positiver True Count bedeutet in der reduzierten Umgebung nicht automatisch einen positiven Spielerwartungswert.
- **Farblogik:** Baseline durchgehend dunkelblau, Counting türkis, Einschränkungen orange, negative Ergebnisse rot.

## Aufbau der Agenda

Die Agenda zeigt nur die drei Hauptteile. Unter jedem Teil stehen drei kurze Orientierungspunkte, die den inhaltlichen Bogen zusammenfassen. Diese Punkte sind **keine festen Unterkapitel** und müssen nicht als eigene Folientitel wiederkehren.

| Hauptteil | Stichpunkte auf der Agenda |
|---|---|
| **Motivation & Theorie** | Blackjack als Entscheidungsproblem<br>Kartenzählen als Zusatzinformation<br>Forschungsrahmen und Grenzen |
| **Methodik & Technik** | Vergleich der Zustandsräume<br>Q-Learning und Hyperparameter<br>Training und Evaluation |
| **Ergebnisse & Fazit** | Lernverlauf und Performance<br>Analyse des Agentenverhaltens<br>Fazit und nächste Schritte |

Damit bleibt die Agenda kompakt. Innerhalb eines Hauptteils dürfen beliebig viele Folien folgen, solange jede Folie eine klar erkennbare Funktion für die Argumentation erfüllt.

## Einstieg

| Nr. | Folientitel | Muss auf die Folie | Hintergrund und Sprechpunkte | Konkrete Darstellung |
|---:|---|---|---|---|
| 1 | **Q-Learning in Blackjack** | Titel: „Evaluation von Q-Learning-Agenten in Blackjack“<br>Untertitel: „Nutzen von Kartenzählinformationen im tabellarischen Q-Learning“<br>Modul, Hochschule, Team, Lehrkraft und Datum | Verglichen werden zwei Q-Learning-Agenten, die sich ausschließlich durch ihre Zustandsinformationen unterscheiden. | Ruhige Titelfolie ohne Ergebnisse oder Forschungsfragen. |
| 2 | **Agenda** | Die drei Hauptteile und jeweils drei kurze Orientierungspunkte aus der Tabelle oben | Den Spannungsbogen erklären: Problem verstehen, Vergleich aufbauen, Nutzen des Countings bewerten. | Drei Spalten. Keine Nummerierung wie 1.1 oder 2.3. |

## Motivation & Theorie

**Ziel des Kapitels:** Das Entscheidungsproblem erklären und begründen, warum Kartenzählinformationen überhaupt relevant sein könnten.

| Nr. | Folientitel | Muss auf die Folie | Hintergrund und Sprechpunkte | Konkrete Darstellung |
|---:|---|---|---|---|
| 3 | **Blackjack als MDP** | Zustand: `sₜ = (Spielersumme, Dealerkarte, nutzbares Ass)`<br>Aktionen: `Stand` oder `Hit`<br>Zyklus: `sₜ → aₜ → Umgebung → rₜ₊₁, sₜ₊₁` | Ein MDP benötigt Zustände, Aktionen, Übergänge und Rewards. Der Agent kennt keine zukünftigen Karten und lernt ausschließlich aus simulierten Runden. | Oben zwei Boxen für Zustand und Aktion, unten der bereits erstellte Entscheidungszyklus. |
| 4 | **Reward und Episode** | Gewinn `+1`<br>Push `0`<br>Verlust `−1`<br>Zwischenschritt `0`<br>`Ø Reward = Win Rate − Loss Rate` | Eine Episode entspricht genau einer Runde. Der finale Ausgang wird über das Q-Update auf vorherige Entscheidungen zurückgeführt. Der Reward bildet kein Einsatzmanagement ab. | Drei große Reward-Karten und darunter eine kurze Episodenlinie. |
| 5 | **Das Deck hat Gedächtnis** | Sechs Decks = 312 Karten<br>Reshuffle bei höchstens 78 Karten<br>Gleiche sichtbare Hände können unterschiedliche Restdecks besitzen | Der Schlitten bleibt über mehrere Episoden bestehen. Deshalb hängt die nächste Kartenwahrscheinlichkeit von bereits gespielten Karten ab. Für die Baseline ist dieser Unterschied unsichtbar. | Zwei gleiche Spielsituationen mit unterschiedlich zusammengesetzten Restdecks vergleichen. |
| 6 | **Running und True Count** | `2–6: +1`<br>`7–9: 0`<br>`10–Ass: −1`<br>`True Count = Running Count / verbleibende Decks` | Der Running Count verdichtet die sichtbare Kartenhistorie. Der True Count setzt diese Summe ins Verhältnis zum verbleibenden Schlitten. Beide Werte sind Näherungen und kein vollständiger Zustand. | Drei Kartengruppen und ein einzelnes Rechenbeispiel. |
| 7 | **Fragen und Grenzen** | **Nutzung:** Verändert Counting die Aktionswahl?<br>**Performance:** Verbessert es Reward und Win Rate?<br>**Konvergenz:** Reichen 100 Mio. Episoden?<br>Kein Split, Double Down, Surrender, Insurance, 3:2-Payout oder variable Einsätze | Die Umgebung isoliert Hit-or-Stand-Entscheidungen. Sie ist für den Modellvergleich geeignet, aber keine vollständige Casino-Simulation. | Links drei Forschungsfragen, rechts ein kompakter Scope-Block. |

## Methodik & Technik

**Ziel des Kapitels:** Zeigen, wie sich die Agenten unterscheiden und wie ein möglichst fairer Vergleich erzeugt wurde.

| Nr. | Folientitel | Muss auf die Folie | Hintergrund und Sprechpunkte | Konkrete Darstellung |
|---:|---|---|---|---|
| 8 | **Baseline: 280 Zustände** | `sᴮ = (Spielersumme, Dealerkarte, nutzbares Ass)`<br>`Q(s) = [Q(Stand), Q(Hit)]`<br>Final beobachtet: **280 Zustände** | Die kleine Tabelle wird schnell vollständig besucht. Zustände wiederholen sich häufig und ihre Q-Werte werden entsprechend oft aktualisiert. | State Encoder, eine beispielhafte Q-Tabellenzeile und die Zahl 280. |
| 9 | **Counting: 113.760 Zustände** | `sᶜ = (Baseline, Running Count, True Count, Deck-Bucket)`<br>Running Count: `−20…20`<br>True Count: `−10…10`<br>Deck-Bucket: `0…6`<br>**406×** so viele Zustände wie die Baseline | Mehr Kontext erzeugt viele seltene Kombinationen. Running Count, True Count und Deck-Bucket sind zudem teilweise redundant. | Größenvergleich 280 zu 113.760; darunter die beiden Encoder. |
| 10 | **Das Q-Learning-Update** | `Q(sₜ,aₜ) ← Q(sₜ,aₜ) + α[rₜ₊₁ + γ max Q(sₜ₊₁,a′) − Q(sₜ,aₜ)]`<br>`α = 0,01`, `γ = 0,99`<br>Training: ε-greedy<br>Evaluation: `ε = 0` | Nach jedem Schritt wird genau ein Tabellenwert angepasst. Der Agent benötigt kein Modell der Kartenübergänge und keine vorab gelabelten Daten. | Formel zentral; nur die vier relevanten Bestandteile farblich erklären. |
| 11 | **Preis der Transparenz** | Vorteil: Q-Werte und Policy direkt prüfbar<br>Nachteil: ähnliche Zustände teilen kein Wissen<br>Neue Zustände starten mit `[0,0]`<br>Bei Gleichstand wählt `argmax` technisch `Stand` | Die `defaultdict` ist eine White-Box, generalisiert aber nicht. Fast gleiche Hand- oder Count-Zustände müssen separat gelernt werden. | Zweispaltiger Vergleich „Nachvollziehbar“ gegen „Keine Generalisierung“. |
| 12 | **Hyperparameter-Auswahl** | Stage A: 16 Konfigurationen, 3 Mio.<br>Stage B: 8 Konfigurationen, 10 Mio.<br>Stage C: 4 Counting + Baseline, 20 Mio.<br>Gewählt: `c05_long_default`<br>`ε: 1,0 → 0,1` | Die Suche verglich Konfigurationspakete, keine vollständige Ablation. Die gewählte Variante war bei 20 Mio. die beste Counting-Konfiguration, lag aber noch hinter der Baseline. | Dreistufiger Funnel plus kleine Box mit der finalen Konfiguration. |
| 13 | **600 Millionen Trainingsrunden** | `2 Agententypen × 3 Seeds × 100 Mio.`<br>Seeds: `1, 42, 123`<br>60 Checkpoints<br>6 finale Modelle<br>Laufzeit: `13 h 01 min 47 s` | Beide Agententypen verwenden dieselben Hyperparameter und dasselbe Trainingsbudget. Primär verglichen wird damit die Zustandsrepräsentation. | Die Multiplikation groß, technische Kennzahlen klein darunter. |
| 14 | **18 Millionen Testepisoden** | Greedy Evaluation mit `ε = 0`<br>Drei Evaluationsseeds je Modell<br>Je Seed 1 Mio. Episoden<br>`6 Modelle × 3 Mio. = 18 Mio.` | Viele Testepisoden schätzen eine feste Policy präzise, ersetzen aber keine unabhängigen Trainingsläufe. Für den Modellvergleich bleiben drei Trainingsseeds entscheidend. | Matrix aus sechs Modellen und je drei Evaluationsseeds. |

## Ergebnisse & Fazit

**Ziel des Kapitels:** Erst Lernverlauf und Endergebnis zeigen, danach prüfen, ob die Zusatzinformation das Verhalten tatsächlich verändert.

| Nr. | Folientitel | Muss auf die Folie | Hintergrund und Sprechpunkte | Konkrete Darstellung |
|---:|---|---|---|---|
| 15 | **Reward im Training** | Oberer Teil von `Doku/figures/checkpoint_progress.png`<br>Baseline früh etwa `−0,05`<br>Counting bei 10 Mio. ca. `−0,0561`<br>Bester Counting-Checkpoint bei 90 Mio. ca. `−0,0476` | Counting benötigt deutlich länger, erreicht am Ende aber ein ähnliches Niveau. Der Verlauf ist nicht monoton und enthält Evaluationsrauschen. | Diagramm groß; nur 10, 90 und 100 Mio. markieren. |
| 16 | **Wachstum der Q-Tabelle** | Unterer Teil von `checkpoint_progress.png`<br>Baseline konstant bei 280<br>Counting: 104.104 → 113.760<br>Tabelle wächst nach 100 Mio. weiter | Stabiler Reward bedeutet nicht vollständige Konvergenz. Counting entdeckt weiterhin Zustände; Lernrate und Trainings-ε bleiben ebenfalls positiv. | Zustandskurve und ein Callout: „Performance stabilisiert sich vor dem Zustandsraum.“ |
| 17 | **Finale Performance** | Baseline Reward: `−0,051562`<br>Counting Reward: `−0,049767`<br>Differenz: `+0,001794`<br>Win Rate: `42,746 % → 42,833 %` | Counting ist aggregiert leicht besser, beide Agenten bleiben aber im negativen Reward-Bereich. Der absolute Effekt ist klein. | Kleine Vergleichstabelle und zwei große Differenzwerte. |
| 18 | **Kein Signifikanznachweis** | Drei Trainingsseed-Paare<br>Alle drei Differenzen positiv<br>`p = 0,153`<br>95-%-Intervall: `[−0,00163; +0,00522]` | Der beobachtete Effekt ist positiv, mit drei unabhängigen Trainingsläufen aber nicht robust belegt. Das Intervall enthält null. | Seed-Paare oder `final_metrics_by_seed.png`; p-Wert deutlich daneben. |
| 19 | **Verhalten nach True Count** | Buckets: `≤−3`, `−2…0`, `1…2`, `≥3`<br>Hit-Rate bei `≤−3`: `46,244 % → 48,623 %`<br>Hit-Rate bei `≥3`: `49,363 % → 44,587 %`<br>Reward-Differenz bei `≥3`: `+0,003474` | Die Zusatzinformation wird genutzt: Bei negativem Count zieht der Agent häufiger, bei positivem Count seltener. Nur im stärksten negativen Bucket verschlechtert sich sein Reward. | `true_count_effects.png`; die zwei größten Hit-Rate-Unterschiede markieren. |
| 20 | **Policy bei TC −3/+3** | `tc_-3/policy_hard.png` und `tc_3/policy_hard.png` desselben Modells<br>Höchstens drei konkrete Aktionswechsel markieren | Die Heatmaps zeigen die Verhaltensänderung auf Handebene. Sie sind über Running Count und Deck-Bucket gemittelt und daher keine einzelnen vollständigen Zustände. | Zwei Hard-Total-Heatmaps. Keine Gegenüberstellung von Hard und Soft beim gleichen Count. |
| 21 | **Antworten auf Leitfragen** | **Nutzung:** Ja, die Policy ändert sich.<br>**Performance:** Leicht besser, aber `p = 0,153`.<br>**Konvergenz:** 100 Mio. reichen zum Aufschließen, nicht als Nachweis. | Gesamturteil: Die Zusatzinformation ist handlungsrelevant, ihr Nutzen ist mit tabellarischem Q-Learning aber teuer erkauft. 406-mal mehr Zustände führen nur zu einer kleinen Verbesserung. | Drei Antwortkarten mit jeweils einer Antwort und einem Beleg. |
| 22 | **Nächste Schritte** | Ablation der Count-Merkmale<br>Mindestens zehn Trainingsseeds<br>Vollständige Regeln und variable Einsätze<br>DQN als Funktionsapproximation | Zuerst muss geklärt werden, welche Zusatzmerkmale nötig sind. Danach sind realistischere Regeln und ein Modell sinnvoll, das Wissen zwischen ähnlichen Zuständen überträgt. Nur mehr Episoden lösen das Repräsentationsproblem nicht. | Priorisierte Roadmap mit vier Schritten. |

## Empfohlene Kürzung für 14 Minuten

Die 22 Folien bilden die vollständige Version. Für einen Vortrag von etwa 14 Minuten sollten vier Folien übersprungen oder zusammengelegt werden:

| Maßnahme | Ergebnis |
|---|---|
| Folie 4 in Folie 3 integrieren | MDP-Zyklus enthält den Reward direkt. |
| Folien 10 und 11 zusammenlegen | Q-Learning-Formel links, Trade-off rechts. |
| Folie 14 als Backup verwenden | Evaluationsdesign auf Folie 13 in zwei Sätzen nennen. |
| Folie 20 als Backup verwenden | Policy-Maps nur bei Rückfragen oder zusätzlicher Zeit zeigen. |

Damit verbleiben **18 Hauptfolien**. Die Ergebnisfolien 15 bis 19 sowie das Fazit sollten nicht gekürzt werden.

## Backupfolien

| Nr. | Kurztitel | Inhalt |
|---:|---|---|
| B1 | **Tuning-Ergebnisse** | Vollständige Ergebnisse der Stages A bis C. |
| B2 | **Finale Hyperparameter** | Alle Werte inklusive ε-Decay und Checkpoint-Intervall. |
| B3 | **Bucket-Tabelle** | Reward-, Win-, Loss- und Aktionsraten je True-Count-Bucket. |
| B4 | **Weitere Policy-Maps** | Hard und Soft Totals für True Count `−3`, `0` und `+3`. |
| B5 | **Ergebnisse je Seed** | Finale Kennzahlen für jeden Trainings- und Evaluationsseed. |
| B6 | **Limitationen** | Reduzierte Regeln, redundante Features, Seed- und Reproduzierbarkeitsgrenzen. |
