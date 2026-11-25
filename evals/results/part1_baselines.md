# Part 1: baselines, ablation and stress tests

Beat F at ±70 ms (mir_eval). "Beat This! raw" is the model's own minimal post-processing; "full pipeline" adds the tempo model and onset alignment (beatagent/tracking.py). Max drift is reported for constant-tempo clips.

| Clip | librosa beat F | Beat This! raw beat F / downbeat F / max drift | Full pipeline beat F / downbeat F / max drift |
|---|---|---|---|
| A1 | 0.2311 | 0.9986 / 0.8295 / 54.19 ms | 1.0 / 1.0 / 0.88 ms |
| A2 | 0.6667 | 0.9632 / 0.9796 / 62.86 ms | 0.9948 / 1.0 / 0.46 ms |
| A3 | 0.9922 | 0.9962 / - / - ms | 1.0 / - / - ms |
| A4 | 1.0 | 0.8571 / 0.8067 / 50.0 ms | 0.9948 / 1.0 / 0.56 ms |
| A5 | 0.8368 | 0.9809 / 0.7746 / - ms | 0.9983 / 0.9931 / - ms |
| A6 | 0.4013 | 0.4971 / 0.9846 / 20.59 ms | 0.9961 / 0.9922 / 1.61 ms |
| A7 | 0.9933 | 0.9978 / 0.9098 / 38.75 ms | 1.0 / 1.0 / 1.59 ms |

## Stress test on A1 (degraded copies of the 6-minute house clip)

| Degradation | Grid | Tempo | Beat F | Downbeat F | Max drift |
|---|---|---|---|---|---|
| clean | constant | 124.0 | 1.0 | 1.0 | 0.88 ms |
| mp3_96k | constant | 124.0 | 1.0 | 1.0 | 0.9 ms |
| noise_15db | constant | 124.0 | 0.9993 | 1.0 | 0.86 ms |
| lowpass_3k | constant | 124.0 | 1.0 | 1.0 | 0.99 ms |
| no_kick_hp200 | constant | 124.0 | 0.998 | 0.9946 | 1.34 ms |

Speed, A1 (6 min) on CPU: {4: 12.46, 14: 12.06} seconds by number of torch threads.
