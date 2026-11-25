# Part 1 results: beat grids and sync

Pack: `testpack\stand-in`. Machine: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel (14 threads), device `cpu`.

| Clip | Grid | Tempo (ref) | Sig | Beat F (±70 ms) | Downbeat F | AMLt | Max drift ms | Beats >10 ms | Analysis s | Conf |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | constant | 124.00 (124.0) | 4/4 | 1.0 | 1.0 | 1.0 | 0.88 | 0 | 11.87 | 0.92 |
| A2 | constant | 140.00 (140.0) | 4/4 | 0.9948 | 1.0 | 0.9891 | 0.46 | 0 | 4.53 | 0.937 |
| A3 | variable | 93.02 (92.973) | 4/4 | 1.0 | - | 1.0 | 20.75 | 8 | 5.22 | 0.968 |
| A4 | constant | 120.00 (120.0) | 4/4 | 0.9948 | 1.0 | 0.9896 | 0.56 | 0 | 4.36 | 0.929 |
| A5 | piecewise | 128.00 (128.0) | 4/4 | 0.9983 | 0.9931 | 0.9964 | 1.08 | 0 | 4.53 | 0.95 |
| A6 | constant | 68.00 (68.0) | 6/8 | 0.9961 | 0.9922 | 0.9919 | 1.61 | 0 | 4.1 | 1.0 |
| A7 | constant | 128.00 (128.0) | 4/4 | 1.0 | 1.0 | 1.0 | 1.59 | 0 | 7.36 | 0.948 |

## Sync A7 -> A1 (overlap from A1 bar 33, A7 bar 1, 60 s)

* Time-stretch ratio (B output/input duration): **1.032253** (127.999 -> 124.0 BPM), mapping: anchored every 16 beats (constant ratio).
* Offset: A1 bar 33 at 62.1148 s, A7 bar 1 at 0.5184 s.
* Kick onsets, grid only: median |offset| 0.81 ms, max 1.91 ms (124 matched kicks).
* Automatic onset nudge: off (p95 |offset| 1.37 ms).
* B downbeats vs A downbeats in the render: max 0.0 ms.
* Reference check (B's reference downbeats mapped through the stretch vs A's reference downbeats): max 0.96 ms, median 0.82 ms.

## Variable-tempo partners synced onto A1 bar 33 (bar-by-bar stretch)

| B | B grid | Ratio | Kick onsets median / max | Reference check median / max | Notes |
|---|---|---|---|---|---|
| A5 | piecewise | 0.967741 | 1.54 / 2.94 ms | 0.33 / 0.34 ms |  |
| A3 | variable | 0.742828 | 4.32 / 33.3 ms | 1.47 / 5.64 ms | Large tempo change for B (35%: 93.0 -> 124.0 BPM). Stretches beyond ~8% start to sound unnatural; a DJ would usually pick a closer track or mix at half/double time. |

A3 is a live take whose kicks scatter 7-12 ms around the beat by design, so its kick-onset numbers measure the drummer as much as the sync; the reference check (intended beat times) measures the mapping.
