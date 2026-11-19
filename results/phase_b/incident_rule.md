# Incident rule

Quiet minutes: 172 one-minute blocks (after a 20-window warm-up) from 43 ordinary episodes driven by r6.

| k | blocks with a run >= k |
|---|---|
| 1 | 72.7% |
| 2 | 58.7% |
| 3 | 32.0% |
| 4 | 25.0% |
| 5 | 18.6% |
| 6 | 16.9% |
| 7 | 13.4% |
| 8 | 12.8% |
| 9 | 10.5% |
| 10 | 10.5% |
| 11 | 9.9% |
| 12 | 7.6% |
| 13 | 5.8% |
| 14 | 5.8% |
| 15 | 5.2% |
| 16 | 5.2% |
| 17 | 5.2% |
| 18 | 5.2% |
| 19 | 4.1% |
| 20 | 4.1% |
| 21 | 3.5% |
| 22 | 2.9% |
| 23 | 2.9% |
| 24 | 2.9% |
| 25 | 2.9% |
| 26 | 2.9% |
| 27 | 2.3% |
| 28 | 2.3% |
| 29 | 1.7% |
| 30 | 0.6% |
| 31 | 0.6% |
| 32 | 0.6% |
| 33 | 0.6% |
| 34 | 0.6% |
| 35 | 0.6% |
| 36 | 0.6% |
| 37 | 0.0% |

Chosen: k = 19 (smallest k >= 2 with at most 5% of quiet blocks), interval = 76, majority 0.5, warm-up 20, chronic share 0.5.

## Incidents with this rule (300 history episodes)

| kind | episodes | building-up | sudden | chronic episodes |
|---|---|---|---|---|
| ambient | 24 | 1 | 2 | 2 |
| cpu_loss | 24 | 24 | 3 | 1 |
| gpu_loss | 25 | 9 | 12 | 4 |
| interference | 24 | 1 | 8 | 3 |
| leak | 26 | 29 | 4 | 6 |
| load_ramp | 26 | 8 | 6 | 4 |
| none | 151 | 0 | 81 | 17 |

Incidents: 188; building-up 72 (38% of incidents)
Chronic episodes: 37 of 300 (12%).

Raise MAGNITUDES once (fewer than 30 building-up incidents): no (72 building-up incidents).

If the magnitudes are raised, the whole history is regenerated into a new directory with the new magnitudes; episodes made with the old ones are never mixed with the new.
