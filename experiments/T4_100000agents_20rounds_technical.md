# SybilCore Experiment Report: T4_100000agents_20rounds

## Configuration

| Parameter | Value |
|-----------|-------|
| Agents | 100,000 |
| Rounds | 20 |
| Rogues Injected | 500 |
| Rogue Type | random mix |
| Injection Round | 3 |
| Compound Spread | 2% per rogue/round |
| Enforcement | none |
| Seed | 42 |

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Runtime | 344.31s |
| Detection Latency | N/A rounds |
| False Positives | 0 |
| Peak Rogue Count | 729 |
| Final Rogue Count | 729 |
| Compound Infections | 214 |
| Peak Corruption Rate | 0.7% |

## Round-by-Round Summary

| Round | Mean Coeff | Max Coeff | Rogues | Infections | Time (ms) |
|-------|-----------|-----------|--------|------------|-----------|
| 1 | 3.9 | 16.7 | 0 | 0 | 4817 |
| 2 | 8.9 | 39.3 | 0 | 0 | 5606 |
| 3 | 23.3 | 89.5 | 515 | 0 | 8354 |
| 4 | 32.2 | 116.9 | 527 | 12 | 8706 |
| 5 | 39.1 | 138.5 | 537 | 10 | 10559 |
| 6 | 40.7 | 141.8 | 553 | 16 | 10963 |
| 7 | 44.5 | 155.5 | 568 | 15 | 14497 |
| 8 | 44.5 | 162.3 | 583 | 15 | 11856 |
| 9 | 47.8 | 159.9 | 594 | 11 | 21149 |
| 10 | 49.5 | 160.2 | 604 | 10 | 13956 |
| 11 | 49.2 | 157.4 | 617 | 13 | 21167 |
| 12 | 51.5 | 162.2 | 627 | 10 | 16279 |
| 13 | 50.9 | 159.4 | 639 | 12 | 17271 |
| 14 | 53.0 | 165.0 | 654 | 15 | 27685 |
| 15 | 54.8 | 168.2 | 665 | 11 | 18573 |
| 16 | 55.0 | 169.5 | 681 | 16 | 19648 |
| 17 | 57.0 | 164.9 | 691 | 10 | 30238 |
| 18 | 57.5 | 166.6 | 703 | 12 | 21549 |
| 19 | 59.4 | 170.9 | 716 | 13 | 22998 |
| 20 | 60.9 | 171.7 | 729 | 13 | 37562 |

## Tier Distribution (Final Round)

- **clear**: 99506 (99.5%) #################################################
- **clouded**: 490 (0.5%) 
- **flagged**: 0 (0.0%) 
- **lethal_eliminator**: 0 (0.0%) 
