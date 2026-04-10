# SybilCore Experiment Report: T5_1000000agents_10rounds

## Configuration

| Parameter | Value |
|-----------|-------|
| Agents | 1,000,000 |
| Rounds | 10 |
| Rogues Injected | 5000 |
| Rogue Type | random mix |
| Injection Round | 3 |
| Compound Spread | 1% per rogue/round |
| Enforcement | none |
| Seed | 42 |

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Runtime | 986.24s |
| Detection Latency | N/A rounds |
| False Positives | 0 |
| Peak Rogue Count | 5426 |
| Final Rogue Count | 5426 |
| Compound Infections | 381 |
| Peak Corruption Rate | 0.5% |

## Round-by-Round Summary

| Round | Mean Coeff | Max Coeff | Rogues | Infections | Time (ms) |
|-------|-----------|-----------|--------|------------|-----------|
| 1 | 3.0 | 8.8 | 0 | 0 | 39871 |
| 2 | 4.0 | 16.7 | 0 | 0 | 49865 |
| 3 | 6.5 | 40.9 | 5045 | 0 | 82261 |
| 4 | 9.0 | 66.7 | 5115 | 70 | 73497 |
| 5 | 11.2 | 91.9 | 5166 | 51 | 74598 |
| 6 | 23.4 | 109.9 | 5216 | 50 | 106951 |
| 7 | 29.5 | 119.9 | 5266 | 50 | 74544 |
| 8 | 32.2 | 134.6 | 5321 | 55 | 140124 |
| 9 | 35.8 | 146.2 | 5368 | 47 | 234389 |
| 10 | 39.2 | 152.8 | 5426 | 58 | 100962 |

## Tier Distribution (Final Round)

- **clear**: 999084 (99.9%) #################################################
- **clouded**: 794 (0.1%) 
- **flagged**: 0 (0.0%) 
- **lethal_eliminator**: 0 (0.0%) 
