## MiniLM Topic (300)

- Queries: 300, Top-k: 3, alpha=0.0, tau=0.6
- Overall: Vanilla 47.67% -> Dual 57.28% (+9.61%)

| Category | Count | Vanilla CCR | Dual CCR | Improvement |
| --- | ---: | ---: | ---: | ---: |
| exclusion | 100 | 54.33% | 60.50% | +6.17% |
| negation | 100 | 22.00% | 22.00% | +0.00% |
| numeric | 100 | 66.67% | 89.33% | +22.67% |

## Local Llama-3.2-3B Topic (60)

- Queries: 60, Top-k: 3, alpha=0.0, tau=0.6
- Overall: Vanilla 40.56% -> Dual 57.50% (+16.94%)

| Category | Count | Vanilla CCR | Dual CCR | Improvement |
| --- | ---: | ---: | ---: | ---: |
| exclusion | 24 | 45.83% | 60.42% | +14.58% |
| negation | 19 | 10.53% | 21.05% | +10.53% |
| numeric | 17 | 66.67% | 94.12% | +27.45% |
