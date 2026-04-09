# Error Analysis Cases

- Source report: `outputs/reports/retrieval_metrics_local_dual.json`
- Number of cases: 20

## Case 1: q-0
- Query: Recommend phones under 100 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 2: q-1
- Query: Suggest menus without items that high sugar
- Category: exclusion
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-7`
  - The menu high sugar.
- Dual top1: `doc-6`
  - This menu is low sugar.

## Case 3: q-2
- Query: Suggest menus without items that high sugar
- Category: exclusion
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-7`
  - The menu high sugar.
- Dual top1: `doc-6`
  - This menu is low sugar.

## Case 4: q-4
- Query: Suggest menus without items that high sugar
- Category: exclusion
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-7`
  - The menu high sugar.
- Dual top1: `doc-6`
  - This menu is low sugar.

## Case 5: q-5
- Query: Recommend phones under 300 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 6: q-6
- Query: Suggest menus without items that high sugar
- Category: exclusion
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-7`
  - The menu high sugar.
- Dual top1: `doc-6`
  - This menu is low sugar.

## Case 7: q-7
- Query: Recommend phones under 300 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 8: q-9
- Query: Recommend phones under 300 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 9: q-10
- Query: Recommend phones under 100 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 10: q-11
- Query: Recommend phones under 300 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 11: q-12
- Query: Recommend phones under 500 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-18`
  - Budget phone priced at $249 with 4GB RAM.

## Case 12: q-13
- Query: Recommend phones under 300 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 13: q-14
- Query: Recommend phones under 100 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 14: q-15
- Query: Suggest menus without items that high sugar
- Category: exclusion
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-7`
  - The menu high sugar.
- Dual top1: `doc-6`
  - This menu is low sugar.

## Case 15: q-16
- Query: Recommend phones under 300 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 16: q-18
- Query: Find rooms that are not noisy
- Category: negation
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: negation-scope failure
- Vanilla top1: `doc-39`
  - Many reviews complain the room feels noisy.
- Dual top1: `doc-39`
  - Many reviews complain the room feels noisy.

## Case 17: q-20
- Query: Suggest menus without items that high sugar
- Category: exclusion
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-7`
  - The menu high sugar.
- Dual top1: `doc-6`
  - This menu is low sugar.

## Case 18: q-22
- Query: Suggest menus without items that high sugar
- Category: exclusion
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-7`
  - The menu high sugar.
- Dual top1: `doc-6`
  - This menu is low sugar.

## Case 19: q-23
- Query: Recommend phones under 100 dollars
- Category: numeric
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: hard residual case
- Vanilla top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.
- Dual top1: `doc-0`
  - Budget phone priced at $89 with 4GB RAM.

## Case 20: q-24
- Query: Find rooms that are not noisy
- Category: negation
- Vanilla CCR@10: 0.2000
- Dual CCR@10: 0.2000
- Delta: 0.0000
- Failure type: negation-scope failure
- Vanilla top1: `doc-39`
  - Many reviews complain the room feels noisy.
- Dual top1: `doc-39`
  - Many reviews complain the room feels noisy.
