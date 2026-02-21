# [U1F4CA] Evaluation Report: RL vs Baselines (Simulated)

**Tag:** `pc10_final`
**Date:** 2026-02-21 12:20

## Summary Table

| method                |   final_total_eva |   final_total_rwa |   final_capital_release |   sell_count |   restruct_count |   keep_count |   override_count_total |   override_guardrail |   override_macro |   violations |
|:----------------------|------------------:|------------------:|------------------------:|-------------:|-----------------:|-------------:|-----------------------:|---------------------:|-----------------:|-------------:|
| RL_BALANCEADO         |       7.25865e+08 |       3.39292e+09 |             4.1143e+08  |          168 |              244 |           88 |                      0 |                    0 |                0 |            0 |
| RL_DESINVERSION       |   48541.4         |  508421           |             5.15266e+08 |          408 |                5 |           87 |                      4 |                    4 |                0 |            0 |
| RL_PRUDENCIAL         |       7.48632e+08 |       3.4048e+09  |             7.68279e+07 |           55 |                0 |          445 |                     18 |                   18 |                0 |            0 |
| Baseline_PRUDENCIAL   |      -7.25814e+08 |       6.41918e+09 |             0           |            0 |                0 |          500 |                      0 |                    0 |                0 |            0 |
| Baseline_DESINVERSION |      -2.13418e+08 |       5.70246e+09 |             5.73374e+07 |           68 |                0 |          432 |                      0 |                    0 |                0 |            0 |

## Key Findings
- **Best EVA:** RL_PRUDENCIAL (748,631,942€)
