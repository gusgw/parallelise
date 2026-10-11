| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | feature-inline-threshold | 0/8 | 245 | 0 | 0.0 | 5x tests-fail, 2x no-edit, 1x recursion-limit |
| agent_pydantic | feature-inline-threshold | 0/8 | 249 | 0 | 0.0 | 8x tests-fail |
| agent_min | feature-inline-threshold | 0/8 | 321 | 21 | 282.5 | 4x repeat-abort, 3x tests-fail, 1x timeout |
