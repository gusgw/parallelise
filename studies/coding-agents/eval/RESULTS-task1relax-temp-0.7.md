| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | feature-inline-threshold | 0/8 | 182 | 0 | 0.0 | 5x tests-fail, 3x recursion-limit |
| agent_pydantic | feature-inline-threshold | 0/8 | 251 | 0 | 0.0 | 8x tests-fail |
| agent_min | feature-inline-threshold | 0/8 | 395 | 23 | 331.9 | 3x tests-fail, 3x no-edit, 2x timeout |
