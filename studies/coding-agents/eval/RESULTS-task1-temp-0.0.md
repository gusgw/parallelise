| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | feature-inline-threshold | 0/8 | 279 | 0 | 0.0 | 4x recursion-limit, 4x tests-fail |
| agent_pydantic | feature-inline-threshold | 0/8 | 243 | 0 | 0.0 | 8x tests-fail |
| agent_min | feature-inline-threshold | 0/8 | 498 | 20 | 268.5 | 7x step-limit, 1x tests-fail |
