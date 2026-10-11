| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | feature-inline-threshold | 0/8 | 154 | 0 | 0.0 | 6x recursion-limit, 2x tests-fail |
| agent_pydantic | feature-inline-threshold | 0/8 | 306 | 0 | 0.0 | 8x tests-fail |
| agent_min | feature-inline-threshold | 0/8 | 404 | 20 | 260.6 | 8x step-limit |
