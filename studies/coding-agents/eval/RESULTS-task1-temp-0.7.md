| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | feature-inline-threshold | 0/8 | 148 | 0 | 0.0 | 5x recursion-limit, 3x tests-fail |
| agent_pydantic | feature-inline-threshold | 0/8 | 227 | 0 | 0.0 | 6x tests-fail, 2x usage-limit |
| agent_min | feature-inline-threshold | 0/8 | 253 | 20 | 252.6 | 5x step-limit, 2x no-edit, 1x repeat-abort |
