| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | feature-inline-threshold | 0/8 | 208 | 0 | 0.0 | 5x recursion-limit, 3x tests-fail |
| agent_pydantic | feature-inline-threshold | 0/8 | 264 | 0 | 0.0 | 7x tests-fail, 1x usage-limit |
| agent_min | feature-inline-threshold | 0/8 | 181 | 20 | 207.9 | 5x step-limit, 2x repeat-abort, 1x no-edit |
