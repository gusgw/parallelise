| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | feature-inline-threshold | 0/8 | 321 | 0 | 0.0 | 5x tests-fail, 2x no-edit, 1x recursion-limit |
| agent_pydantic | feature-inline-threshold | 0/8 | 243 | 0 | 0.0 | 7x tests-fail, 1x no-edit |
| agent_min | feature-inline-threshold | 0/8 | 514 | 20 | 289.9 | 7x tests-fail, 1x repeat-abort |
