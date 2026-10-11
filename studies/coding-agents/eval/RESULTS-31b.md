| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_min | debug-best-weather | 10/20 | 130 | 16 | 150.9 | 10x passed, 5x step-limit, 5x repeat-abort |
| agent_min_improved | debug-best-weather | 14/20 | 158 | 20 | 182.9 | 14x passed, 6x step-limit |
| agent_langgraph | debug-best-weather | 20/20 | 161 | 12 | 131.3 | 20x passed |
| agent_langgraph_small | debug-best-weather | 19/20 | 124 | 17 | 149.1 | 19x passed, 1x recursion-limit |
| agent_pydantic | debug-best-weather | 18/20 | 130 | 19 | 191.3 | 18x passed, 2x usage-limit |
| agent_pydantic_min | debug-best-weather | 16/20 | 181 | 22 | 236.0 | 16x passed, 4x usage-limit |
| agent_pydantic_typed | debug-best-weather | 0/20 | 134 | 0 | 0.0 | 18x tests-fail, 2x no-edit |
