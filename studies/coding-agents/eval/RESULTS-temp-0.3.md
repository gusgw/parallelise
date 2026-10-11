| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_min | debug-best-weather | 4/20 | 65 | 18 | 150.4 | 9x repeat-abort, 7x step-limit, 4x passed |
| agent_min_improved | debug-best-weather | 5/20 | 72 | 17 | 134.0 | 11x repeat-abort, 5x passed, 4x step-limit |
| agent_langgraph | debug-best-weather | 20/20 | 147 | 12 | 130.4 | 20x passed |
| agent_langgraph_small | debug-best-weather | 20/20 | 124 | 17 | 141.7 | 20x passed |
| agent_pydantic | debug-best-weather | 12/20 | 107 | 18 | 182.8 | 12x passed, 8x usage-limit |
| agent_pydantic_min | debug-best-weather | 10/20 | 124 | 20 | 205.5 | 10x passed, 10x usage-limit |
| agent_pydantic_typed | debug-best-weather | 0/20 | 191 | 0 | 0.0 | 20x tests-fail |
| agent_pydantic_typed_checked | debug-best-weather | 0/20 | 388 | 0 | 0.0 | 20x tests-fail |
