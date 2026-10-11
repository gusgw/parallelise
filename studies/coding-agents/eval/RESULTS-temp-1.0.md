| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_min | debug-best-weather | 13/20 | 133 | 20 | 180.7 | 13x passed, 7x step-limit |
| agent_min_improved | debug-best-weather | 15/20 | 159 | 20 | 180.3 | 15x passed, 5x step-limit |
| agent_langgraph | debug-best-weather | 20/20 | 155 | 13 | 148.5 | 20x passed |
| agent_langgraph_small | debug-best-weather | 20/20 | 120 | 18 | 145.5 | 20x passed |
| agent_pydantic | debug-best-weather | 20/20 | 146 | 19 | 196.8 | 20x passed |
| agent_pydantic_min | debug-best-weather | 20/20 | 153 | 18 | 194.1 | 20x passed |
| agent_pydantic_typed | debug-best-weather | 0/20 | 141 | 0 | 0.0 | 19x tests-fail, 1x no-edit |
| agent_pydantic_typed_checked | debug-best-weather | 0/20 | 223 | 0 | 0.0 | 19x tests-fail, 1x no-edit |
