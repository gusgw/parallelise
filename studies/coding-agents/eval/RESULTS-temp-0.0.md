| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_min | debug-best-weather | 0/20 | 41 | 16 | 121.4 | 20x repeat-abort |
| agent_min_improved | debug-best-weather | 0/20 | 56 | 19 | 149.1 | 20x repeat-abort |
| agent_langgraph | debug-best-weather | 20/20 | 156 | 11 | 118.8 | 20x passed |
| agent_langgraph_small | debug-best-weather | 20/20 | 108 | 16 | 129.8 | 20x passed |
| agent_pydantic | debug-best-weather | 6/20 | 99 | 23 | 259.5 | 14x usage-limit, 6x passed |
| agent_pydantic_min | debug-best-weather | 7/20 | 95 | 23 | 259.5 | 13x usage-limit, 7x passed |
| agent_pydantic_typed | debug-best-weather | 0/20 | 462 | 0 | 0.0 | 20x tests-fail |
| agent_pydantic_typed_checked | debug-best-weather | 0/20 | 714 | 0 | 0.0 | 20x tests-fail |
