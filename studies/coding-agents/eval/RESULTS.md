| harness | task | success | median s | breakdown |
|---|---|---|---|---|
| agent_min | debug-best-weather | 4/20 | 461 | 11x repeat-abort, 4x passed, 2x no-edit, 2x step-limit, 1x tests-fail |
| agent_min_improved | debug-best-weather | 3/20 | 699 | 12x step-limit, 3x timeout, 3x passed, 1x tests-fail, 1x repeat-abort |
| agent_langgraph | debug-best-weather | 6/20 | 619 | 7x recursion-limit, 6x passed, 4x timeout, 2x tests-fail, 1x no-edit |
| agent_langgraph_small | debug-best-weather | 6/20 | 424 | 8x recursion-limit, 6x passed, 2x timeout, 2x tests-fail, 2x still-broken |
| agent_pydantic | debug-best-weather | 6/20 | 707 | 6x tests-fail, 6x passed, 6x timeout, 2x usage-limit |
| agent_pydantic_min | debug-best-weather | 4/20 | 602 | 9x tests-fail, 4x passed, 3x usage-limit, 3x timeout, 1x still-broken |
| agent_pydantic_typed | debug-best-weather | 0/20 | 40 | 12x no-edit, 4x still-broken, 4x tests-fail |
