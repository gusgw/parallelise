| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | debug-msgid-trace | 0/2 | 85 | 0 | 0.0 | 2x recursion-limit |
| agent_pydantic | debug-msgid-trace | 0/2 | 295 | 0 | 0.0 | 2x usage-limit |
| agent_min | debug-msgid-trace | 0/2 | 23 | 20 | 124.2 | 1x step-limit, 1x no-edit |
