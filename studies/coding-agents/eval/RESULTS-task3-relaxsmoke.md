| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | debug-msgid-trace | 0/3 | 352 | 0 | 0.0 | 2x no-edit, 1x recursion-limit |
| agent_pydantic | debug-msgid-trace | 0/3 | 111 | 0 | 0.0 | 1x tests-fail, 1x no-edit, 1x usage-limit |
| agent_min | debug-msgid-trace | 0/3 | 19 | 14 | 92.9 | 2x no-edit, 1x repeat-abort |
