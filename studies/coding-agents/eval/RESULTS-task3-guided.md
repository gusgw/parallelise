| harness | task | success | median s | med turns | med ktok | breakdown |
|---|---|---|---|---|---|---|
| agent_langgraph | debug-msgid-trace | 0/3 | 94 | 0 | 0.0 | 3x tests-fail |
| agent_pydantic | debug-msgid-trace | 0/3 | 148 | 0 | 0.0 | 2x tests-fail, 1x usage-limit |
| agent_min | debug-msgid-trace | 0/3 | 119 | 27 | 354.5 | 2x tests-fail, 1x repeat-abort |
