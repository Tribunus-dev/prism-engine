---
title: Call the server
order: 4
---

Use the OpenAI-compatible local endpoint.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

## Evidence rule

Observed behavior is scoped to the exact commit, build features,
model, machine, route, and receipt. Do not generalize one successful
run into universal backend support.
