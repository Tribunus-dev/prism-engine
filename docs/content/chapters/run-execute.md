---
title: Run
order: 3
---

Start the local execution surface.

```bash
cargo run --release --bin prism --features full-apple -- \
  run qwen2.5-0.5b-instruct
```

Interactive chat is also exposed through `prism run <model> --chat`.
