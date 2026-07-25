---
title: Pull and compile
order: 2
---

From model identity to ComputeImage.

```bash
cargo run --release --bin prism --features full-apple -- \
  pull Qwen/Qwen2.5-0.5B-Instruct
```

The compiled model is stored under `${PRISM_HOME:-$HOME/.prism}/models/<name>/`
with `model.cimage`, configuration, and tokenizer artifacts.
