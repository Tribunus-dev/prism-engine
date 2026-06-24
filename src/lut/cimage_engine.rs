#[cfg(feature = "prism-backend")]
pub mod cimage_engine {
    use std::path::Path;
    use std::time::Instant;
    use mlx_rs::{Array, Stream, fast, nn};
    use mlx_rs::ops::{quantize_device, quantized_matmul_device, concatenate_device, add, multiply};
    use mlx_rs::ops::indexing::argmax_axis_device;
    use tribunus_compute_core::quantization::cimage::CImageReader;
    use tribunus_compute_core::lut::graph::{ComputeNode, ModelGraph};
    use tribunus_compute_core::lut::engine::InferenceStats;

    struct QW { w: Array, s: Array, b: Array }
    struct LW { q: Option<QW>, k: Option<QW>, v: Option<QW>, o: Option<QW>,
                gate: Option<QW>, up: Option<QW>, down: Option<QW> }

    pub struct CimageEngine {
        layers: Vec<LW>,
        tok_emb: Array, norm_w: Array, lm_head: Option<QW>,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        vocab_size: usize, rope_theta: f32, norm_eps: f32,
    }

    impl CimageEngine {
        pub fn load(path: &Path, graph: ModelGraph) -> Result<Self, String> {
            let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
            let reader = CImageReader::open(path)?;
            let st = Stream::default();
            let n_layers = graph.num_layers as usize;
            let (n_heads, n_kv_heads, head_dim) = cfg_attn(&graph);

            // Get all tensor keys from cimage (format independent of graph keys)
            let all_keys: Vec<String> = reader.header.tensors.keys().cloned().collect();

            // Find embed by key suffix
            let embed_key = all_keys.iter().find(|k| k.contains("embed_tokens"))
                .ok_or_else(|| "no embed_tokens in cimage".to_string())?;
            // Use cimage tensor record dimensions (graph may be wrong)
            let embed_rec = reader.tensor(embed_key).unwrap();
            let vs = embed_rec.dim_m as usize;
            let hidden_dim = embed_rec.dim_n as usize;
            // Embedding is palettized — dequantize to FP32
            let tok_emb = load_one_qw_f32(&reader, &data, embed_key)
                .ok_or_else(|| "dequantize embed failed".to_string())?;

            // Find norm weight
            let norm_key = all_keys.iter().find(|k| k.contains("norm") && k.contains("weight") && !k.contains("layer"));
            let norm_w = match norm_key {
                Some(k) => load_one_f32_1d(&reader, &data, k, hidden_dim)?,
                None => Array::from_slice(&[1.0f32], &[1]),
            };

            // Build per-layer weights by matching cimage keys by pattern
            let mut layers: Vec<LW> = (0..n_layers).map(|_| LW {
                q: None, k: None, v: None, o: None, gate: None, up: None, down: None,
            }).collect();

            for key in &all_keys {
                let li = match key.split(".layers.").nth(1).and_then(|s| s.split('.').next().and_then(|n| n.parse::<usize>().ok())) {
                    Some(idx) if idx < n_layers => idx,
                    _ => continue,
                };
                let w = load_one_qw(&reader, &data, key, &st);
                if key.contains(".q_proj.") { layers[li].q = w; }
                else if key.contains(".k_proj.") { layers[li].k = w; }
                else if key.contains(".v_proj.") { layers[li].v = w; }
                else if key.contains(".o_proj.") { layers[li].o = w; }
                else if key.contains(".gate_proj.") { layers[li].gate = w; }
                else if key.contains(".up_proj.") { layers[li].up = w; }
                else if key.contains(".down_proj.") { layers[li].down = w; }
            }

            let lm_head = all_keys.iter().find_map(|k| {
                if k.contains("lm_head") { load_one_qw(&reader, &data, k, &st) } else { None }
            });

            Ok(Self {
                layers, tok_emb, norm_w, lm_head,
                n_heads, n_kv_heads, head_dim,
                vocab_size: vs, rope_theta: cfg_rope(&graph), norm_eps: cfg_eps(&graph),
            })
        }

        pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<InferenceStats, String> {
            let t0 = Instant::now();
            let st = Stream::default();
            let mut kv: Vec<(Option<Array>, Option<Array>)> = (0..self.layers.len()).map(|_| (None, None)).collect();
            let mut gen = Vec::with_capacity(max_tokens);
            let mut pos: i64 = 0;
            let mut h = embed(&self.tok_emb, prompt[0] as usize);
            for _ in 0..max_tokens {
                for li in 0..self.layers.len() {
                    let lw = &self.layers[li];
                    let hn = fast::rms_norm_device(&h, &self.norm_w, self.norm_eps, &st).unwrap();
                    let q = qm(&hn, lw.q.as_ref().unwrap(), &st);
                    let k = qm(&hn, lw.k.as_ref().unwrap(), &st);
                    let v = qm(&hn, lw.v.as_ref().unwrap(), &st);
                    let nh = self.n_heads as i32; let nkv = self.n_kv_heads as i32; let hd = self.head_dim as i32;
                    let qr = fast::rope_device(&q.reshape(&[1, nh, hd]).unwrap(), hd, false, Some(self.rope_theta), 1.0, pos as i32, None::<&Array>, &st).unwrap();
                    let kr = fast::rope_device(&k.reshape(&[1, nkv, hd]).unwrap(), hd, false, Some(self.rope_theta), 1.0, pos as i32, None::<&Array>, &st).unwrap();
                    let vr = v.reshape(&[1, nkv, hd]).unwrap();
                    let new_k = match kv[li].0.take() {
                        Some(prev_k) => concatenate_device(&[prev_k, kr], &st).unwrap(),
                        None => kr,
                    };
                    let new_v = match kv[li].1.take() {
                        Some(prev_v) => concatenate_device(&[prev_v, vr], &st).unwrap(),
                        None => vr,
                    };
                    kv[li].0 = Some(new_k);
                    kv[li].1 = Some(new_v);
                    let fk_ref = kv[li].0.as_ref().unwrap();
                    let fv_ref = kv[li].1.as_ref().unwrap();
                    let attn = fast::scaled_dot_product_attention_device(&qr, fk_ref, fv_ref, 1.0 / (hd as f32).sqrt(), None::<fast::ScaledDotProductAttentionMask>, &st).unwrap();
                    let attn = attn.reshape(&[1, (nh * hd) as i32]).unwrap();
                    h = add(&h, &qm(&attn, lw.o.as_ref().unwrap(), &st)).unwrap();
                    let hn = fast::rms_norm_device(&h, &self.norm_w, self.norm_eps, &st).unwrap();
                    let gate = nn::silu(&qm(&hn, lw.gate.as_ref().unwrap(), &st)).unwrap();
                    let up = qm(&hn, lw.up.as_ref().unwrap(), &st);
                    let gu = multiply(&gate, &up).unwrap();
                    h = add(&h, &qm(&gu, lw.down.as_ref().unwrap(), &st)).unwrap();
                }
                h = fast::rms_norm_device(&h, &self.norm_w, self.norm_eps, &st).unwrap();
                let logits = match &self.lm_head {
                    Some(ref lm) => qm(&h, lm, &st),
                    None => h.slice(&[0, 0], &[1, self.vocab_size as i32], &[1, 1]).unwrap().to_owned(),
                };
                let flat = logits.reshape(&[-1]).unwrap();
                let next: u32 = argmax_axis_device(&flat, 0, false, &st).unwrap().item();
                gen.push(next); pos += 1;
                if next == 0 || next == 2 { break; }
                h = embed(&self.tok_emb, next as usize);
            }
            Ok(InferenceStats { prompt_tokens: prompt.len(), generated_tokens: gen, total_time_ms: t0.elapsed().as_secs_f64() * 1000.0 })
        }
    }

    fn embed(e: &Array, token: usize) -> Array {
        let s = e.shape();
        e.slice(&[token as i32, 0], &[token as i32 + 1, s[1]], &[1, 1]).unwrap().to_owned()
    }

    fn qm(x: &Array, w: &QW, st: &Stream) -> Array {
        quantized_matmul_device(x, &w.w, &w.s, &w.b, false, 64, 4, st).unwrap()
    }
    /// Dequantize a palettized cimage tensor to FP32 (no final quantization).
    fn load_one_qw_f32(r: &CImageReader, d: &[u8], key: &str) -> Option<Array> {
        let rec = r.tensor(key)?;
        let p = &d[rec.offset as usize..][..rec.size as usize];
        let om = rec.dim_m as usize;
        let im = rec.dim_n as usize;
        let mut f32v = Vec::with_capacity(om * im);
        for row in 0..om {
            let mut cb = [0.0f32; 16];
            for i in 0..16 {
                cb[i] = half::f16::from_bits(u16::from_le_bytes([p[row*32 + i*2], p[row*32 + i*2 + 1]])).to_f32();
            }
            let io = om * 32 + row * ((im + 1) / 2);
            for i in 0..im {
                let byte = p[io + i / 2];
                f32v.push(cb[if i % 2 == 0 { byte & 0x0F } else { byte >> 4 } as usize]);
            }
        }
        Some(Array::from_slice(&f32v, &[om as i32, im as i32]))
    }

    fn load_one_f32_1d(r: &CImageReader, d: &[u8], key: &str, n: usize) -> Result<Array, String> {
        let arr = load_one_f32(r, d, key)?;
        // Reshape from flat to [n]
        Ok(arr.reshape(&[n as i32]).unwrap_or(arr))
    }

    fn load_one_f32_2d(r: &CImageReader, d: &[u8], key: &str, rows: usize, cols: usize) -> Result<Array, String> {
        let arr = load_one_f32(r, d, key)?;
        Ok(arr.reshape(&[rows as i32, cols as i32]).unwrap_or(arr))
    }

    fn load_one_f32(r: &CImageReader, d: &[u8], key: &str) -> Result<Array, String> {
        let rec = r.tensor(key).ok_or_else(|| format!("missing: {key}"))?;
        let p = &d[rec.offset as usize..][..rec.size as usize];
        let mut f32v = Vec::with_capacity(p.len() / 2);
        for i in 0..p.len() / 2 {
            f32v.push(half::f16::from_bits(u16::from_le_bytes([p[i*2], p[i*2+1]])).to_f32());
        }
        Ok(Array::from_slice(&f32v, &[f32v.len() as i32]))
    }

    fn load_one_qw(r: &CImageReader, d: &[u8], key: &str, st: &Stream) -> Option<QW> {
        let rec = r.tensor(key)?;
        let p = &d[rec.offset as usize..][..rec.size as usize];
        let om = rec.dim_m as usize; let im = rec.dim_n as usize;
        let mut f32v = Vec::with_capacity(om * im);
        for row in 0..om {
            let mut cb = [0.0f32; 16];
            for i in 0..16 {
                cb[i] = half::f16::from_bits(u16::from_le_bytes([p[row*32 + i*2], p[row*32 + i*2 + 1]])).to_f32();
            }
            let io = om * 32 + row * ((im + 1) / 2);
            for i in 0..im {
                let byte = p[io + i / 2];
                f32v.push(cb[if i % 2 == 0 { byte & 0x0F } else { byte >> 4 } as usize]);
            }
        }
        let (w, s, b) = quantize_device(&Array::from_slice(&f32v, &[om as i32, im as i32]), Some(64), Some(4), st).ok()?;
        Some(QW { w, s, b })
    }

    fn cfg_attn(g: &ModelGraph) -> (usize, usize, usize) {
        for n in &g.nodes { if let ComputeNode::ScaledDotProductAttention { num_heads, num_kv_heads, head_dim } = n { return (*num_heads as usize, *num_kv_heads as usize, *head_dim as usize); }}
        (14, 2, 64)
    }
    fn cfg_rope(g: &ModelGraph) -> f32 { for n in &g.nodes { if let ComputeNode::RotaryEmbedding { rope_theta, .. } = n { return *rope_theta; }} 10000.0 }
    fn cfg_eps(g: &ModelGraph) -> f32 { for n in &g.nodes { if let ComputeNode::Norm { eps, .. } = n { return *eps; }} 1e-6 }
}
