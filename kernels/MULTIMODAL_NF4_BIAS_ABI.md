# Multimodal NF4 bias ABI — answer + the v1-compatible extension spec

Response to the review finding that the multimodal NF4 path hard-codes zero
biases (`runner.rs` allocates a fresh zero buffer per dispatch) while the
descriptor/packer cannot express bias residency at all — and to the open
question it raises.

## The open question, answered directly

**Are multimodal NF4 biases structurally zero forever, or an interim omission?**

Both, in a precise sense:

- **Zero today by construction, not by accident.** The NF4 quantizer uses the
  symmetric [-1,1] codebook with absmax scaling; `quantize_nf4_group` returns
  `bias = 0` always, and the GPU packer writes `biases_out[..] = 0.0`. This was
  a *measured* decision this branch made: the affine-bias variant was A/B'd
  during the codebook fix and bought −0.2% reconstruction error (noise) vs
  −18.9% for the symmetric codebook. There is no near-term plan for nonzero
  biases.
- **But the bias SLOT is deliberate format generality** — the affine dequant
  `w = codebook[idx]·scale + bias` is baked into the kernel ABI (buffer(2)),
  the main-path loader requires a `BlockBiases` segment
  (`require_nf4_biases()`), and `derive_nf4_tile640_arena_abi` validates a full
  weight/scale/bias triplet. The **multimodal path is the odd one out**: it
  cannot even *describe* bias residency, so if an affine variant ever wins a
  future A/B, multimodal projections need an ABI turn the main path doesn't.

So: the loader's required segment and the shared-residency story are RIGHT;
the multimodal descriptor is the gap. The spec below closes it **without
breaking v1 artifacts** — implement when convenient, or immediately before any
nonzero-bias experiment.

## Why the naive fix (grow the record) is wrong

`ProjectionTensorRecord` is `#[repr(C)]` and the loader walks the table with
`size_of::<ProjectionTensorRecord>()` stride (cimage_loader.rs:938, 1060,
1066). Adding `bias_offset`/`bias_length` fields changes the stride and
silently mis-parses every existing v1 cimage. Rejected.

## The v1-compatible design (all mechanics verified against the tree)

Three observations make a no-stride-change extension possible:

1. **Bias geometry is scale-parallel by construction.** For NF4Tile640,
   biases have byte-for-byte the same layout as scales (`[tiles × 5]` f32 per
   row — the same invariant the main arena ABI enforces). A record therefore
   needs NO independent bias offsets: lay the bias segment out **parallel to
   the scale segment** (`bias_offset ≡ scale_offset`, `bias_length ≡
   scale_length`, different segment).
2. **The descriptor has exactly one spare field**: `image_reserved: u16` —
   the exact width of a segment index. Repurpose it as
   `projection_bias_segment_index` (same offset, same size → descriptor layout
   unchanged, magic/version untouched).
3. **`ProjectionTensorRecord.flags: u8` is written as 0 by every existing
   packer.** Gate ALL bias reads on a new record flag, and v1 artifacts can
   never take the bias path — not even by accident.

### The changes, file by file

| File | Change |
|---|---|
| `compile/ternary.rs` | `SegmentKind::MultimodalProjectionBiases = 26` (25 = ModelArtifacts is the current tail) |
| `multimodal/descriptor.rs` | rename `image_reserved` → `projection_bias_segment_index` (u16, same slot); add `ProjectionTensorRecord::FLAG_HAS_BIAS: u8 = 1 << 0`; extend `validate_nf4_tile640`: flag set ⇒ `scale_length > 0` |
| `cimage_packer/pipeline.rs` | add `projection_biases: Vec<u8>` to `SynthesizedMultimodalSegments`, filled **parallel to** `projection_scales` from the NF4 pack output (currently dropped); new `SEG_MM_PROJ_BIASES` plan segment + write path (mirror the scales sites at ~1191/1315/1606); include in `payload_hasher` (~1798); descriptor index assignment mirrors the scales pattern (~822: real index when `length > 0`, else `u16::MAX`); set `FLAG_HAS_BIAS` on each NF4 record whose biases were captured |
| `multimodal/binding.rs` | `resolve_optional_segment(..., MultimodalProjectionBiases)`; expose a per-record bias view mirroring the scale accessor (offsets are the record's `scale_offset`/`scale_length` — the parallel-layout contract) |
| `orchestrator/runner.rs` | `run_nf4_multimodal_projection`: if `record.flags & FLAG_HAS_BIAS != 0` and the bias binding resolved → bind the real segment slice; **else keep the zero-filled buffer as the documented v1-compat fallback** (it is numerically correct for every artifact the symmetric quantizer has ever produced) |

### Compatibility proof

- **Old reader, new artifact**: v1 loaders ignore unknown segment kinds and the
  repurposed u16 (they never read `image_reserved`); records parse at the same
  stride; the flag bit is unknown but unconsulted. Dequant remains correct
  because new artifacts still carry zero biases until a nonzero-bias quantizer
  exists.
- **New reader, old artifact**: `flags == 0` on every record ⇒ bias path never
  consulted ⇒ zero-buffer fallback ⇒ bit-identical behavior. The zeroed
  `image_reserved` field (which would otherwise alias segment index 0) is
  unreachable behind the record flag — this is why the flag gate, not the
  descriptor index, is the load-bearing guard.
- **Absence sentinel**: matches the existing convention (`u16::MAX` /
  out-of-range → `resolve_optional_segment` returns `None`).

### Validation gates (when implemented)

1. Pack a tiny multimodal NF4 fixture → assert bias segment length ==
   scale segment length, records carry the flag, descriptor index resolves.
2. Round-trip: new reader on a **pre-change cimage** → byte-identical dispatch
   inputs vs the zero-buffer path (the v1-compat regression test).
3. `derive`-style parallel check: for every flagged record,
   `bias view length == scale view length` at load time (mirror the main
   ABI's triplet validation).

## Status

- Runner's zero-bias allocation stays (correct for all current artifacts) but
  is now documented as the v1-compat fallback of this spec rather than an
  unexplained hack.
- Implementation is deliberately deferred to a single clean pass — a
  binary-format change should not ship half-done; this spec is written so that
  pass is mechanical.
