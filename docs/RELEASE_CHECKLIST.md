# Prism Release Checklist

> Use this checklist for every Prism release. Each item must be verified or completed before
> a release candidate can proceed to shipping.

---

## Prerequisites

Before starting a release, ensure the following are installed and available:

- **Rust toolchain** — stable channel, matching `rust-toolchain.toml` or the version in
  `Cargo.toml` package metadata. Verify with `rustup show`.
- **Cargo** — latest stable. Verify with `cargo --version`.
- **Git** — configured with commit signing if applicable.
- **CoreML runtime host** — macOS with Apple Silicon or Intel + ANE for fixture and
  qualification tests.
- **Test devices/partitions** — for post-install doctor and rollback tests, a partition or
  device that is not the active boot volume is required.
- **Network access** — to fetch crate dependencies and, for signed manifests, access to
  the release signing key (via `ssh-agent` or hardware token).

---

## Step-by-Step Release Procedure

### 1. Branch and version bump

1. Create a release branch from `main` (or the intended commit):
   `git checkout -b release/vX.Y.Z`
2. Bump the version in `Cargo.toml` to the target release version.
3. Commit the version bump: `git commit -m "chore: bump version to X.Y.Z"`

### 2. Run validation gates

Execute each gate in the order listed below. Do not proceed to the next gate until the
current one passes.

### 3. Generate release artifacts

1. Build release binaries: `cargo build --release`
2. Run the release bundle creation script (if one exists) or archive the build output
   manually.
3. Compute and record checksums for every artifact in the release bundle.
4. Generate the **compatibility manifest** from real qualification runs (see
   Compatibility Manifest).
5. Generate the **release manifest** listing each artifact, its checksum, and the
   compatibility manifest digest.
6. Sign the release manifest with the project signing key.

### 4. Stage and test

1. Deploy the release bundle to a staging environment.
2. Run `prism doctor` post-install and verify all checks pass.
3. Execute the rollback test procedure.
4. Run diagnostics export and verify the output is complete.

### 5. Ship

1. Tag the release commit: `git tag -s vX.Y.Z -m "Release vX.Y.Z"`
2. Push tag: `git push origin vX.Y.Z`
3. Upload the signed release manifest and bundle to the distribution endpoint.
4. Update the stable channel pointer (if applicable).

---

## Checklist

### Test Suite

- [ ] Full repository test suite green (`cargo test`)
- [ ] LUT fixture hermeticity guard green
- [ ] Feature matrix passes (all 6 profiles)
- [ ] Reliability suite green
- [ ] No root-relative config reads

### Compatibility and Qualification

- [ ] Compatibility manifest generated from real qualification
- [ ] At least one real image artifact `RepeatabilityQualified`
- [ ] Stable channel refuses artifacts without `RepeatabilityQualified` status
- [ ] Compatibility manifest digest recorded in release manifest

### Release Integrity

- [ ] Release bundle checksum verification
- [ ] Release manifest signed or checksum-verified
- [ ] Release version bumped in `Cargo.toml`

### Post-Install and Diagnostics

- [ ] Post-install doctor green
- [ ] Rollback test green
- [ ] Diagnostics export test green

### Documentation

- [ ] Known limitations documented

---

## Validation Gates

| Gate | What it checks | How to run |
|---|---|---|
| Full test suite | All unit, integration, and doc tests pass | `cargo test` |
| LUT fixture hermeticity | LUT fixtures produce deterministic, platform-independent results | `cargo test lut_fixture_hermeticity` (or equivalent test target) |
| Feature matrix | All 6 profile combinations compile and pass their smoke tests | `./scripts/test_feature_matrix.sh` or `cargo test --features <profile>` for each profile |
| Reliability suite | Long-running stress/fuzz tests for memory safety and crash resistance | `cargo test --test reliability -- --ignored` |
| Root-relative config guard | The binary does not read configuration files from `/` or other root-absolute paths outside its install prefix | `./scripts/check_root_config.sh` |
| Compatibility manifest | The generated compatibility manifest matches the actual hardware/software capability matrix | `prism manifest generate --compatibility` |
| Repeatability qualification | At least one image artifact has been qualified as repeatable (deterministic output) | `prism qualify --check image` |
| Release bundle checksums | Every file in the release bundle has a matching SHA-256 checksum | `shasum -a 256 -c checksums.txt` |
| Post-install doctor | `prism doctor` reports no errors on a clean install | `prism doctor` |
| Rollback test | Switching from the new release back to the previous release succeeds cleanly | Follow rollback procedure (see Rollback Verification) |
| Diagnostics export | `prism diagnostics export` produces a complete, valid archive | `prism diagnostics export --output /tmp/diag.zip` |

---

## Rollback Verification

1. Ensure the previous release is still available on disk or in the backup bundle.
2. Execute the rollback command or procedure (e.g., `prism rollback --to v<previous>`).
3. Verify the previous version's `prism doctor` passes.
4. Confirm the previous version can process a known-good artifact.
5. Execute the diagnostics export test on the rolled-back installation.
6. Restore the new release after verification completes.

---

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `cargo test` fails with linker errors | Missing system libraries or toolchain mismatch | Run `rustup update` and verify `rustup show` matches `rust-toolchain.toml` |
| LUT fixture test fails on CI but passes locally | Endianness or platform-specific floating-point differences | Ensure the hermeticity guard uses a fixed-point or bit-exact comparison; check CI runner architecture |
| Feature matrix test fails for one profile | Missing feature flag or conditional compilation error | Inspect `cfg` blocks for that profile; verify all feature-gated code compiles with the profile's flag set |
| `prism doctor` reports config read from root path | A hardcoded `/etc/prism` or similar absolute path was introduced | Search codebase for `"/etc/"`, `"/usr/local/etc"`, and similar; replace with platform-appropriate prefix via `dirs` or `configPath` API |
| Rollback fails with "active volume is current release" | Rollback requires a non-active boot volume or dual-partition layout | Boot from a recovery partition or secondary volume, then perform the rollback |
| Diagnostics export produces an empty or truncated archive | A diagnostic collector panicked or timed out | Re-run with `--verbose`; check stdout/stderr for collector failures; increase timeout for slow collectors |
| Release manifest checksum mismatch | Artifact was rebuilt or corrupted after checksum computation | Rebuild from the tagged commit; re-compute checksums immediately before uploading |
| Stable channel accepts an unqualified artifact | Qualification enforcement not wired in the router | Verify admission gate and router enforce `RepeatabilityQualified` for the stable profile; see admission module |
| Compatibility manifest digest mismatch | Manifest was regenerated after the release manifest was signed | Re-run compatibility qualification, regenerate both manifests, and re-sign |
