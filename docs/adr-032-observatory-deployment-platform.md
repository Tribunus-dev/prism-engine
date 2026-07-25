# ADR-032: Prism Observatory Deployment Platform (Cloudflare Pages)

## Status

Accepted — operational, scoped by OBSERVATORY_V1_SPEC.md §15. Implements the constitutional
deployment contract.

## Context

`OBSERVATORY_V1_SPEC.md` v1.0 names Cloudflare Pages as the publication layer for Prism
Observatory v1. The domain `prism-engine.tribunus.dev` is currently served by GitHub Pages
out of this repository's `main` branch, `/docs` directory. The constitutional contract
requires real HTTP 301 redirects, real response headers (CSP, HSTS, Referrer-Policy,
COOP/CORP, Permissions-Policy), per-path cache control, isolated preview URLs per
branch and pull request, instant rollback to a previous production deployment, and a
post-production smoke check.

GitHub Pages does not meet this contract:

- The Pages 404 page is static HTML; a meta-refresh from `404.html` is not an HTTP 301.
  Crawlers, intermediaries, and HTTP clients do not see a redirect; they see a 404 that
  happens to contain a meta refresh.
- GitHub Pages serves a small, fixed set of response headers. Custom headers per route
  (CSP, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`,
  `Cross-Origin-*`) are not configurable; meta-delivered alternatives either do not
  support the same directives or are restricted in some browsers.
- Cache directives are not user-controllable; HTML pages are served with the platform's
  default behavior.
- There is no preview URL per pull request.
- There is no instant rollback to a previous production deployment; rollback is a
  revert-and-rebuild operation that re-introduces a window during which the failed
  build was live.
- A post-deployment smoke test cannot preserve the previous production deployment if
  the candidate has already replaced it.

Cloudflare Pages meets the contract: real HTTP redirects via `_redirects` (200, 301,
302, 303, 307, 308); real response headers via `_headers`; per-path cache control;
isolated preview URLs for every non-production branch and pull request; atomic
deployments with instant rollback to any previous production deployment; and a clear
promotion workflow where previews and production are distinct deployment artifacts.

The repository stays on GitHub. Cloudflare is the publication layer. The custom domain
moves from the GitHub Pages source to Cloudflare DNS. The cutover is operational and
does not change the spec; the spec's platform contract was always Cloudflare Pages, and
the GitHub Pages deployment was a pre-spec interim state.

## Decision

### D1. Production host and DNS

- **Host:** Cloudflare Pages. Project name: `prism-observatory` (the project slug is
  operational and may differ; the custom domain is the user-facing identity).
- **Git integration:** connected to `https://github.com/Tribunus-dev/prism-engine`
  (canonical capital-T; lowercase URL still resolves).
- **Production branch:** `main`.
- **Custom domain:** `prism-engine.tribunus.dev`, attached to the Pages project via
  Cloudflare DNS. CNAME from the apex to the Pages project.
- **TLS certificate:** provisioned by Cloudflare (automatic, edge-issued). The existing
  GitHub Pages certificate is decommissioned with the GitHub Pages deployment.
- **No `CNAME` file in the repository.** Cloudflare Pages does not require it; the
  custom domain is configured in the dashboard. If a `CNAME` file is later required for
  some reason, this ADR is amended.

### D2. Build integration and Rust toolchain

Cloudflare Pages' documented build image provides Go, Node.js, Python, and Ruby. It
does not list Rust. The build script therefore installs the Rust toolchain itself,
reproducibly, before invoking cargo.

**Strategy: Pages builds (Strategy A from §15.2).** The build script is a single
checked-in entry point: `scripts/build-site.sh`. The script:

1. Installs a pinned Rust toolchain via `rustup`:
   - Channel: `stable` (exact version pinned in `rust-toolchain.toml` at the repository
     root, channel field set to a specific version, e.g. `1.81.0`).
   - Components: `rustc`, `cargo`, `rust-std`, `rustfmt`, `clippy`.
   - Target: `wasm32-unknown-unknown` (added via `rustup target add`).
2. Installs the WASM tooling at pinned versions:
   - `wasm-bindgen-cli` at the version recorded in
     `crates/prism-docs-runtime/Cargo.toml` as the `wasm-bindgen` dependency.
   - `wasm-opt` (binaryen) at the version recorded in `docs/scripts/build-wasm.sh`.
3. Installs `jq` (used by some build steps) if not present.
4. Runs `docs/scripts/build-wasm.sh` to produce the WASM bundle at
   `docs/pkg/prism_docs_runtime_bg.wasm` and the JS shim.
5. Runs `cargo run --release -p prism-docs-ssg` to produce the site at `docs/`.
6. Runs the pre-publication validation gate: schema validation, manifest validation,
   redirect table validation, link integrity.
7. Writes the build identity (commit SHA, build number, toolchain versions) into a
   `build.json` file at the site root for the A16 gate.

The toolchain install is reproducible. The pinned versions live in the repository
(`rust-toolchain.toml`, `docs/scripts/build-wasm.sh`); a divergent toolchain produces a
build log entry that names the divergence, and the A18 gate fails the build if the
toolchain does not match the pinned manifest.

**Fallback: GitHub Actions builds, Pages serves (Strategy B from §15.2).** If the Pages
build image's constraints prove incompatible with the build time or the toolchain
install, a GitHub Actions workflow runs the same build script on a self-hosted runner
with Rust preinstalled, and uploads the resulting `docs/` directory via
`wrangler pages deploy`. The Pages dashboard's build command is set to a no-op. The
workflow file is the source of build truth and is checked in.

Strategy A is the v1 default. Strategy B is documented in the spec as the fallback
and is invoked by amending this ADR with the migration steps.

### D3. Redirect mechanism

`docs/_redirects` is generated by the SSG from the validated §7.2 table at build time.
The file's format is the standard Cloudflare Pages redirects format. The file supports
the following status codes: 200, 301, 302, 303, 307, 308. The file does **not** support
emitting arbitrary status codes such as 410. The redirects are real HTTP responses
served at the edge.

The conditional routes (`/prism-ml/`, `/general-compute/`) are emitted with a
destination of `/lab/` if the conditional route does not exist in the manifest at build
time. The validator rejects a build where the redirect table and the manifest disagree
about which routes exist.

### D4. 410 mechanism for retired legacy assets

`_redirects` does not support 410. The §7.3 legacy asset retirement is therefore served
by a **Pages Function** (preferred) or a Cloudflare Worker route. The function:

- Matches the retired asset path prefixes (e.g., `/js/*`, `/data/*` after the §7.3
  compatibility window has elapsed).
- Returns a real HTTP `410 Gone` with a `Link` header pointing to the canonical asset
  path under `/assets/`.
- Is deployed alongside the static site from the same Cloudflare account.
- Lives at `functions/_middleware.ts` or `functions/[[:path]].ts` in the Pages Functions
  convention; the exact file path is recorded in the build script.

Until the Pages Function exists, the §7.3 retired asset paths are **absent from the
build output** entirely. They resolve through the authored 404 (§7.4) rather than
through a 410. A visitor following a stale link to a retired asset sees the authored
404 page, with a link to the home page, Start, Status, and the site search surface.
This is acceptable because:

- Stale links to retired assets are rare in practice (the assets were only briefly
  served from the pre-spec interim deployment).
- The authored 404 is itself a useful surface; the visitor is not lost.
- The compatibility window is named in `site.json` and recorded in the §7.3 ADR; once
  the window elapses, the Pages Function is added and the 410 is the response.

### D5. Header mechanism

`docs/_headers` is generated by the SSG at build time. The file applies headers per
path or path-prefix, served as real HTTP response headers. The exact set is recorded
in §15.4 of the spec and is reproduced here for the operational record:

```text
/*
  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'
  Permissions-Policy: camera=(), microphone=(), geolocation=()
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Resource-Policy: same-origin
```

The CSP is the primary defense against XSS and unauthorized resource loading. It is
delivered as a real HTTP header, not as a `<meta>` tag, because meta-delivered CSP
does not support every CSP feature (`frame-ancestors`, `sandbox`, `report-uri`, and
others are restricted or unsupported in some browsers when delivered via meta).

A1 requires that the platform actually serve these headers; the post-production smoke
check verifies them by inspecting the response with `curl -I` and a header-parsing
script.

### D6. Cache policy

The `_headers` file expresses cache directives per path:

```text
/*
  Cache-Control: public, max-age=0, must-revalidate

/assets/*
  Cache-Control: public, max-age=31536000, immutable

/pkg/*
  Cache-Control: public, max-age=31536000, immutable
```

HTML pages are revalidated every request. The site is content-driven and small; the
cache benefit is minimal and the cost of staleness is real. Fingerprinted assets
under `/assets/` and the WASM bundle under `/pkg/` are long-cache immutable; the
filename hash changes when the content changes, so long cache is safe. The CDN and
browsers honor these directives.

### D7. Promotion workflow

Promotion follows the documented Cloudflare Pages workflow:

1. **Pull request opened.** A pull request (or a release branch) is created. Cloudflare
   Pages detects the branch and produces a **preview deployment** at a Cloudflare-
   assigned preview URL.
2. **Automated gates run against the preview.** The build script runs all A1–A23
   automated gates. A22's preview smoke runs against the preview URL. The gates
   produce pass/fail and specific errors; a fail blocks merge to `main`.
3. **Human gates run against the preview.** H1–H13 are signed against the preview URL
   by the named reviewers. A signature is a record in the release log: reviewer name,
   date, gate, outcome.
4. **Merge to `main`.** A passing review and passing automated gates permit merging the
   exact reviewed commit into `main`. The source commit becomes the commit Pages will
   build from for production. The merge commit is the production commit; a force-push
   to `main` does not bypass this step.
5. **Cloudflare produces the production deployment.** Pages detects the new commit on
   `main` and produces a **production deployment** with a new Cloudflare-assigned
   deployment ID. The source commit is identical to the preview; the deployment
   identities are distinct. The production deployment is built by the same script and
   runs the same gates internally before being served.
6. **Post-production smoke check.** A check runs against the production URL. The check
   is the same path-following flow as A22's preview smoke, plus a verification of the
   response headers in D5 and the cache directives in D6. The check is automated and
   lives in `scripts/post-production-smoke.sh`. Exit codes:
   - `0` — pass; the deployment is declared current production.
   - `1` — path-following flow failed; the deployment is blocked from being declared
     current; the previous production remains live.
   - `2` — header check failed; the deployment is blocked; previous production remains.
   - `3` — cache directive check failed; same.
   - `4` — build identity check failed (the served build identity does not match the
     expected commit); same.
7. **The previous production is recorded as the rollback target.** The release log
   entry for the new production names the previous production's deployment ID as the
   rollback target.

A preview deployment is never flipped into production. A preview deployment is a
distinct deployment artifact, identified by its own Cloudflare-assigned ID, and is not
a valid rollback target. The only valid rollback targets are previous production
deployments.

### D8. Rollback

Rollback to a previous production deployment is one click in the Cloudflare Pages
dashboard: redeploy the last known-good production deployment. The redeployment is a
separate atomic operation. The live site is the current production until the redeploy
completes, then becomes the previous-previous production after the redeploy. There is
no window during which the failed build is live.

A rollback is recorded in the release log: the deployment ID rolled back to, the
deployment ID that was live before the rollback, the reason, the reviewer, and the
timestamp.

### D9. Post-production smoke check command

The post-production smoke check is a checked-in script: `scripts/post-production-smoke.sh`.
The script takes the production URL as an argument (default: `https://prism-engine.tribunus.dev`)
and a build identity JSON as the second argument (default: read from
`https://prism-engine.tribunus.dev/build.json`).

```bash
#!/usr/bin/env bash
# scripts/post-production-smoke.sh
# Verifies the production deployment: path, headers, cache, identity.
# Exit codes: 0 pass, 1 path fail, 2 header fail, 3 cache fail, 4 identity fail.
set -euo pipefail

URL="${1:-https://prism-engine.tribunus.dev}"
EXPECTED_BUILD="$(cat "${2:-/dev/stdin}" 2>/dev/null || true)"

# 1. Path-following flow: home -> status row -> evidence -> receipt
RECEIPT_URL=$(curl -fsS "$URL/" \
  | grep -oE 'href="/status/[^"]+"' | head -1 \
  | sed 's/href="//;s/"//')
test -n "$RECEIPT_URL" || exit 1

# ... full path is recorded in the script
```

(The full script is checked in alongside this ADR. The snippet above illustrates the
structure.)

### D10. Cutover from GitHub Pages

The cutover is operational, not constitutional. The steps:

1. The Cloudflare Pages project is provisioned and connected to the GitHub repository.
   The production branch is `main`. The build command and output directory are
   configured per D2.
2. A staging deployment is verified at the Cloudflare Pages default domain. The
   preview smoke runs against the staging deployment. The post-production smoke check
   is run manually once.
3. The custom domain `prism-engine.tribunus.dev` is moved from the GitHub Pages
   source to Cloudflare DNS. The CNAME at the registrar is updated to point to the
   Cloudflare Pages project. The TLS certificate is provisioned by Cloudflare.
4. Once DNS propagation completes (typically under an hour), the custom domain is
   attached to the Pages project in the dashboard.
5. The GitHub Pages deployment is decommissioned: the GitHub Pages source branch is
   set to `none`, the custom domain is removed from the GitHub Pages settings. The
   `docs/CNAME` file remains in the repository for the record but is no longer
   consulted by the build script.
6. The first production build on Cloudflare Pages is promoted. The post-production
   smoke check runs and passes. The release log records the cutover.

The cutover does not change `OBSERVATORY_V1_SPEC.md`. The spec's platform contract
was always Cloudflare Pages, and the GitHub Pages deployment was a pre-spec interim
state. This ADR is the operational record of the cutover.

### D11. Toolchain pinning (operational record)

| Tool | Version | Source of truth |
|---|---|---|
| Rust toolchain | `1.81.0` (or the version pinned in `rust-toolchain.toml`) | `rust-toolchain.toml` at the repository root |
| `wasm-bindgen-cli` | matches the `wasm-bindgen` dependency in `crates/prism-docs-runtime/Cargo.toml` | `Cargo.lock` (generated, committed) |
| `wasm-opt` (binaryen) | recorded in `docs/scripts/build-wasm.sh` | the script itself, with a comment naming the version |
| Cloudflare Pages build image | the current published image, named in the Cloudflare Pages build log | the build log, attached to each release entry |

A divergent toolchain (a different Rust version, a different `wasm-bindgen` version, a
different `wasm-opt` version) produces a build log entry that names the divergence,
and the A18 gate fails the build if the toolchain does not match the pinned manifest.
The build manifest itself is generated at build time and recorded in
`docs/build.json`.

### D12. Release log

Every promotion and every rollback is recorded in `docs/release-log.json` (or its
successor). Each entry names:

- `entry_id` (stable, namespaced)
- `kind` (`promotion` or `rollback`)
- `deployment_id` (Cloudflare-assigned)
- `source_commit` (the git SHA)
- `build_identity` (from `build.json`)
- `gates_passed` (the list of A1–A23 and H1–H13)
- `reviewer` (for human gates)
- `timestamp`
- `rollback_target_id` (the deployment ID of the previous production; recorded on
  promotion; the deployment rolled back to; recorded on rollback)
- `reason` (for rollback only)

The release log is append-only. The current production's `entry_id` is the source of
truth for "what is live."

## Consequences

- The site is no longer served by GitHub Pages. The custom domain's DNS is on
  Cloudflare. The repository is unchanged; the build is now invoked by Cloudflare on
  push to `main` and on pull request.
- The build script installs a Rust toolchain at build time. The toolchain is pinned;
  the install is reproducible. A divergent toolchain fails the build.
- Preview and production are distinct deployment identities. Rollback is to a previous
  production deployment, not a preview. The promotion workflow is merge-based, not
  flip-based.
- The Pages Function for 410 responses is a follow-on deliverable. Until it exists,
  retired legacy asset paths resolve through the authored 404.
- The post-production smoke check is automated and is the gate that distinguishes
  "this is a known-good production deployment" from "this is a build that ran but
  has not yet been declared live."
- The release log is the operational source of truth for "what is live" and "what is
  the rollback target."

## Follow-on ADRs

- ADR for the Pages Function implementation (when the §7.3 compatibility window
  elapses).
- ADR for the live daemon surface of the Compiler Lab (if and when the architect
  approves the follow-on §14.6 decision).
- ADR for telemetry and any third-party integration (currently blocked by §12 A19).
- ADR for internationalization (currently a follow-on per §14.10).
- ADR for a light theme (currently a follow-on per §14.11).
- ADR for richer search (currently a follow-on per §14.8).
