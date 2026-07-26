# Prism Observatory v1 — A-list audit

- **Source:** `local:/Users/user/Developer/GitHub/prism-engine/docs`
- **Generated:** 2026-07-26T09:15:32Z
- **Total:** 22
- **Pass:** 11  **Fail:** 0  **Skip:** 8  **Warn:** 3

| # | Status | Axiom | Evidence | Detail |
|---|--------|-------|----------|--------|
| 1 | ✓ PASS | **A1** Route integrity | 12 canonical routes, 404 present, no legacy surface |  |
| 2 | ! WARN | **A2** Status-vocabulary purity (linter) | 21 potential match(es) | H1 review queue:   • /: 'active' in "urface-3:    #262638;  /* the active card; the selected stage */  …"   • /: 'active' in "lemented:  #6ad4ff;  /* cyan; active; present */   --color-state-q…"   • /: 'active' in "te. */ @media (forced-colors: active) {   :root {     --color-bg: …"   • /: 'active' in " target sizing. Anything interactive below 600px  * viewport gets …"   • /: 'live' in "e not streaming. They are not live. They are the receipt's evide…"   • /architecture/: 'active' in "urface-3:    #262638;  /* the active card; the selected stage */  …"   • /architecture/: 'active' in "lemented:  #6ad4ff;  /* cyan; active; present */   --color-state-q…"   • /architecture/: 'active' in "te. */ @media (forced-colors: active) {   :root {     --color-bg: …"   • …and 13 more |
| 3 | ✓ PASS | **A3** Data-layer validation | 12 JSON file(s) parsed |  |
| 4 | ✓ PASS | **A4** Evidence-boundary completeness | 0 validated, 0 qualifying, 0 released — all fields present |  |
| 5 | ✓ PASS | **A5** Chapter list locality | no `.chapters` list with >20 items (largest seen: 0 on ) |  |
| 6 | ✓ PASS | **A6** Component registration | 22 components, each with a single authority statement |  |
| 7 | ✓ PASS | **A7** Manuscript-to-page match | 12 v1 pages in manuscript (15 total), 13 rendered (incl. 404), all routed |  |
| 8 | ✓ PASS | **A8** Diagram caption and description | no <figure> elements in the rendered site |  |
| 9 | ○ SKIP | **A9** Reduced motion compliance | static CSS has prefers-reduced-motion block; full visual check needs a browser |  |
| 10 | ○ SKIP | **A10** Keyboard parity | static proxies pass (:focus-visible, skip link, <button>); full tab-order traversal needs a browser |  |
| 11 | ○ SKIP | **A11** Screen-reader parity | landmarks present on 12/12 pages, 0 images all with alt; full check (heading nesting, aria-live) needs axe-core |  |
| 12 | ✓ PASS | **A12** No-JS rendering | 13 pages, min 3448 chars, Observatory has 12 stages in HTML |  |
| 13 | ✓ PASS | **A13** Schema and cross-reference integrity | 13 schema file(s) parsed |  |
| 14 | ○ SKIP | **A14** Evidence applicability | claims.json not present |  |
| 15 | ! WARN | **A15** Canonical URLs, sitemap, robots | canonical 12/12, og 12/12, twitter 12/12 | no /sitemap.xml; no /robots.txt |
| 16 | ✓ PASS | **A16** Build identity and source commit | build_id on all 12 pages, build.json has build_id and commit |  |
| 17 | ○ SKIP | **A17** Status not by color alone | static proxies pass (shape rule + forced-colors block); full visual check needs a forced-colors render |  |
| 18 | ○ SKIP | **A18** Performance budget | max HTML 14302 B at /computeimage/specimen/, max critical CSS 10531 B at /computeimage/, max JS 0 B at , bundle 20613 B; LCP/CLS/INP need a browser |  |
| 19 | ✓ PASS | **A19** Security and privacy | no third-party requests, no eval, no inline event handlers, no iframes, no extra inline scripts |  |
| 20 | ○ SKIP | **A20** Accessibility extras | static proxies pass (44px touch targets, fg/bg tokens); full check (contrast ratios, 200%/400% zoom) needs a browser |  |
| 21 | ! WARN | **A21** /docs/ allowlist | 52 file(s) at site root | not in the expected root set: prism-observation-protocol.md, adr-002-progressive-tensor-update.md, adr-033-observatory-schema-binding.md, ARCHITECTURE.md, adr-032-observatory-deployment-platform.md, adr-027-memory-model.md, prism-bonsai-technical-brief.md, prism-meaning-runtime.md, physics-of-prism.md, .DS_Store, CAMPAIGN.md, prism_semantic_region_ir_implementation_plan.md, maturity-audit-2026-07-15.md, adr-003-canonical-ecs-world.md, cimage-layout-abi-v1.md, adr-028-ntb-cluster-coordination.md, prism-interaction-runtime.md, adr-005-ecs-native-compiler-absorption.md, semantic-region-ir.md, website-full-materialization-plan.md, RELEASE_CHECKLIST.md, prism-semantics.md, MIGRATION.md, adr-001-resumable-ternary-distill-compiler.md, current-capabilities.md, audit-report.md, prism-constitution.md, adr-029-prism-fabric-os-blackhole.md, semantic-region-ir-full-implementation-plan.md, pareto-optimization.md, adr-031-aiter-atom-rocm-provider.md, design-review-checklist.md, prism-epistemology.md, site.js, prism-engine-website-full-materialization-plan.md, prism-experience-architecture.md, prism-experience-architecture-v2.md, adr-034-observatory-evidence-freeze.md, adr-026-workspace-consolidation.md, adr-030-prism-fabric-os-strix-halo.md, favicon.svg, ecs-architecture.md, prism-runtime.md, prism-ontology.md |
| 22 | ○ SKIP | **A22** Deployment smoke test | 13 routes present, build_id=ssg-17380, commit=8550c1b4; live path: home → status row → evidence → receipt |  |
