/* tslint:disable */
/* eslint-disable */

/**
 * The `wasm-bindgen` entry point. JS calls this after the
 * page is parsed.
 *
 * The function reads the prelude from the
 * `<script type="application/json" id="prism-prelude">`
 * element, hydrates the world, then walks the page's DOM
 * regions that carry `data-prism-region` and reconciles
 * each one against the world projection. The reconciliation
 * is a full-region replace for now; a real diff is the next
 * push.
 *
 * Sets `data-prism-hydrated="true"` on `<body>` so tests and
 * the user can verify the hydration ran.
 */
export function prism_hydrate(): void;

/**
 * Read the visitor state as a JSON string. JS can call this
 * to mirror the world state into the JS bridge.
 */
export function prism_visitor_state_json(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly prism_hydrate: () => [number, number];
    readonly prism_visitor_state_json: () => [number, number, number, number];
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
