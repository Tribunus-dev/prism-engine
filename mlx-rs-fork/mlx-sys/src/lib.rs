#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Level Zero stubs — provide symbols missing when MLX is built without the
// Level Zero (Intel GPU) backend.
include!("level_zero_stubs.rs");
