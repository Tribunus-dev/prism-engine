use std::{env, path::PathBuf, process::Command};

fn build_and_link_mlx_c() {
    // build the mlx-c project
    // Step 1: cmake configure (fetches sources, creates build tree)
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mlx_c_src = manifest_dir.join("src/mlx-c");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("build");
    let install_prefix = out_dir.join("install");

    let mut cmake_args = vec![
        "-S".to_string(), mlx_c_src.to_str().unwrap().to_string(),
        "-B".to_string(), build_dir.to_str().unwrap().to_string(),
        format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.to_str().unwrap()),
        format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={}", std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "26.5".into())),
        "-DMLX_BUILD_METAL=OFF".to_string(),
        "-DMLX_BUILD_ACCELERATE=OFF".to_string(),
    ];

    #[cfg(debug_assertions)]
    { cmake_args.push("-DCMAKE_BUILD_TYPE=Debug".to_string()); }
    #[cfg(not(debug_assertions))]
    { cmake_args.push("-DCMAKE_BUILD_TYPE=Release".to_string()); }
    #[cfg(feature = "metal")]
    { cmake_args.push("-DMLX_BUILD_METAL=ON".to_string()); }
    #[cfg(feature = "accelerate")]
    { cmake_args.push("-DMLX_BUILD_ACCELERATE=ON".to_string()); }

    // Use local mlx-tribunus checkout instead of fetching from GitHub.
    // Avoids authentication issues with the private repository and
    // guarantees a consistent source tree.
    cmake_args.push(format!(
        "-DFETCHCONTENT_SOURCE_DIR_MLX={}",
        manifest_dir.join("../../../mlx-tribunus").canonicalize().unwrap_or_else(|_| manifest_dir.join("../../../mlx-tribunus")).display()
    ));

    // Metal shader compilation is disabled because the mlx-tribunus fork's
    // .metal files have bfloat16_t/half type collisions (
    // bf16.h: `typedef half bfloat16_t`) that produce duplicate template
    // instantiations on this Metal compiler.  On M1–M4 we prefer fp32 GPU
    // compute anyway (fp32 ALUs make fp16 conversions wasteful), so the
    // custom Metal shader library is unnecessary — MLX uses Accelerate +
    // CPU backends for all compute.
    cmake_args.push("-DMLX_BUILD_METAL=OFF".to_string());
    eprintln!("Metal shader compilation disabled (mlx-tribunus fork bfloat16_t issue)");

    let status = Command::new("cmake")
        .args(&cmake_args)
        .status()
        .expect("failed to run cmake configure");
    if !status.success() {
        panic!("cmake configure failed");
    }

    // Patch bf16.h: apply struct-based bfloat16_t fallback for macOS 26+
    let patches_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("patches");
    // (without this, bfloat16_t == half, which causes duplicate instantiations
    // across every .metal file that instantiates BOTH float16_t and bfloat16_t)
    let bf16_path = build_dir.join("_deps/mlx-src/mlx/backend/metal/kernels/bf16.h");
    let bf16_patched = patches_dir.join("bf16_patched.h");
    if bf16_patched.exists() && bf16_path.exists() {
        let content = std::fs::read_to_string(&bf16_patched).unwrap_or_default();
        // Upstream recommendation: use __HAVE_BFLOAT__ check (works in JIT context)
        let content = content.replace(
            "__has_extension(metal_bfloat)",
            "defined(__HAVE_BFLOAT__)",
        );
        std::fs::write(&bf16_path, &content).unwrap();
        eprintln!("Patched bf16.h with struct-based bfloat16_t fallback (__HAVE_BFLOAT__)");
    }

    // Patch bf16_math.h: guard half-typed instantiations on macOS 26+
    // where bfloat16_t falls back to `half` and Metal already provides
    // native half math functions.
    let bf16_math_path = build_dir.join("_deps/mlx-src/mlx/backend/metal/kernels/bf16_math.h");
    if bf16_math_path.exists() {
        let content = std::fs::read_to_string(&bf16_math_path).unwrap_or_default();
        let guarded = content.replace(
            "#if __METAL_VERSION__ < 310000",
            "#if defined(__HAVE_BFLOAT__) && __METAL_VERSION__ < 310000",
        );
        if content != guarded {
            std::fs::write(&bf16_math_path, &guarded).unwrap();
            eprintln!("Patched bf16_math.h for macOS 26+ compatibility (half guard)");
        } else {
            // Already patched or content unchanged — ok
        }
    }

    // Patch utils.h: guard instantiate_float_limit(bfloat16_t) on macOS 26+
    let utils_h_path = build_dir.join("_deps/mlx-src/mlx/backend/metal/kernels/utils.h");
    if utils_h_path.exists() {
        let content = std::fs::read_to_string(&utils_h_path).unwrap_or_default();
        let guarded = content.replace(
            "instantiate_float_limit(bfloat16_t);\n",
            "#if defined(__HAVE_BFLOAT__)\ninstantiate_float_limit(bfloat16_t);\n#endif\n",
        );
        let guarded = guarded.replace(
            "instantiate_arg_reduce(bfloat16, bfloat16_t)",
            "#if defined(__HAVE_BFLOAT__)\ninstantiate_arg_reduce(bfloat16, bfloat16_t)\n#endif",
        );
        if content != guarded {
            std::fs::write(&utils_h_path, &guarded).unwrap();
            eprintln!("Patched utils.h for macOS 26+ compatibility (bfloat16_t guards)");
        }
    }

    // Recursively guard every unguarded bfloat16_t template instantiation in all
    // .metal files.  When bfloat16_t == half (no native Metal bfloat), these
    // produce duplicate template instantiations with float16/half variants.
    // Wrapping in `#if defined(__HAVE_BFLOAT__)` / `#endif` suppresses them;
    // the half/float16_t instantiations cover all needed precision.
    let metal_kernels = build_dir.join("_deps/mlx-src/mlx/backend/metal/kernels");
    fn guard_bfloat16_kernels(dir: &std::path::Path, metal_kernels: &std::path::Path) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    guard_bfloat16_kernels(&path, metal_kernels);
                } else if path.extension().map_or(false, |e| e == "metal") {
                    let Ok(content) = std::fs::read_to_string(&path) else { continue };
                    let mut lines: Vec<&str> = content.lines().collect();
                    let mut modified = false;
                    let mut i = 0;
                    while i < lines.len() {
                        let line = lines[i];
                        // Only consider lines that call an instantiate* macro
                        // with bfloat16 as a type argument and aren't already commented.
                        if line.contains("bfloat16")
                            && (line.contains("instantiate") || line.contains("instantiate_kernel"))
                            && !line.trim().starts_with("//")
                        {
                            // Walk backwards past blank/comment lines to find the
                            // nearest non-blank, non-comment line.
                            let mut already_guarded = false;
                            for j in (0..i).rev() {
                                let prev = lines[j].trim();
                                if prev.is_empty() || prev.starts_with("//") {
                                    continue;
                                }
                                if prev.contains("__HAVE_BFLOAT__") {
                                    already_guarded = true;
                                }
                                break;
                            }
                            if !already_guarded {
                                lines.insert(i, "  #if defined(__HAVE_BFLOAT__)");
                                lines.insert(i + 2, "  #endif");
                                i += 2;
                                modified = true;
                                let name = path.strip_prefix(metal_kernels)
                                    .unwrap_or(&path)
                                    .display();
                                eprintln!(
                                    "Patched {} line {} — guarded bfloat16 instantiation",
                                    name, i
                                );
                            }
                        }
                        i += 1;
                    }
                    if modified {
                        let new_content = lines.join("\n");
                        let _ = std::fs::write(&path, &new_content);
                    }
                }
            }
        }
    }
    guard_bfloat16_kernels(&metal_kernels, &metal_kernels);

    // Patch device.cpp: macOS 26 SDK removed nullptr terminator from NS::Dictionary::dictionary
    let device_cpp = build_dir.join("_deps/mlx-src/mlx/backend/metal/device.cpp");
    if device_cpp.exists() {
        let content = std::fs::read_to_string(&device_cpp).unwrap_or_default();
        if !content.contains("// macOS 26: NS::Dictionary") {
            let guarded = content.replace(
                "NS::Dictionary::dictionary(macro_key, macro_val, nullptr)",
                "// macOS 26: NS::Dictionary::dictionary no longer takes nullptr terminator\nNS::Dictionary::dictionary(macro_key, macro_val)",
            );
            if content != guarded {
                std::fs::write(&device_cpp, &guarded).unwrap();
                eprintln!("Patched device.cpp for macOS 26+ NS::Dictionary API");
            }
        }
    }

    // Step 2: build (make)
    let status = Command::new("cmake")
        .args(["--build", build_dir.to_str().unwrap()])
        .args(["-j", "8"])
        .args(["--target", "install"])
        .status()
        .expect("failed to run cmake --build");
    if !status.success() {
        panic!("cmake --build failed");
    }

    println!("cargo:rustc-link-search=native={}/lib", install_prefix.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-lib=lapack");
        println!("cargo:rustc-link-lib=lapacke");
    } else {
        println!("cargo:rustc-link-lib=c++");
    }
    println!("cargo:rustc-link-lib=dylib=objc");
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" || std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "ios" {
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    #[cfg(feature = "metal")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" || std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "ios" {
            println!("cargo:rustc-link-lib=framework=Metal");
        }
    }

    #[cfg(feature = "accelerate")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" || std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "ios" {
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
    }
}

fn main() {
    #[cfg(not(feature = "stub"))]
    {
        build_and_link_mlx_c();

        // generate bindings
        let bindings = bindgen::Builder::default()
            .rust_target("1.73.0".parse().expect("rust-version"))
            .header("src/mlx-c/mlx/c/mlx.h")
            .header("src/mlx-c/mlx/c/linalg.h")
            .header("src/mlx-c/mlx/c/error.h")
            .header("src/mlx-c/mlx/c/transforms_impl.h")
            .clang_arg("-Isrc/mlx-c")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        // Write the bindings to the $OUT_DIR/bindings.rs file.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }

    #[cfg(feature = "stub")]
    {
        // Write a dummy bindings file so the crate compiles
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        let dummy_bindings = r#"
pub type mlx_array = *mut std::ffi::c_void;
pub type mlx_stream = *mut std::ffi::c_void;

#[repr(C)]
pub struct mlx_optional_int_ {
    pub value: i32,
    pub has_value: bool,
}

#[repr(C)]
pub struct mlx_optional_dtype_ {
    pub value: i32,
    pub has_value: bool,
}

#[no_mangle]
pub unsafe extern "C" fn mlx_get_active_memory(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_get_cache_memory(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_get_peak_memory(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_clear_cache() {}
#[no_mangle]
pub unsafe extern "C" fn mlx_set_cache_limit(_prev: *mut usize, _limit: usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_get_memory_limit(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_set_memory_limit(_prev: *mut usize, _limit: usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_metal_is_available(_res: *mut bool) -> i32 { 0 }
#[no_mangle]
pub unsafe extern "C" fn mlx_reshape_ffi(_x: mlx_array, _shape_ar: *const i32, _ndim: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_transpose_ffi(_x: mlx_array, _axes: *const i32, _n_axes: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_slice_ffi(_x: mlx_array, _start: *const i32, _stop: *const i32, _stride: *const i32, _n_axes: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_concatenate_ffi(_arrays: *const mlx_array, _n_arrays: i32, _axis: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_pad_ffi(_x: mlx_array, _pad_widths: *const i32, _n_pads: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_array_new_data_managed_payload(
    _data: *const std::ffi::c_void, 
    _shape: *const i32, 
    _dim: i32, 
    _dtype: u32, 
    _payload: *mut std::ffi::c_void, 
    _dtor: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>
) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_fast_scaled_dot_product_attention(
    _res: *mut mlx_array,
    _q: mlx_array,
    _k: mlx_array,
    _v: mlx_array,
    _scale: f32,
    _mask_mode: *const std::ffi::c_char,
    _mask_arr: mlx_array,
    _sinks: mlx_array,
    _stream: mlx_stream,
) -> i32 {
    0
}
#[no_mangle]
pub unsafe extern "C" fn mlx_array_new() -> mlx_array { std::ptr::null_mut() }
"#;
        std::fs::write(out_path.join("bindings.rs"), dummy_bindings).expect("dummy bindings");
    }

    // Emit build-generated version constants
    let mlx_c_version = std::fs::read_to_string("src/mlx-c/VERSION")
        .unwrap_or_else(|_| "0.6.0".to_string())
        .trim()
        .to_string();
    println!("cargo:rustc-env=MLX_C_VERSION={}", mlx_c_version);
    println!("cargo:rustc-env=MLX_CORE_TARGET=v0.31.2");
    println!("cargo:rustc-env=MLX_SYS_VERSION=0.6.0-tribunus.1");
    println!("cargo:rustc-env=MLX_RS_BASE_COMMIT=93ed8db");
}
