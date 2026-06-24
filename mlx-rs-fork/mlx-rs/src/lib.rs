//! Unofficial rust bindings for the [MLX
//! framework](https://github.com/ml-explore/mlx).

#![deny(unused_unsafe, missing_debug_implementations, missing_docs)]
#![cfg_attr(test, allow(clippy::approx_constant))]

#[macro_use]
/// Macros for mlx-rs
pub mod macros;

#[cfg(not(feature = "stub"))]
mod array;
#[cfg(not(feature = "stub"))]
pub mod builder;
#[cfg(not(feature = "stub"))]
mod device;
#[cfg(not(feature = "stub"))]
mod dtype;
#[cfg(not(feature = "stub"))]
pub mod error;
#[cfg(not(feature = "stub"))]
pub mod fast;
#[cfg(not(feature = "stub"))]
pub mod fft;
#[cfg(not(feature = "stub"))]
pub mod linalg;
#[cfg(not(feature = "stub"))]
pub mod losses;
#[cfg(not(feature = "stub"))]
pub mod module;
#[cfg(not(feature = "stub"))]
pub mod nested;
#[cfg(not(feature = "stub"))]
pub mod nn;
#[cfg(not(feature = "stub"))]
pub mod ops;
#[cfg(not(feature = "stub"))]
pub mod optimizers;
#[cfg(not(feature = "stub"))]
pub mod quantization;
#[cfg(not(feature = "stub"))]
pub mod random;
#[cfg(not(feature = "stub"))]
/// Stream types for MLX operations.
pub mod stream;
#[cfg(not(feature = "stub"))]
pub mod transforms;
#[cfg(not(feature = "stub"))]
pub mod utils;
#[cfg(not(feature = "stub"))]
/// Backend foundation module (evidence, capabilities, reference ops).
pub mod backend;

#[cfg(feature = "stub")]
/// Dummy types for stub mode
pub mod stub {
    //! Dummy types for stub mode
    use std::marker::PhantomData;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// Dummy Dtype
    pub enum Dtype {
        /// Bool
        Bool,
        /// Uint8
        Uint8,
        /// Uint16
        Uint16,
        /// Uint32
        Uint32,
        /// Uint64
        Uint64,
        /// Int8
        Int8,
        /// Int16
        Int16,
        /// Int32
        Int32,
        /// Int64
        Int64,
        /// Float16
        Float16,
        /// Float32
        Float32,
        /// Float64
        Float64,
        /// Bfloat16
        Bfloat16,
        /// Complex64
        Complex64,
    }

    #[derive(Debug, Clone, PartialEq)]
    /// Dummy Array
    pub struct Array {
        _marker: PhantomData<()>,
    }

    impl AsRef<Array> for Array {
        fn as_ref(&self) -> &Array { self }
    }

    /// Helper for anything that converts to Array
    pub trait IntoArray {
        /// Convert to array
        fn into_array(self) -> Array;
    }

    impl IntoArray for f32 {
        fn into_array(self) -> Array { Array { _marker: PhantomData } }
    }

    impl IntoArray for Array {
        fn into_array(self) -> Array { self }
    }

    impl IntoArray for &Array {
        fn into_array(self) -> Array { Array { _marker: PhantomData } }
    }

    impl IntoArray for Option<&Array> {
        fn into_array(self) -> Array { Array { _marker: PhantomData } }
    }

    impl Array {
        /// Create from slice
        pub fn from_slice<T>(_data: &[T], _shape: &[i32]) -> Self {
            Self { _marker: PhantomData }
        }
        /// Create from f32
        pub fn from_f32(_val: f32) -> Self {
            Self { _marker: PhantomData }
        }
        /// Create ones
        pub fn ones<T>(_shape: &[i32]) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// Create full
        pub fn full<T>(_shape: &[i32], _val: impl IntoArray) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// Get shape
        pub fn shape(&self) -> &[i32] {
            &[]
        }
        /// Try as slice
        pub fn try_as_slice<T>(&self) -> MlxResult<&[T]> {
            Err(error::Exception::custom("stub backend"))
        }
        /// As slice
        pub fn as_slice<T>(&self) -> &[T] {
            &[]
        }
        /// Multiply
        pub fn multiply(&self, _other: &Self) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// Add
        pub fn add(&self, _other: &Self) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// Subtract
        pub fn subtract(&self, _other: &Self) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// Divide
        pub fn divide(&self, _other: &Self) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// Matmul
        pub fn matmul(&self, _other: &Self) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// nbytes
        pub fn nbytes(&self) -> usize {
            0
        }
        /// as_dtype
        pub fn as_dtype(&self, _dtype: Dtype) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// eval
        pub fn eval(&self) -> MlxResult<()> {
            Ok(())
        }
        /// strides
        pub fn strides(&self) -> &[usize] {
            &[]
        }
        /// dtype
        pub fn dtype(&self) -> Dtype {
            Dtype::Float32
        }
        /// ndim
        pub fn ndim(&self) -> usize {
            0
        }
        /// reshape
        pub fn reshape(&self, _shape: &[i32]) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// transpose
        pub fn transpose(&self) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// from_raw_data
        pub unsafe fn from_raw_data(_data: *const std::ffi::c_void, _shape: &[i32], _dtype: Dtype) -> Self {
            Self { _marker: PhantomData }
        }
        /// size
        pub fn size(&self) -> usize {
            0
        }
        /// index
        pub fn index<T>(&self, _indices: T) -> Self {
            Self { _marker: PhantomData }
        }
        /// from_ptr
        pub unsafe fn from_ptr(_ptr: mlx_sys::mlx_array) -> Self {
            Self { _marker: PhantomData }
        }
        /// mean_axes
        pub fn mean_axes(&self, _axes: &[i32], _keep_dims: bool) -> MlxResult<Self> {
            Ok(Self { _marker: PhantomData })
        }
        /// index_mut
        pub fn index_mut<T>(&mut self, _indices: T, _val: &Self) {
            // no-op
        }
        /// as_ptr
        pub fn as_ptr(&self) -> mlx_sys::mlx_array {
            std::ptr::null_mut()
        }
        /// item
        pub fn item<T: Default>(&self) -> T {
            T::default()
        }
    }

    /// Dummy MlxResult
    pub type MlxResult<T, E = error::Exception> = Result<T, E>;

    /// Dummy error module
    pub mod error {
        /// Dummy Result
        pub type Result<T, E = Exception> = std::result::Result<T, E>;
        /// Dummy Exception
        #[derive(Debug, Clone)]
        pub struct Exception(pub String);
        impl Exception {
            /// Custom error
            pub fn custom(m: impl Into<String>) -> Self { Self(m.into()) }
        }
        impl std::fmt::Display for Exception {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl std::error::Error for Exception {}

        impl From<Exception> for String {
            fn from(e: Exception) -> Self { e.0 }
        }
    }

    /// Dummy ops module
    pub mod ops {
        use super::{Array, MlxResult};
        /// Dummy indexing module
        pub mod indexing {
            use super::{Array, MlxResult};
            /// Dummy IndexOp trait
            pub trait IndexOp {}
            impl IndexOp for Array {}
            /// Dummy IndexMutOp trait
            pub trait IndexMutOp {}
            impl IndexMutOp for Array {}
            /// Dummy take_along_axis
            pub fn take_along_axis(_a: &Array, _indices: &Array, _axis: i32) -> MlxResult<Array> {
                Ok(Array { _marker: std::marker::PhantomData })
            }
            /// Dummy argmax_axis
            pub fn argmax_axis(_a: &Array, _axis: i32, _keep_dims: bool) -> MlxResult<Array> {
                Ok(Array { _marker: std::marker::PhantomData })
            }
            /// Dummy take_axis
            pub fn take_axis(_a: &Array, _indices: &Array, _axis: i32) -> MlxResult<Array> {
                Ok(Array { _marker: std::marker::PhantomData })
            }
        }
        /// Dummy quantized_matmul
        pub fn quantized_matmul(_x: &Array, _w: &Array, _s: &Array, _b: &Array, _transpose: bool, _group_size: i32, _bits: i32) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy transpose_axes
        pub fn transpose_axes(_a: &Array, _axes: &[i32]) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy softmax_axes
        pub fn softmax_axes(_a: &Array, _axes: &[i32], _precise: impl Into<Option<bool>>) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy tile
        pub fn tile(_a: &Array, _reps: &[i32]) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy sigmoid
        pub fn sigmoid(_a: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy reshape
        pub fn reshape(_a: &Array, _shape: &[i32]) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy add
        pub fn add(_a: &Array, _b: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy multiply
        pub fn multiply(_a: &Array, _b: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy tanh
        pub fn tanh(_a: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy rsqrt
        pub fn rsqrt(_a: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy cos
        pub fn cos(_a: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy sin
        pub fn sin(_a: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy concatenate
        pub fn concatenate(_arrays: &[&Array]) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy concatenate_axis
        pub fn concatenate_axis(_arrays: &[&Array], _axis: i32) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy zeros
        pub fn zeros<T>(_shape: &[i32]) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy mean_axes
        pub fn mean_axes(_a: &Array, _axes: &[i32], _keep_dims: bool) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy dequantize
        pub fn dequantize(_x: &Array, _s: &Array, _b: &Array, _group_size: i32, _bits: i32) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy transpose
        pub fn transpose(_a: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy matmul
        pub fn matmul(_a: &Array, _b: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
    }

    /// Dummy fast module
    pub mod fast {
        use super::{Array, MlxResult};
        /// Dummy rms_norm
        pub fn rms_norm(_x: &Array, _w: &Array, _eps: f32) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy rope
        pub fn rope(_a: &Array, _dims: i32, _traditional: bool, _base: Option<f32>, _scale: f32, _offset: i32, _freqs: Option<&Array>) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
    }

    /// Dummy nn module
    pub mod nn {
        use super::{Array, MlxResult};
        /// Dummy silu
        pub fn silu(_x: &Array) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
    }

    /// Helper for optional stream
    pub trait OptionalStream {
        /// Convert to option
        fn as_option(&self) -> Option<&Stream>;
    }

    /// Dummy random module
    pub mod random {
        use super::{Array, MlxResult, OptionalStream};
        /// Dummy key
        pub fn key(_seed: u64) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
        /// Dummy categorical
        pub fn categorical(_logits: &Array, _shape: Option<&[i32]>, _num_samples: Option<i32>, _stream: impl OptionalStream) -> MlxResult<Array> {
            Ok(Array { _marker: std::marker::PhantomData })
        }
    }

    /// Dummy Device
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Device;
    impl Device {
        /// Try default
        pub fn try_default() -> Result<Self, String> {
            Ok(Self)
        }
        /// GPU
        pub fn gpu() -> Self {
            Self
        }
        /// CPU
        pub fn cpu() -> Self {
            Self
        }
    }

    impl std::fmt::Display for Device {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "stub")
        }
    }

    /// Dummy Stream
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Stream;

    impl Stream {
        /// New
        pub fn new() -> Self { Self }
        /// as_ptr
        pub fn as_ptr(&self) -> *mut std::ffi::c_void {
            std::ptr::null_mut()
        }
    }

    impl OptionalStream for &Array {
        fn as_option(&self) -> Option<&Stream> { None }
    }

    impl OptionalStream for Option<&Stream> {
        fn as_option(&self) -> Option<&Stream> { *self }
    }

    impl OptionalStream for Option<&Array> {
        fn as_option(&self) -> Option<&Stream> { None }
    }

    impl OptionalStream for &Option<Array> {
        fn as_option(&self) -> Option<&Stream> { None }
    }

    impl OptionalStream for &Option<&Array> {
        fn as_option(&self) -> Option<&Stream> { None }
    }

    /// Dummy transforms module
    pub mod transforms {
        /// Dummy eval
        pub fn eval(_args: impl IntoIterator<Item = super::Array>) -> super::MlxResult<()> {
            Ok(())
        }
    }
}

#[cfg(feature = "stub")]
pub use stub::*;

#[cfg(not(feature = "stub"))]
pub use array::*;
#[cfg(not(feature = "stub"))]
pub use device::*;
#[cfg(not(feature = "stub"))]
pub use dtype::*;
#[cfg(not(feature = "stub"))]
pub use stream::*;

pub(crate) mod constants {
    /// The default length of the stack-allocated vector in `SmallVec<[T; DEFAULT_STACK_VEC_LEN]>`
    pub(crate) const DEFAULT_STACK_VEC_LEN: usize = 4;
}

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}

    impl Sealed for () {}

    impl<A> Sealed for (A,) where A: Sealed {}
    impl<A, B> Sealed for (A, B)
    where
        A: Sealed,
        B: Sealed,
    {
    }
    impl<A, B, C> Sealed for (A, B, C)
    where
        A: Sealed,
        B: Sealed,
        C: Sealed,
    {
    }
}
