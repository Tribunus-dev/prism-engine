#[cfg(feature = "storage-adapters")]
pub mod duckdb;
#[cfg(feature = "storage-adapters")]
pub mod pg;
#[cfg(feature = "storage-adapters")]
pub mod valkey;

#[cfg(feature = "storage-adapters")]
pub use duckdb::DuckDbAdapter;
#[cfg(feature = "storage-adapters")]
pub use pg::PgAdapter;
#[cfg(feature = "storage-adapters")]
pub use valkey::ValkeyAdapter;
