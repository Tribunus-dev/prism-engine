//! Valkey (Redis-compatible) cache wrapper for the evidence dashboard.
//!
//! Provides `DashboardCache` with typed key helpers, TTL-aware get/set, and
//! pattern-based invalidation.

use fred::clients::RedisClient;
use fred::error::RedisError;
use fred::prelude::*;

/// Thin async wrapper around the `fred` RedisClient for dashboard caching.
///
/// All operations are no-ops when `client` is `None` (e.g. Valkey is
/// unavailable or was not configured).
pub struct DashboardCache {
    pub client: Option<RedisClient>,
}

impl DashboardCache {
    /// Connect to a Valkey / Redis instance.
    ///
    /// The `url` should be a standard Redis URL such as
    /// `redis://127.0.0.1:6379` or `rediss://user:pass@host:6380`.
    ///
    /// Returns `DashboardCache` with the connected client, or a
    /// `RedisError` if the initial handshake fails.
    pub async fn connect(url: &str) -> Result<DashboardCache, RedisError> {
        let config = if url.starts_with("redis://") || url.starts_with("rediss://") {
            RedisConfig::from_url(url)?
        } else {
            // Accept bare host:port as a convenience
            RedisConfig::from_url(&format!("redis://{}", url))?
        };

        let client = RedisClient::new(config, None, None, None);
        client.init().await?;
        Ok(DashboardCache {
            client: Some(client),
        })
    }

    /// Return an empty / disabled cache (all operations are no-ops).
    pub fn disabled() -> Self {
        DashboardCache { client: None }
    }

    /// Retrieve a cached string value by key.
    ///
    /// Returns `None` if the key is absent or the client is disabled.
    pub async fn get_cached(&self, key: &str) -> Option<String> {
        let client = self.client.as_ref()?;
        client.get::<Option<String>, _>(key).await.unwrap_or(None)
    }

    /// Store a string value with a TTL (seconds).
    ///
    /// Silently no-ops when the client is disabled.
    pub async fn set_cached(&self, key: &str, value: &str, ttl_secs: u64) {
        if let Some(client) = &self.client {
            let _: Result<(), _> = client
                .set(
                    key,
                    value,
                    Some(Expiration::EX(ttl_secs as i64)),
                    None,
                    false,
                )
                .await;
        }
    }

    /// Invalidate all keys matching `pattern` using a Lua EVAL script for
    /// atomicity.
    ///
    /// This uses Redis `KEYS` internally, which is acceptable for a
    /// dashboard with low key cardinality. For production caches with
    /// millions of keys, prefer `SCAN`-based iteration.
    ///
    /// Silently no-ops when the client is disabled.
    pub async fn invalidate(&self, pattern: &str) {
        if let Some(client) = &self.client {
            let script = r#"
                local keys = redis.call('KEYS', ARGV[1])
                for _, k in ipairs(keys) do redis.call('DEL', k) end
                return #keys
            "#;
            let _: Result<(), _> = client
                .custom::<()>(
                    cmd("EVAL"),
                    vec![script, "0".to_string(), pattern.to_string()],
                )
                .await;
        }
    }
}

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

/// Cache key for the list of all cimage artifact digests.
pub fn cimage_list_key() -> &'static str {
    "dashboard:cimages:list"
}

/// Cache key for a single cimage artifact by digest.
pub fn cimage_key(digest: &str) -> String {
    format!("dashboard:cimage:{}", digest)
}

/// Cache key for admission scatter data for a given artifact digest.
pub fn scatter_key(digest: &str) -> String {
    format!("dashboard:scatter:{}", digest)
}

/// Cache key for admissions data for a given tensor key.
pub fn admission_key(tensor_key: &str) -> String {
    format!("dashboard:admissions:{}", tensor_key)
}
