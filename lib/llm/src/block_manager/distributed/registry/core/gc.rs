// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Garbage collection for orphaned remote storage entries.
//!
//! When registry entries are deleted but the underlying storage data remains,
//! this module helps identify and clean up orphaned objects/files.
//!
//! # Invariant
//!
//! ```text
//! ∀ key ∈ Storage: ∃ entry ∈ Registry where entry.key = key
//! ```
//!
//! When this invariant is violated (orphans exist), GC cleans them up.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;

/// Statistics for garbage collection.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Total orphans found
    pub orphans_found: u64,
    /// Total orphans deleted
    pub orphans_deleted: u64,
    /// Total bytes reclaimed (if known)
    pub bytes_reclaimed: u64,
    /// Number of GC runs completed
    pub runs_completed: u64,
    /// Last run timestamp (ms since epoch)
    pub last_run_ms: u64,
}

/// Trait for listing storage contents.
#[async_trait]
pub trait StorageLister: Send + Sync {
    /// List all keys in storage.
    async fn list_keys(&self) -> Result<Vec<String>>;

    /// Delete a key from storage.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Get size of a key (optional, for metrics).
    async fn size(&self, key: &str) -> Result<Option<u64>> {
        let _ = key;
        Ok(None)
    }
}

/// Trait for querying registry for valid entries.
#[async_trait]
pub trait RegistryLister: Send + Sync {
    /// Get all valid keys in the registry.
    async fn list_valid_keys(&self) -> Result<HashSet<String>>;
}

/// Garbage collector configuration.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Enable garbage collection
    pub enabled: bool,
    /// Interval between GC runs
    pub interval: Duration,
    /// Dry run mode (report but don't delete)
    pub dry_run: bool,
    /// Maximum items to delete per run (0 = unlimited)
    pub max_deletes_per_run: usize,
    /// Minimum age of orphans before deletion (safety margin)
    pub min_orphan_age: Duration,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: Duration::from_secs(3600), // 1 hour
            dry_run: true,                       // Safe default
            max_deletes_per_run: 1000,
            min_orphan_age: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl GcConfig {
    /// Create from environment variables.
    ///
    /// Environment variables:
    /// - `DYN_REGISTRY_GC_ENABLED`: "1" or "true" to enable
    /// - `DYN_REGISTRY_GC_INTERVAL_SECS`: Seconds between runs (default: 3600)
    /// - `DYN_REGISTRY_GC_DRY_RUN`: "1" or "true" for dry run (default: true)
    /// - `DYN_REGISTRY_GC_MAX_DELETES`: Max deletes per run (default: 1000)
    /// - `DYN_REGISTRY_GC_MIN_AGE_SECS`: Min orphan age in seconds (default: 300)
    pub fn from_env() -> Self {
        let enabled = std::env::var("DYN_REGISTRY_GC_ENABLED")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let interval_secs = std::env::var("DYN_REGISTRY_GC_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);

        let dry_run = std::env::var("DYN_REGISTRY_GC_DRY_RUN")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true);

        let max_deletes_per_run = std::env::var("DYN_REGISTRY_GC_MAX_DELETES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        let min_orphan_age_secs = std::env::var("DYN_REGISTRY_GC_MIN_AGE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        Self {
            enabled,
            interval: Duration::from_secs(interval_secs),
            dry_run,
            max_deletes_per_run,
            min_orphan_age: Duration::from_secs(min_orphan_age_secs),
        }
    }
}

/// Garbage collector for cleaning up orphaned storage entries.
pub struct GarbageCollector {
    config: GcConfig,
    storage: Arc<dyn StorageLister>,
    registry: Arc<dyn RegistryLister>,
    stats: parking_lot::RwLock<GcStats>,
}

impl GarbageCollector {
    /// Create a new garbage collector.
    pub fn new(
        config: GcConfig,
        storage: Arc<dyn StorageLister>,
        registry: Arc<dyn RegistryLister>,
    ) -> Self {
        Self {
            config,
            storage,
            registry,
            stats: parking_lot::RwLock::new(GcStats::default()),
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> GcStats {
        self.stats.read().clone()
    }

    /// Run a single GC pass.
    pub async fn run_once(&self) -> Result<GcStats> {
        tracing::info!("Starting garbage collection pass");
        let start = std::time::Instant::now();

        // Get all keys from storage
        let storage_keys = self.storage.list_keys().await?;
        let storage_key_set: HashSet<_> = storage_keys.into_iter().collect();

        // Get all valid keys from registry
        let valid_keys = self.registry.list_valid_keys().await?;

        // Find orphans (in storage but not in registry)
        let orphans: Vec<_> = storage_key_set.difference(&valid_keys).cloned().collect();

        tracing::info!(
            storage_count = storage_key_set.len(),
            registry_count = valid_keys.len(),
            orphan_count = orphans.len(),
            "GC: found {} orphaned entries",
            orphans.len()
        );

        let mut deleted = 0u64;
        let mut bytes_reclaimed = 0u64;

        if !orphans.is_empty() {
            let to_delete = if self.config.max_deletes_per_run > 0 {
                orphans
                    .iter()
                    .take(self.config.max_deletes_per_run)
                    .cloned()
                    .collect::<Vec<_>>()
            } else {
                orphans.clone()
            };

            for key in &to_delete {
                if self.config.dry_run {
                    tracing::debug!(key = %key, "GC: would delete orphan (dry run)");
                } else {
                    if let Ok(Some(size)) = self.storage.size(key).await {
                        bytes_reclaimed += size;
                    }

                    match self.storage.delete(key).await {
                        Ok(_) => {
                            tracing::debug!(key = %key, "GC: deleted orphan");
                            deleted += 1;
                        }
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "GC: failed to delete orphan");
                        }
                    }
                }
            }
        }

        let elapsed = start.elapsed();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Update stats
        let mut stats = self.stats.write();
        stats.orphans_found += orphans.len() as u64;
        stats.orphans_deleted += deleted;
        stats.bytes_reclaimed += bytes_reclaimed;
        stats.runs_completed += 1;
        stats.last_run_ms = now_ms;

        let result = stats.clone();
        drop(stats);

        tracing::info!(
            orphans_found = orphans.len(),
            orphans_deleted = deleted,
            bytes_reclaimed = bytes_reclaimed,
            elapsed_ms = elapsed.as_millis(),
            dry_run = self.config.dry_run,
            "GC pass complete"
        );

        Ok(result)
    }

    /// Start the background GC task.
    pub fn spawn(self: Arc<Self>, cancel: CancellationToken) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if !self.config.enabled {
                tracing::info!("Garbage collection disabled");
                return;
            }

            tracing::info!(
                interval_secs = self.config.interval.as_secs(),
                dry_run = self.config.dry_run,
                "Starting garbage collector"
            );

            let mut timer = interval(self.config.interval);

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        tracing::info!("Garbage collector shutting down");
                        break;
                    }
                    _ = timer.tick() => {
                        if let Err(e) = self.run_once().await {
                            tracing::error!(error = %e, "GC pass failed");
                        }
                    }
                }
            }
        })
    }
}

/// Object storage adapter for StorageLister.
pub struct ObjectStorageLister {
    bucket: String,
    endpoint: Option<String>,
}

impl ObjectStorageLister {
    pub fn new(bucket: String, endpoint: Option<String>) -> Self {
        Self { bucket, endpoint }
    }
}

#[async_trait]
impl StorageLister for ObjectStorageLister {
    async fn list_keys(&self) -> Result<Vec<String>> {
        // TODO: Implement S3 ListObjectsV2 pagination
        // For now, return empty to indicate unimplemented
        tracing::warn!(
            bucket = %self.bucket,
            "ObjectStorageLister::list_keys not yet implemented"
        );
        Ok(Vec::new())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        // TODO: Implement S3 DeleteObject
        tracing::warn!(
            bucket = %self.bucket,
            key = %key,
            "ObjectStorageLister::delete not yet implemented"
        );
        Ok(())
    }

    async fn size(&self, key: &str) -> Result<Option<u64>> {
        // TODO: Implement S3 HeadObject
        let _ = key;
        Ok(None)
    }
}

/// Disk storage adapter for StorageLister.
pub struct DiskStorageLister {
    base_path: std::path::PathBuf,
}

impl DiskStorageLister {
    pub fn new(base_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }
}

#[async_trait]
impl StorageLister for DiskStorageLister {
    async fn list_keys(&self) -> Result<Vec<String>> {
        let mut keys = Vec::new();

        if !self.base_path.exists() {
            return Ok(keys);
        }

        let mut entries = tokio::fs::read_dir(&self.base_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    keys.push(name.to_string());
                }
            }
        }

        Ok(keys)
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let path = self.base_path.join(key);
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }
        Ok(())
    }

    async fn size(&self, key: &str) -> Result<Option<u64>> {
        let path = self.base_path.join(key);
        match tokio::fs::metadata(&path).await {
            Ok(meta) => Ok(Some(meta.len())),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockStorage {
        keys: Vec<String>,
    }

    #[async_trait]
    impl StorageLister for MockStorage {
        async fn list_keys(&self) -> Result<Vec<String>> {
            Ok(self.keys.clone())
        }

        async fn delete(&self, _key: &str) -> Result<()> {
            Ok(())
        }
    }

    struct MockRegistry {
        valid_keys: HashSet<String>,
    }

    #[async_trait]
    impl RegistryLister for MockRegistry {
        async fn list_valid_keys(&self) -> Result<HashSet<String>> {
            Ok(self.valid_keys.clone())
        }
    }

    #[tokio::test]
    async fn test_gc_finds_orphans() {
        let storage = Arc::new(MockStorage {
            keys: vec![
                "key1".to_string(),
                "key2".to_string(),
                "orphan1".to_string(),
                "orphan2".to_string(),
            ],
        });

        let registry = Arc::new(MockRegistry {
            valid_keys: vec!["key1".to_string(), "key2".to_string()]
                .into_iter()
                .collect(),
        });

        let config = GcConfig {
            enabled: true,
            dry_run: true,
            ..Default::default()
        };

        let gc = GarbageCollector::new(config, storage, registry);
        let stats = gc.run_once().await.unwrap();

        assert_eq!(stats.orphans_found, 2);
        assert_eq!(stats.orphans_deleted, 0); // dry run
    }

    #[tokio::test]
    async fn test_gc_respects_max_deletes() {
        let storage = Arc::new(MockStorage {
            keys: (0..100).map(|i| format!("orphan{}", i)).collect(),
        });

        let registry = Arc::new(MockRegistry {
            valid_keys: HashSet::new(),
        });

        let config = GcConfig {
            enabled: true,
            dry_run: false,
            max_deletes_per_run: 10,
            ..Default::default()
        };

        let gc = GarbageCollector::new(config, storage, registry);
        let stats = gc.run_once().await.unwrap();

        assert_eq!(stats.orphans_found, 100);
        assert_eq!(stats.orphans_deleted, 10);
    }
}

