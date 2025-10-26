use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Tracks usage statistics for a single item.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FrecencyEntry {
    pub frequency: u32,
    pub first_access: i64,
    pub last_access: i64,
    pub prev_access: i64,
}

impl FrecencyEntry {
    #[inline]
    fn now_to_secs(time: SystemTime) -> i64 {
        time.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    #[inline]
    fn secs_to_time(secs: i64) -> Option<SystemTime> {
        if secs <= 0 {
            return None;
        }
        Some(UNIX_EPOCH + Duration::from_secs(secs as u64))
    }

    #[inline]
    fn update_timestamps(&mut self, now: SystemTime) {
        let now_secs = Self::now_to_secs(now);
        if self.frequency == 0 {
            self.first_access = now_secs;
            self.prev_access = now_secs;
        } else if self.last_access > 0 {
            self.prev_access = self.last_access;
        }
        self.last_access = now_secs;
        self.frequency = self.frequency.saturating_add(1);
    }

    /// Returns the first time the item was seen.
    pub fn first_access(&self) -> Option<SystemTime> {
        Self::secs_to_time(self.first_access)
    }

    /// Returns when the item was most recently seen.
    pub fn last_access(&self) -> Option<SystemTime> {
        Self::secs_to_time(self.last_access)
    }

    /// Returns when the item was seen before the latest access.
    pub fn prev_access(&self) -> Option<SystemTime> {
        Self::secs_to_time(self.prev_access)
    }
}

/// Configuration for the frecency scoring algorithm.
#[derive(Debug, Clone, Copy)]
pub struct FrecencyConfig {
    pub half_life: Duration,
    pub momentum_window: Duration,
    pub momentum_max_boost: f64,
}

impl Default for FrecencyConfig {
    fn default() -> Self {
        Self {
            half_life: Duration::from_secs(24 * 60 * 60),
            momentum_window: Duration::from_secs(6 * 60 * 60),
            momentum_max_boost: 0.5,
        }
    }
}

/// Individual score components returned for debugging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrecencyComponents {
    pub raw: f64,
    pub frequency_component: f64,
    pub decay_component: f64,
    pub momentum_component: f64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct PersistedStore {
    entries: HashMap<String, FrecencyEntry>,
}

#[derive(Debug)]
struct FrecencyInner {
    config: FrecencyConfig,
    path: Option<PathBuf>,
    entries: RwLock<HashMap<String, FrecencyEntry>>,
    dirty: AtomicBool,
}

impl FrecencyInner {
    fn score_entry(&self, entry: &FrecencyEntry, now: SystemTime) -> FrecencyComponents {
        if entry.frequency == 0 || entry.last_access <= 0 {
            return FrecencyComponents {
                raw: 0.0,
                frequency_component: 0.0,
                decay_component: 0.0,
                momentum_component: 0.0,
            };
        }

        let freq_component = ((entry.frequency as f64) + 1.0).log2();

        let now_secs = FrecencyEntry::now_to_secs(now) as f64;
        let last_secs = entry.last_access as f64;
        let elapsed = (now_secs - last_secs).max(0.0);
        let half_life = self.config.half_life.as_secs_f64().max(f64::EPSILON);
        let decay_component = 0.5_f64.powf(elapsed / half_life);

        let mut momentum_component = 1.0;
        if entry.prev_access > 0 {
            let prev_secs = entry.prev_access as f64;
            let delta = (last_secs - prev_secs).max(0.0);
            let window = self.config.momentum_window.as_secs_f64().max(f64::EPSILON);
            if delta < window {
                let ratio = delta / window;
                momentum_component += self.config.momentum_max_boost * (1.0 - ratio);
            }
        }

        let raw = freq_component * decay_component * momentum_component;
        FrecencyComponents {
            raw,
            frequency_component: freq_component,
            decay_component,
            momentum_component,
        }
    }

    fn load_from_path(path: &Path, config: FrecencyConfig) -> io::Result<Self> {
        if !path.exists() {
            return Ok(Self {
                config,
                path: Some(path.to_path_buf()),
                entries: RwLock::new(HashMap::new()),
                dirty: AtomicBool::new(false),
            });
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let persisted: PersistedStore =
            bincode::deserialize_from(reader).map_err(io::Error::other)?;
        Ok(Self {
            config,
            path: Some(path.to_path_buf()),
            entries: RwLock::new(persisted.entries),
            dirty: AtomicBool::new(false),
        })
    }

    fn persist_to_path(&self, path: &Path) -> io::Result<()> {
        if !self.dirty.load(Ordering::SeqCst) {
            return Ok(());
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let tmp_path = path.with_extension("tmp");
        let entries_snapshot = {
            let entries = self.entries.read();
            entries.clone()
        };

        let file = fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        let persisted = PersistedStore {
            entries: entries_snapshot,
        };

        if let Err(err) = bincode::serialize_into(&mut writer, &persisted) {
            let _ = fs::remove_file(&tmp_path);
            self.dirty.store(true, Ordering::SeqCst);
            return Err(io::Error::other(err));
        }

        if let Err(err) = writer.flush() {
            let _ = fs::remove_file(&tmp_path);
            self.dirty.store(true, Ordering::SeqCst);
            return Err(err);
        }

        drop(writer);

        if let Err(err) = fs::rename(&tmp_path, path) {
            let _ = fs::remove_file(&tmp_path);
            self.dirty.store(true, Ordering::SeqCst);
            return Err(err);
        }

        self.dirty.store(false, Ordering::SeqCst);
        Ok(())
    }
}

/// Thread-safe frecency store that mirrors fzf's scoring behaviour.
#[derive(Clone)]
pub struct FrecencyStore {
    inner: std::sync::Arc<FrecencyInner>,
}

impl FrecencyStore {
    /// Creates a frecency store that lives purely in memory.
    pub fn new(config: FrecencyConfig) -> Self {
        Self {
            inner: Arc::new(FrecencyInner {
                config,
                path: None,
                entries: RwLock::new(HashMap::new()),
                dirty: AtomicBool::new(false),
            }),
        }
    }

    /// Loads the frecency database from disk or creates it if it does not exist.
    pub fn load_or_default(path: impl AsRef<Path>, config: FrecencyConfig) -> io::Result<Self> {
        Ok(Self {
            inner: Arc::new(FrecencyInner::load_from_path(path.as_ref(), config)?),
        })
    }

    /// Returns the configuration of this store.
    pub fn config(&self) -> FrecencyConfig {
        self.inner.config
    }

    /// Explicitly saves the database to disk. No-op if no file path was configured.
    pub fn save(&self) -> io::Result<()> {
        match &self.inner.path {
            Some(path) => self.inner.persist_to_path(path),
            None => Ok(()),
        }
    }

    /// Returns whether the store contains unsaved updates.
    pub fn is_dirty(&self) -> bool {
        self.inner.dirty.load(Ordering::SeqCst)
    }

    /// Returns a snapshot of all entries.
    pub fn entries(&self) -> HashMap<String, FrecencyEntry> {
        self.inner.entries.read().clone()
    }

    /// Removes an entry from the database.
    pub fn remove(&self, key: &str) -> bool {
        let mut entries = self.inner.entries.write();
        let removed = entries.remove(key).is_some();
        if removed {
            self.inner.dirty.store(true, Ordering::SeqCst);
        }
        removed
    }

    /// Clears all entries from the database.
    pub fn clear(&self) {
        let mut entries = self.inner.entries.write();
        if !entries.is_empty() {
            entries.clear();
            self.inner.dirty.store(true, Ordering::SeqCst);
        }
    }

    /// Returns a copy of the entry for the given key, if it exists.
    pub fn get(&self, key: &str) -> Option<FrecencyEntry> {
        self.inner.entries.read().get(key).copied()
    }

    /// Returns the path where this store persists data, if configured.
    pub fn path(&self) -> Option<&Path> {
        self.inner.path.as_deref()
    }

    /// Increases the frequency count and updates timestamps for an item.
    pub fn update(&self, key: &str, now: SystemTime) {
        let mut entries = self.inner.entries.write();
        let entry = entries.entry(key.to_string()).or_default();
        entry.update_timestamps(now);
        self.inner.dirty.store(true, Ordering::SeqCst);
    }

    /// Returns the raw frecency score for the provided key.
    pub fn score_for(&self, key: &str, now: SystemTime) -> f64 {
        let entries = self.inner.entries.read();
        let entry = match entries.get(key) {
            Some(entry) => entry,
            None => return 0.0,
        };
        self.inner.score_entry(entry, now).raw
    }

    /// Returns detailed score components. Useful for debugging.
    pub fn score_components(&self, key: &str, now: SystemTime) -> Option<FrecencyComponents> {
        let entries = self.inner.entries.read();
        let entry = entries.get(key)?;
        Some(self.inner.score_entry(entry, now))
    }
}
