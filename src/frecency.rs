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

// Time unit constants
const SECONDS_PER_HOUR: u64 = 60 * 60;
const SECONDS_PER_DAY: u64 = 24 * SECONDS_PER_HOUR;

// Default frecency parameters
const DEFAULT_HALF_LIFE_DAYS: u64 = 1;
const DEFAULT_MOMENTUM_WINDOW_HOURS: u64 = 6;
const DEFAULT_MOMENTUM_MAX_BOOST: f32 = 0.5;

// Frecency algorithm constants
const HALF_LIFE_DECAY_BASE: f32 = 0.5;
const FREQUENCY_SMOOTHING: f32 = 1.0;
const MOMENTUM_BASELINE: f32 = 1.0;

/// Tracks usage statistics for a single item.
///
/// Timestamps use u32 Unix seconds (valid until year 2106).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FrecencyEntry {
    pub frequency: u32,
    pub first_access: u32,
    pub last_access: u32,
    pub prev_access: u32,
}

impl FrecencyEntry {
    #[inline]
    fn now_to_secs(time: SystemTime) -> u32 {
        // Safe truncation: values fit in u32 until year 2106
        time.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32
    }

    #[inline]
    fn secs_to_time(secs: u32) -> Option<SystemTime> {
        if secs == 0 {
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
    pub momentum_max_boost: f32,
}

impl Default for FrecencyConfig {
    fn default() -> Self {
        Self {
            half_life: Duration::from_secs(DEFAULT_HALF_LIFE_DAYS * SECONDS_PER_DAY),
            momentum_window: Duration::from_secs(DEFAULT_MOMENTUM_WINDOW_HOURS * SECONDS_PER_HOUR),
            momentum_max_boost: DEFAULT_MOMENTUM_MAX_BOOST,
        }
    }
}

/// Individual score components returned for debugging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrecencyComponents {
    pub raw: f32,
    pub frequency_component: f32,
    pub decay_component: f32,
    pub momentum_component: f32,
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
    // Cached f32 conversions to avoid repeated Duration -> f32 conversion
    half_life_secs: f32,
    window_secs: f32,
}

impl FrecencyInner {
    /// Computes the frecency score for an entry using a u32 timestamp.
    ///
    /// The frecency algorithm combines three components:
    /// - **Frequency**: log2(frequency + 1) - rewards repeated use, logarithmic to prevent dominance
    /// - **Recency (decay)**: 0.5^(age/half_life) - exponential decay based on time since last access
    /// - **Momentum**: 1.0 + boost*(1 - Δ/window) - rewards items used in rapid succession
    ///
    /// Final score = frequency × decay × momentum
    ///
    /// The frequency component adds 1 before taking log2 to:
    /// 1. Avoid log(0) for new items
    /// 2. Smooth the logarithmic curve for small frequencies
    ///
    /// The decay component uses base 0.5 (half-life decay) where the score halves
    /// every `half_life` duration since last access.
    ///
    /// The momentum component starts at baseline 1.0 (no boost). If the time between
    /// the last two accesses is within the momentum window, a boost is applied that
    /// decreases linearly as the gap approaches the window size.
    fn score_entry_with_timestamp(
        &self,
        entry: &FrecencyEntry,
        now_secs: u32,
    ) -> FrecencyComponents {
        if entry.frequency == 0 || entry.last_access == 0 {
            return FrecencyComponents {
                raw: 0.0,
                frequency_component: 0.0,
                decay_component: 0.0,
                momentum_component: 0.0,
            };
        }

        // Frequency component: log2(freq + 1)
        // Adding FREQUENCY_SMOOTHING (1.0) avoids log(0) and smooths small values
        let freq_component = ((entry.frequency as f32) + FREQUENCY_SMOOTHING).log2();

        // Recency decay: 0.5^(elapsed / half_life)
        // Uses HALF_LIFE_DECAY_BASE (0.5) for exponential decay
        // Use saturating_sub to handle potential clock skew
        let elapsed = now_secs.saturating_sub(entry.last_access) as f32;
        let half_life = self.half_life_secs.max(f32::EPSILON);
        let decay_component = HALF_LIFE_DECAY_BASE.powf(elapsed / half_life);

        // Momentum component: starts at MOMENTUM_BASELINE (1.0)
        // Adds boost if time between last two accesses is within momentum window
        let mut momentum_component = MOMENTUM_BASELINE;
        if entry.prev_access > 0 {
            // Use saturating_sub to prevent underflow
            let delta = entry.last_access.saturating_sub(entry.prev_access) as f32;
            let window = self.window_secs.max(f32::EPSILON);
            if delta < window {
                // Linear decay: boost decreases from max to 0 as delta approaches window
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

    /// Computes the frecency score for an entry using a SystemTime.
    fn score_entry(&self, entry: &FrecencyEntry, now: SystemTime) -> FrecencyComponents {
        let now_secs = FrecencyEntry::now_to_secs(now);
        self.score_entry_with_timestamp(entry, now_secs)
    }

    fn load_from_path(path: &Path, config: FrecencyConfig) -> io::Result<Self> {
        let half_life_secs = config.half_life.as_secs_f32();
        let window_secs = config.momentum_window.as_secs_f32();

        if !path.exists() {
            return Ok(Self {
                config,
                path: Some(path.to_path_buf()),
                entries: RwLock::new(HashMap::new()),
                dirty: AtomicBool::new(false),
                half_life_secs,
                window_secs,
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
            half_life_secs,
            window_secs,
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
                half_life_secs: config.half_life.as_secs_f32(),
                window_secs: config.momentum_window.as_secs_f32(),
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
    ///
    /// This clones the entire HashMap including all keys and values.
    /// For large datasets with frequent access, consider using `get()` for
    /// individual lookups instead.
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
        // Check if entry exists first to avoid allocating for existing keys
        if let Some(entry) = entries.get_mut(key) {
            entry.update_timestamps(now);
        } else {
            // Only allocate String for new keys
            let mut entry = FrecencyEntry::default();
            entry.update_timestamps(now);
            entries.insert(key.to_string(), entry);
        }
        self.inner.dirty.store(true, Ordering::SeqCst);
    }

    /// Returns the raw frecency score for the provided key.
    pub fn score_for(&self, key: &str, now: SystemTime) -> f32 {
        let entry = {
            let entries = self.inner.entries.read();
            match entries.get(key) {
                Some(entry) => *entry,
                None => return 0.0,
            }
        }; // Lock released here
        self.inner.score_entry(&entry, now).raw
    }

    /// Returns the raw frecency score for the provided key using a u32 timestamp.
    ///
    /// This is more efficient when scoring multiple items with the same timestamp,
    /// as it avoids repeated SystemTime to u32 conversions.
    pub fn score_for_timestamp(&self, key: &str, now_secs: u32) -> f32 {
        let entry = {
            let entries = self.inner.entries.read();
            match entries.get(key) {
                Some(entry) => *entry,
                None => return 0.0,
            }
        }; // Lock released here
        self.inner.score_entry_with_timestamp(&entry, now_secs).raw
    }

    /// Returns detailed score components. Useful for debugging.
    pub fn score_components(&self, key: &str, now: SystemTime) -> Option<FrecencyComponents> {
        let entry = {
            let entries = self.inner.entries.read();
            *entries.get(key)?
        }; // Lock released here
        Some(self.inner.score_entry(&entry, now))
    }
}
