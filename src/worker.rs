use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::mem::take;
use std::sync::atomic::{self, AtomicBool, AtomicU32};
use std::sync::Arc;
use std::time::SystemTime;

use nucleo_matcher::Config;
use parking_lot::Mutex;
use rayon::{prelude::*, ThreadPool};

use crate::frecency::FrecencyStore;
use crate::par_sort::par_quicksort;
use crate::pattern::{self, MultiPattern};
use crate::{boxcar, Match};

struct Matchers(Box<[UnsafeCell<nucleo_matcher::Matcher>]>);

impl Matchers {
    // this is not a true mut from ref, we use a cell here
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut nucleo_matcher::Matcher {
        &mut *self.0[rayon::current_thread_index().unwrap()].get()
    }
}

unsafe impl Sync for Matchers {}
unsafe impl Send for Matchers {}

pub(crate) struct FrecencyContext<T: Sync + Send + 'static> {
    pub(crate) store: Arc<FrecencyStore>,
    pub(crate) key_extractor: Arc<dyn Fn(&T) -> Option<String> + Send + Sync>,
}

impl<T: Sync + Send + 'static> Clone for FrecencyContext<T> {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            key_extractor: Arc::clone(&self.key_extractor),
        }
    }
}

impl<T: Sync + Send + 'static> FrecencyContext<T> {
    fn score_for(&self, item: &T, now_secs: u32) -> f32 {
        (self.key_extractor)(item)
            .map(|key| self.store.score_for_timestamp(&key, now_secs))
            .unwrap_or(0.0)
    }
}

pub(crate) struct Worker<T: Sync + Send + 'static> {
    pub(crate) running: bool,
    matchers: Matchers,
    pub(crate) matches: Vec<Match>,
    pub(crate) pattern: MultiPattern,
    pub(crate) sort_results: bool,
    pub(crate) reverse_items: bool,
    pub(crate) canceled: Arc<AtomicBool>,
    pub(crate) should_notify: Arc<AtomicBool>,
    pub(crate) was_canceled: bool,
    pub(crate) last_snapshot: u32,
    notify: Arc<(dyn Fn() + Sync + Send)>,
    pub(crate) items: Arc<boxcar::Vec<T>>,
    frecency: Option<FrecencyContext<T>>,
    in_flight: Vec<u32>,
}

impl<T: Sync + Send + 'static> Worker<T> {
    pub(crate) fn item_count(&self) -> u32 {
        self.last_snapshot - self.in_flight.len() as u32
    }
    pub(crate) fn update_config(&mut self, config: Config) {
        for matcher in self.matchers.0.iter_mut() {
            matcher.get_mut().config = config.clone();
        }
    }
    pub(crate) fn sort_results(&mut self, sort_results: bool) {
        self.sort_results = sort_results;
    }
    pub(crate) fn reverse_items(&mut self, reverse_items: bool) {
        self.reverse_items = reverse_items;
    }

    pub(crate) fn set_frecency(&mut self, frecency: Option<FrecencyContext<T>>) {
        self.frecency = frecency;
    }

    pub(crate) fn new(
        worker_threads: Option<usize>,
        config: Config,
        notify: Arc<(dyn Fn() + Sync + Send)>,
        cols: u32,
    ) -> (ThreadPool, Self) {
        let worker_threads = worker_threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |it| it.get()));
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("nucleo worker {i}"))
            .num_threads(worker_threads)
            .build()
            .expect("creating threadpool failed");
        let matchers = (0..worker_threads)
            .map(|_| UnsafeCell::new(nucleo_matcher::Matcher::new(config.clone())))
            .collect();
        let worker = Worker {
            running: false,
            matchers: Matchers(matchers),
            last_snapshot: 0,
            matches: Vec::new(),
            // just a placeholder
            pattern: MultiPattern::new(cols as usize),
            sort_results: true,
            reverse_items: false,
            canceled: Arc::new(AtomicBool::new(false)),
            should_notify: Arc::new(AtomicBool::new(false)),
            was_canceled: false,
            notify,
            items: Arc::new(boxcar::Vec::with_capacity(2 * 1024, cols)),
            frecency: None,
            in_flight: Vec::with_capacity(64),
        };
        (pool, worker)
    }

    unsafe fn process_new_items(&mut self, unmatched: &AtomicU32) {
        let matchers = &self.matchers;
        let pattern = &self.pattern;
        let frecency = self.frecency.as_ref().cloned();
        let now_secs = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;
        self.matches.reserve(self.in_flight.len());
        self.in_flight.retain(|&idx| {
            let Some(item) = self.items.get(idx) else {
                return true;
            };
            if let Some(score) = pattern.score(item.matcher_columns, matchers.get()) {
                let frecency_score = frecency
                    .as_ref()
                    .map_or(0.0, |ctx| ctx.score_for(item.data, now_secs));
                self.matches.push(Match {
                    score,
                    frecency_score,
                    idx,
                });
            };
            false
        });
        let new_snapshot = self.items.par_snapshot(self.last_snapshot);
        if new_snapshot.end() != self.last_snapshot {
            let end = new_snapshot.end();
            let in_flight = Mutex::new(&mut self.in_flight);
            let frecency_parallel = frecency.clone();
            let items = new_snapshot.map(|(idx, item)| {
                let Some(item) = item else {
                    in_flight.lock().push(idx);
                    unmatched.fetch_add(1, atomic::Ordering::Relaxed);
                    return Match {
                        score: 0,
                        frecency_score: 0.0,
                        idx: u32::MAX,
                    };
                };
                if self.canceled.load(atomic::Ordering::Relaxed) {
                    return Match {
                        score: 0,
                        frecency_score: 0.0,
                        idx,
                    };
                }
                let Some(score) = pattern.score(item.matcher_columns, matchers.get()) else {
                    unmatched.fetch_add(1, atomic::Ordering::Relaxed);
                    return Match {
                        score: 0,
                        frecency_score: 0.0,
                        idx: u32::MAX,
                    };
                };
                let frecency_score = frecency_parallel
                    .as_ref()
                    .map_or(0.0, |ctx| ctx.score_for(item.data, now_secs));
                Match {
                    score,
                    frecency_score,
                    idx,
                }
            });
            self.matches.par_extend(items);
            self.last_snapshot = end;
        }
    }

    fn remove_in_flight_matches(&mut self) {
        let mut off = 0;
        self.in_flight.retain(|&i| {
            let is_in_flight = self.items.get(i).is_none();
            if is_in_flight {
                self.matches.remove((i - off) as usize);
                off += 1;
            }
            is_in_flight
        });
    }

    unsafe fn process_new_items_trivial(&mut self) {
        let frecency = self.frecency.as_ref().cloned();
        let now_secs = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;
        let new_snapshot = self.items.snapshot(self.last_snapshot);
        if new_snapshot.end() != self.last_snapshot {
            let end = new_snapshot.end();
            let items = new_snapshot.filter_map(|(idx, item)| {
                let Some(item) = item else {
                    self.in_flight.push(idx);
                    return None;
                };
                let frecency_score = frecency
                    .as_ref()
                    .map_or(0.0, |ctx| ctx.score_for(item.data, now_secs));
                Some(Match {
                    score: 0,
                    frecency_score,
                    idx,
                })
            });
            self.matches.extend(items);
            self.last_snapshot = end;
        }
    }

    pub(crate) unsafe fn run(&mut self, pattern_status: pattern::Status, cleared: bool) {
        self.running = true;
        self.was_canceled = false;

        if cleared {
            self.last_snapshot = 0;
            self.in_flight.clear();
            self.matches.clear();
        }

        // TODO: be smarter around reusing past results for rescoring
        if self.pattern.is_empty() {
            self.reset_matches();
            self.process_new_items_trivial();
            if self.should_notify.load(atomic::Ordering::Relaxed) {
                (self.notify)();
            }
            return;
        }

        if pattern_status == pattern::Status::Rescore {
            self.reset_matches();
        }

        let mut unmatched = AtomicU32::new(0);
        if pattern_status != pattern::Status::Unchanged && !self.matches.is_empty() {
            self.process_new_items_trivial();
            let matchers = &self.matchers;
            let pattern = &self.pattern;
            let frecency = self.frecency.as_ref().cloned();
            let now_secs = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u32;
            self.matches
                .par_iter_mut()
                .take_any_while(|_| !self.canceled.load(atomic::Ordering::Relaxed))
                .for_each(|match_| {
                    if match_.idx == u32::MAX {
                        debug_assert_eq!(match_.score, 0);
                        unmatched.fetch_add(1, atomic::Ordering::Relaxed);
                        return;
                    }
                    // safety: in-flight items are never added to the matches
                    let item = self.items.get_unchecked(match_.idx);
                    if let Some(score) = pattern.score(item.matcher_columns, matchers.get()) {
                        match_.score = score;
                        match_.frecency_score = frecency
                            .as_ref()
                            .map_or(0.0, |ctx| ctx.score_for(item.data, now_secs));
                    } else {
                        unmatched.fetch_add(1, atomic::Ordering::Relaxed);
                        match_.score = 0;
                        match_.frecency_score = 0.0;
                        match_.idx = u32::MAX;
                    }
                });
        } else {
            self.process_new_items(&unmatched);
        }

        let canceled = self.sort_matches();
        if canceled {
            self.was_canceled = true;
        } else {
            self.matches
                .truncate(self.matches.len() - take(unmatched.get_mut()) as usize);
            if self.should_notify.load(atomic::Ordering::Relaxed) {
                (self.notify)();
            }
        }
    }

    unsafe fn sort_matches(&mut self) -> bool {
        if self.sort_results {
            par_quicksort(
                &mut self.matches,
                |match1, match2| {
                    if let Some(ordering) =
                        match1.frecency_score.partial_cmp(&match2.frecency_score)
                    {
                        if ordering != Ordering::Equal {
                            return ordering == Ordering::Greater;
                        }
                    }
                    if match1.score != match2.score {
                        return match1.score > match2.score;
                    }
                    if match1.idx == u32::MAX {
                        return false;
                    }
                    if match2.idx == u32::MAX {
                        return true;
                    }
                    // the tie breaker is comparatively rarely needed so we keep it
                    // in a branch especially because we need to access the items
                    // array here which involves some pointer chasing
                    let item1 = self.items.get_unchecked(match1.idx);
                    let item2 = &self.items.get_unchecked(match2.idx);
                    let len1: u32 = item1
                        .matcher_columns
                        .iter()
                        .map(|haystack| haystack.len() as u32)
                        .sum();
                    let len2 = item2
                        .matcher_columns
                        .iter()
                        .map(|haystack| haystack.len() as u32)
                        .sum();
                    if len1 == len2 {
                        if self.reverse_items {
                            match2.idx < match1.idx
                        } else {
                            match1.idx < match2.idx
                        }
                    } else {
                        len1 < len2
                    }
                },
                &self.canceled,
            )
        } else {
            par_quicksort(
                &mut self.matches,
                |match1, match2| {
                    if match1.idx == u32::MAX {
                        return false;
                    }
                    if match2.idx == u32::MAX {
                        return true;
                    }
                    if self.reverse_items {
                        match2.idx < match1.idx
                    } else {
                        match1.idx < match2.idx
                    }
                },
                &self.canceled,
            )
        }
    }

    fn reset_matches(&mut self) {
        self.matches.clear();
        self.matches
            .extend((0..self.last_snapshot).map(|idx| Match {
                score: 0,
                frecency_score: 0.0,
                idx,
            }));
        // there are usually only very few in flight items (one for each writer)
        self.remove_in_flight_matches();
    }
}
