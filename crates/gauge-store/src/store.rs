use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use crate::format::{self, Partition, PartitionIndex};
use crate::matcher;
use crate::model::{
    Matcher, Sample, Series, SeriesSamples, StoreConfig, StoreError, StoreStats, partition_start,
    series_id, validate_series,
};
use crate::wal::{Wal, WalRecord};

const WAL_FILE: &str = "wal.log";

#[derive(Clone)]
pub struct GaugeStore {
    inner: Arc<Inner>,
}

pub type Store = GaugeStore;

pub struct WriteHandle {
    inner: Arc<Inner>,
}

pub struct ReadHandle {
    inner: Arc<Inner>,
}

pub struct SelectIter {
    inner: std::vec::IntoIter<SeriesSamples>,
}

struct SeriesSnapshot {
    series: Series,
    partitions: Vec<format::OpenStoredSeries>,
    head: Vec<Sample>,
}

struct Inner {
    data_root: PathBuf,
    config: StoreConfig,
    state: RwLock<State>,
    wal: Mutex<Wal>,
}

struct State {
    head: BTreeMap<u64, HeadSeries>,
    head_index: PartitionIndex,
    partitions: BTreeMap<i64, Partition>,
    catalog: BTreeMap<u64, Series>,
    latest: BTreeMap<u64, i64>,
    stats: StoreStats,
}

struct HeadSeries {
    series: Series,
    samples: BTreeMap<i64, f64>,
}

impl State {
    fn empty() -> Self {
        Self {
            head: BTreeMap::new(),
            head_index: PartitionIndex::default(),
            partitions: BTreeMap::new(),
            catalog: BTreeMap::new(),
            latest: BTreeMap::new(),
            stats: StoreStats::default(),
        }
    }

    fn add_catalog(&mut self, series: Series) -> Result<u64, StoreError> {
        validate_series(&series)?;
        let id = series_id(&series);
        if let Some(existing) = self.catalog.get(&id) {
            if existing != &series {
                return Err(StoreError::CorruptStorage(
                    "series ID hash collision".to_owned(),
                ));
            }
        } else {
            self.catalog.insert(id, series);
        }
        Ok(id)
    }

    fn insert_head(&mut self, id: u64, series: Series, sample: Sample) {
        self.latest
            .entry(id)
            .and_modify(|latest| *latest = (*latest).max(sample.timestamp))
            .or_insert(sample.timestamp);
        let head = self.head.entry(id).or_insert_with(|| {
            index_series(&mut self.head_index, id, &series);
            HeadSeries {
                series,
                samples: BTreeMap::new(),
            }
        });
        head.samples.insert(sample.timestamp, sample.value);
    }

    fn replay(&mut self, record: WalRecord) -> Result<(), StoreError> {
        let id = self.add_catalog(record.series.clone())?;
        self.insert_head(id, record.series, record.sample);
        Ok(())
    }

    fn sample_records(&self) -> Vec<WalRecord> {
        self.head
            .values()
            .flat_map(|head| {
                head.samples.iter().map(|(&timestamp, &value)| WalRecord {
                    series: head.series.clone(),
                    sample: Sample::new(timestamp, value),
                })
            })
            .collect()
    }

    fn recompute_latest(&mut self) -> Result<(), StoreError> {
        self.latest.clear();
        for (id, stored) in self
            .partitions
            .values()
            .flat_map(|partition| partition.series.iter())
        {
            if let Some(timestamp) = format::latest_timestamp(stored)? {
                self.latest
                    .entry(*id)
                    .and_modify(|latest| *latest = (*latest).max(timestamp))
                    .or_insert(timestamp);
            }
        }
        for (id, head) in &self.head {
            if let Some((&timestamp, _)) = head.samples.last_key_value() {
                self.latest
                    .entry(*id)
                    .and_modify(|latest| *latest = (*latest).max(timestamp))
                    .or_insert(timestamp);
            }
        }
        Ok(())
    }
}

impl GaugeStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with_config(path, StoreConfig::default())
    }

    pub fn open_with_config(
        path: impl AsRef<Path>,
        config: StoreConfig,
    ) -> Result<Self, StoreError> {
        config.validate()?;
        let data_root = path.as_ref().join("data");
        fs::create_dir_all(&data_root)?;
        let (partitions, catalog) = format::load_partitions(&data_root)?;
        let (wal, records) = Wal::open(data_root.join(WAL_FILE))?;
        let mut state = State::empty();
        state.partitions = partitions;
        state.catalog = catalog;
        state.recompute_latest()?;
        for record in records {
            state.replay(record)?;
        }
        Ok(Self {
            inner: Arc::new(Inner {
                data_root,
                config,
                state: RwLock::new(state),
                wal: Mutex::new(wal),
            }),
        })
    }

    pub fn writer(&self) -> WriteHandle {
        WriteHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    pub fn reader(&self) -> ReadHandle {
        ReadHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    pub fn append(&self, series: Series, sample: Sample) -> Result<(), StoreError> {
        self.writer().append(series, sample)
    }

    pub fn write(&self, series: Series, sample: Sample) -> Result<(), StoreError> {
        self.append(series, sample)
    }

    pub fn append_batch(&self, samples: &[(Series, Sample)]) -> Result<(), StoreError> {
        self.writer().append_batch(samples)
    }

    pub fn select(
        &self,
        matchers: &[Matcher],
        t_min: i64,
        t_max: i64,
    ) -> Result<Vec<SeriesSamples>, StoreError> {
        self.reader().select(matchers, t_min, t_max)
    }

    pub fn flush(&self, now_millis: i64) -> Result<(), StoreError> {
        flush_inner(&self.inner, Some(now_millis), false, None::<fn()>)
    }

    /// Flush with a deterministic callback invoked while the write lock is
    /// held. This is useful for proving read/flush overlap in callers' tests.
    pub fn flush_with_hook<F>(&self, now_millis: i64, hook: F) -> Result<(), StoreError>
    where
        F: FnOnce(),
    {
        flush_inner(&self.inner, Some(now_millis), false, Some(hook))
    }

    pub fn shutdown(&self, now_millis: i64) -> Result<(), StoreError> {
        flush_inner(&self.inner, Some(now_millis), true, None::<fn()>)
    }

    pub fn flush_all(&self) -> Result<(), StoreError> {
        flush_inner(&self.inner, None, true, None::<fn()>)
    }

    pub fn stats(&self) -> Result<StoreStats, StoreError> {
        Ok(read_lock(&self.inner.state)?.stats)
    }
}

impl WriteHandle {
    pub fn append(&self, series: Series, sample: Sample) -> Result<(), StoreError> {
        self.append_batch(&[(series, sample)])
    }

    pub fn write(&self, series: Series, sample: Sample) -> Result<(), StoreError> {
        self.append(series, sample)
    }

    pub fn append_batch(&self, samples: &[(Series, Sample)]) -> Result<(), StoreError> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut state = write_lock(&self.inner.state)?;
        let mut records = Vec::with_capacity(samples.len());
        let mut ids = Vec::with_capacity(samples.len());
        let mut batch_latest = state.latest.clone();
        for (series, sample) in samples {
            let id = state.add_catalog(series.clone())?;
            if let Some(&newest) = batch_latest.get(&id)
                && sample.timestamp
                    < newest.saturating_sub(self.inner.config.out_of_order_tolerance_ms)
            {
                state.stats.rejected_out_of_order += 1;
                return Err(StoreError::OutOfOrder {
                    timestamp: sample.timestamp,
                    newest,
                });
            }
            batch_latest
                .entry(id)
                .and_modify(|latest| *latest = (*latest).max(sample.timestamp))
                .or_insert(sample.timestamp);
            ids.push(id);
            records.push(WalRecord {
                series: series.clone(),
                sample: *sample,
            });
        }
        let mut wal = mutex_lock(&self.inner.wal)?;
        wal.append(&records)?;
        drop(wal);
        for ((series, sample), id) in samples.iter().zip(ids) {
            state.insert_head(id, series.clone(), *sample);
        }
        Ok(())
    }
}

impl ReadHandle {
    pub fn select(
        &self,
        matchers: &[Matcher],
        t_min: i64,
        t_max: i64,
    ) -> Result<Vec<SeriesSamples>, StoreError> {
        let snapshot = self.snapshot(matchers, t_min, t_max)?;
        select_snapshot(snapshot, t_min, t_max)
    }

    fn snapshot(
        &self,
        matchers: &[Matcher],
        t_min: i64,
        t_max: i64,
    ) -> Result<Vec<SeriesSnapshot>, StoreError> {
        if t_min > t_max {
            return Ok(Vec::new());
        }
        let compiled = matcher::compile(matchers)?;
        let state = read_lock(&self.inner.state)?;
        let mut candidate_ids = BTreeSet::new();
        for partition in state.partitions.values() {
            candidate_ids.extend(matcher::candidates(&partition.index, &compiled));
        }
        candidate_ids.extend(matcher::candidates(&state.head_index, &compiled));
        let mut result = Vec::new();
        let mut open_files = BTreeMap::new();
        for id in candidate_ids {
            let Some(series) = state.catalog.get(&id) else {
                continue;
            };
            if !matcher::matches(series, &compiled) {
                continue;
            }
            let partitions = state
                .partitions
                .values()
                .filter_map(|partition| partition.series.get(&id).cloned())
                .map(|stored| format::open_series(&stored, &mut open_files))
                .collect::<Result<Vec<_>, _>>()?;
            let head = state
                .head
                .get(&id)
                .map(|head| {
                    head.samples
                        .range(t_min..=t_max)
                        .map(|(&timestamp, &value)| Sample::new(timestamp, value))
                        .collect()
                })
                .unwrap_or_default();
            result.push(SeriesSnapshot {
                series: series.clone(),
                partitions,
                head,
            });
        }
        Ok(result)
    }

    pub fn select_iter(
        &self,
        matchers: &[Matcher],
        t_min: i64,
        t_max: i64,
    ) -> Result<SelectIter, StoreError> {
        Ok(SelectIter {
            inner: self.select(matchers, t_min, t_max)?.into_iter(),
        })
    }

    pub fn select_with_hook<F>(
        &self,
        matchers: &[Matcher],
        t_min: i64,
        t_max: i64,
        hook: F,
    ) -> Result<Vec<SeriesSamples>, StoreError>
    where
        F: FnOnce(),
    {
        let snapshot = self.snapshot(matchers, t_min, t_max)?;
        hook();
        select_snapshot(snapshot, t_min, t_max)
    }
}

fn select_snapshot(
    snapshot: Vec<SeriesSnapshot>,
    t_min: i64,
    t_max: i64,
) -> Result<Vec<SeriesSamples>, StoreError> {
    let mut result = Vec::new();
    for mut snapshot in snapshot {
        let mut samples = BTreeMap::new();
        for stored in &mut snapshot.partitions {
            for sample in format::read_open_series_samples(stored)? {
                if (t_min..=t_max).contains(&sample.timestamp) {
                    samples.insert(sample.timestamp, sample.value);
                }
            }
        }
        for sample in snapshot.head {
            samples.insert(sample.timestamp, sample.value);
        }
        if !samples.is_empty() {
            result.push(SeriesSamples {
                series: snapshot.series,
                samples: samples
                    .into_iter()
                    .map(|(timestamp, value)| Sample::new(timestamp, value))
                    .collect(),
            });
        }
    }
    Ok(result)
}

impl Iterator for SelectIter {
    type Item = SeriesSamples;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

fn flush_inner<F>(
    inner: &Arc<Inner>,
    now: Option<i64>,
    flush_all: bool,
    hook: Option<F>,
) -> Result<(), StoreError>
where
    F: FnOnce(),
{
    flush_inner_with_limit(
        inner,
        now,
        flush_all,
        hook,
        crate::format::MAX_ENCODED_CHUNK_SIZE,
    )
}

fn flush_inner_with_limit<F>(
    inner: &Arc<Inner>,
    now: Option<i64>,
    flush_all: bool,
    hook: Option<F>,
    max_encoded_chunk_size: u64,
) -> Result<(), StoreError>
where
    F: FnOnce(),
{
    let mut state = write_lock(&inner.state)?;
    let cutoff = now.map(|clock| clock.saturating_sub(inner.config.retention_ms));
    let active_partition =
        now.map(|clock| partition_start(clock, inner.config.partition_duration_ms));
    let starts: BTreeSet<i64> = state
        .head
        .values()
        .flat_map(|head| {
            head.samples
                .keys()
                .map(|&timestamp| partition_start(timestamp, inner.config.partition_duration_ms))
        })
        .filter(|start| {
            flush_all
                || now.is_some_and(|clock| partition_is_safe_to_flush(*start, clock, &inner.config))
                || cutoff.is_some_and(|limit| {
                    partition_end(*start, inner.config.partition_duration_ms) <= limit
                })
        })
        .collect();

    let mut rewritten = BTreeMap::new();
    for start in &starts {
        let combined = combine_partition(&state, *start, inner.config.partition_duration_ms)?;
        if !combined.is_empty() {
            rewritten.insert(
                *start,
                format::publish_partition_with_limit(
                    &inner.data_root,
                    *start,
                    &combined,
                    max_encoded_chunk_size,
                )?,
            );
        }
    }

    // The read path snapshots its sources before decoding them. Invoke the
    // test hook after publication and before the in-memory swap so a reader
    // can decode its snapshot while this swap is in flight.
    if let Some(hook) = hook {
        hook();
    }
    let delete: Vec<i64> = cutoff
        .map(|limit| {
            state
                .partitions
                .keys()
                .copied()
                .chain(rewritten.keys().copied())
                .chain(state.head.values().flat_map(|head| {
                    head.samples.keys().map(|&timestamp| {
                        partition_start(timestamp, inner.config.partition_duration_ms)
                    })
                }))
                .filter(|start| {
                    partition_end(*start, inner.config.partition_duration_ms) <= limit
                        && active_partition != Some(*start)
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default();

    for (&start, partition) in &rewritten {
        state.partitions.insert(start, partition.clone());
        remove_head_partition(&mut state, start, inner.config.partition_duration_ms);
        state.stats.flushed_partitions += 1;
    }
    for start in &delete {
        state.partitions.remove(start);
        remove_head_partition(&mut state, *start, inner.config.partition_duration_ms);
    }
    state.recompute_latest()?;
    let records = state.sample_records();
    let mut wal = mutex_lock(&inner.wal)?;
    wal.rewrite(&records)?;
    drop(wal);

    for start in delete {
        if format::delete_partition(&inner.data_root, start)? {
            state.stats.deleted_partitions += 1;
        }
    }
    Ok(())
}

fn partition_is_safe_to_flush(start: i64, now: i64, config: &StoreConfig) -> bool {
    partition_end(start, config.partition_duration_ms)
        .checked_add(config.out_of_order_tolerance_ms)
        .is_some_and(|safe_at| safe_at <= now)
}

fn partition_end(start: i64, duration: i64) -> i64 {
    start.saturating_add(duration)
}

fn combine_partition(
    state: &State,
    start: i64,
    duration: i64,
) -> Result<BTreeMap<u64, (Series, Vec<Sample>)>, StoreError> {
    let mut combined: BTreeMap<u64, (Series, BTreeMap<i64, f64>)> = BTreeMap::new();
    if let Some(partition) = state.partitions.get(&start) {
        for (&id, stored) in &partition.series {
            let entry = combined
                .entry(id)
                .or_insert_with(|| (stored.series.clone(), BTreeMap::new()));
            for sample in format::read_series_samples(stored)? {
                entry.1.insert(sample.timestamp, sample.value);
            }
        }
    }
    for (&id, head) in &state.head {
        let entry = combined
            .entry(id)
            .or_insert_with(|| (head.series.clone(), BTreeMap::new()));
        for (&timestamp, &value) in &head.samples {
            if partition_start(timestamp, duration) == start {
                entry.1.insert(timestamp, value);
            }
        }
        if entry.1.is_empty() {
            combined.remove(&id);
        }
    }
    Ok(combined
        .into_iter()
        .map(|(id, (series, samples))| {
            (
                id,
                (
                    series,
                    samples
                        .into_iter()
                        .map(|(timestamp, value)| Sample::new(timestamp, value))
                        .collect(),
                ),
            )
        })
        .collect())
}

fn remove_head_partition(state: &mut State, start: i64, duration: i64) {
    let empty: Vec<u64> = state
        .head
        .iter_mut()
        .filter_map(|(&id, head)| {
            head.samples
                .retain(|&timestamp, _| partition_start(timestamp, duration) != start);
            head.samples.is_empty().then_some(id)
        })
        .collect();
    for id in empty {
        state.head.remove(&id);
        remove_index_id(&mut state.head_index, id);
    }
}

fn index_series(index: &mut PartitionIndex, id: u64, series: &Series) {
    index.offsets.insert(id, (0, 0));
    index
        .labels
        .entry(("__name__".to_owned(), series.name.clone()))
        .or_default()
        .insert(id);
    for (name, value) in &series.labels {
        index
            .labels
            .entry((name.clone(), value.clone()))
            .or_default()
            .insert(id);
    }
}

fn remove_index_id(index: &mut PartitionIndex, id: u64) {
    index.offsets.remove(&id);
    let empty: Vec<_> = index
        .labels
        .iter_mut()
        .filter_map(|(key, ids)| {
            ids.remove(&id);
            ids.is_empty().then_some(key.clone())
        })
        .collect();
    for key in empty {
        index.labels.remove(&key);
    }
}

fn read_lock<'a>(
    lock: &'a RwLock<State>,
) -> Result<std::sync::RwLockReadGuard<'a, State>, StoreError> {
    lock.read().map_err(|_| StoreError::LockPoisoned)
}

fn write_lock<'a>(
    lock: &'a RwLock<State>,
) -> Result<std::sync::RwLockWriteGuard<'a, State>, StoreError> {
    lock.write().map_err(|_| StoreError::LockPoisoned)
}

fn mutex_lock<'a>(lock: &'a Mutex<Wal>) -> Result<std::sync::MutexGuard<'a, Wal>, StoreError> {
    lock.lock().map_err(|_| StoreError::LockPoisoned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_writer_refuses_flush_before_wal_truncation() {
        let path = std::env::temp_dir().join(format!("gauge-writer-cap-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        let config = StoreConfig::default()
            .with_partition_duration_ms(100)
            .with_out_of_order_tolerance_ms(0);
        let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
        store
            .append(
                Series::from_labels("cpu", [("job", "writer-cap")]),
                Sample::new(1, 7.0),
            )
            .unwrap();
        let result = flush_inner_with_limit(&store.inner, Some(100), true, None::<fn()>, 1);
        assert!(matches!(result, Err(StoreError::Codec(_))));
        drop(store);

        let reopened = GaugeStore::open_with_config(&path, config).unwrap();
        let selected = reopened.select(&[], 0, 2).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].samples, vec![Sample::new(1, 7.0)]);
        fs::remove_dir_all(path).unwrap();
    }
}
