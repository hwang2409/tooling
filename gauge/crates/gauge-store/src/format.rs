use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use crate::codec;
use crate::model::{LabelIndex, Sample, Series, StoreError};

const CHUNKS_MAGIC: &[u8; 8] = b"GAUCH001";
const INDEX_MAGIC: &[u8; 8] = b"GAUIN001";
const FORMAT_VERSION: u8 = 2;
const CHUNKS_FILE: &str = "chunks.bin";
const INDEX_FILE: &str = "index.bin";
const MAX_STRING_SIZE: usize = 16 * 1024 * 1024;
pub(crate) const MAX_ENCODED_CHUNK_SIZE: u64 = 256 * 1024 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
type LoadedPartitions = (BTreeMap<i64, Partition>, BTreeMap<u64, Series>);
type EncodedPartition = (Vec<u8>, Vec<u8>);

#[derive(Clone)]
pub(crate) struct Partition {
    pub(crate) series: BTreeMap<u64, StoredSeries>,
    pub(crate) index: PartitionIndex,
}

#[derive(Clone)]
pub(crate) struct StoredSeries {
    pub(crate) series: Series,
    pub(crate) chunks: Vec<ChunkRef>,
    pub(crate) latest_timestamp: Option<i64>,
}

#[derive(Clone)]
pub(crate) struct ChunkRef {
    path: PathBuf,
    record_offset: u64,
    record_length: u64,
    encoded_offset: u64,
    encoded_length: u64,
    sequence: u64,
}

pub(crate) struct OpenStoredSeries {
    chunks: Vec<OpenChunkRef>,
}

struct OpenChunkRef {
    file: Arc<File>,
    record_offset: u64,
    record_length: u64,
    encoded_offset: u64,
    encoded_length: u64,
    sequence: u64,
}

#[derive(Clone, Default)]
pub(crate) struct PartitionIndex {
    pub(crate) labels: LabelIndex,
    pub(crate) offsets: BTreeMap<u64, (u64, u64)>,
    entries: BTreeMap<u64, Vec<IndexedChunk>>,
}

#[derive(Clone)]
struct IndexedChunk {
    series: Series,
    sequence: u64,
    record_offset: u64,
    record_length: u64,
    encoded_offset: u64,
    encoded_length: u64,
    latest_timestamp: Option<i64>,
}

pub(crate) fn partition_path(data_root: &Path, start: i64) -> PathBuf {
    data_root.join(start.to_string())
}

pub(crate) fn load_partitions(data_root: &Path) -> Result<LoadedPartitions, StoreError> {
    fs::create_dir_all(data_root)?;
    cleanup_transients(data_root)?;
    let mut partitions = BTreeMap::new();
    let mut catalog = BTreeMap::new();
    for entry in fs::read_dir(data_root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Ok(start) = name.parse::<i64>() else {
            continue;
        };
        let partition = read_partition(&entry.path())?;
        for (&id, stored) in &partition.series {
            if let Some(existing) = catalog.get(&id) {
                if existing != &stored.series {
                    return Err(StoreError::CorruptStorage(
                        "series ID hash collision".to_owned(),
                    ));
                }
            } else {
                catalog.insert(id, stored.series.clone());
            }
        }
        partitions.insert(start, partition);
    }
    Ok((partitions, catalog))
}

pub(crate) fn publish_partition_with_limit(
    data_root: &Path,
    start: i64,
    series: &BTreeMap<u64, (Series, Vec<Sample>)>,
    max_encoded_chunk_size: u64,
) -> Result<Partition, StoreError> {
    let final_dir = partition_path(data_root, start);
    if final_dir.exists() {
        let existing = read_partition(&final_dir)?;
        if partition_matches(&existing, series)? {
            return Ok(existing);
        }
        let delta = partition_delta(&existing, series)?;
        if delta.is_empty() {
            return Ok(existing);
        }
        return append_delta(&final_dir, &delta, max_encoded_chunk_size);
    }

    let temp_dir = data_root.join(format!(
        ".{start}.tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;
    let (chunks, index) = encode_partition_with_limit(series, 0, max_encoded_chunk_size)?;
    write_synced(&temp_dir.join(CHUNKS_FILE), &chunks)?;
    write_synced(&temp_dir.join(INDEX_FILE), &index)?;
    sync_directory(&temp_dir)?;
    if std::env::var_os("GAUGE_STORE_CRASH_AFTER_TEMP_SYNC").is_some() {
        std::process::abort();
    }
    fs::rename(&temp_dir, &final_dir)?;
    sync_directory(data_root)?;
    read_partition(&final_dir)
}

fn partition_delta(
    existing: &Partition,
    expected: &BTreeMap<u64, (Series, Vec<Sample>)>,
) -> Result<BTreeMap<u64, (Series, Vec<Sample>)>, StoreError> {
    let mut delta = BTreeMap::new();
    for (&id, (series, samples)) in expected {
        let existing_samples = existing
            .series
            .get(&id)
            .map(read_series_samples)
            .transpose()?;
        let changed: Vec<_> = samples
            .iter()
            .copied()
            .filter(|sample| {
                existing_samples.as_ref().is_none_or(|stored| {
                    stored.iter().all(|old| {
                        old.timestamp != sample.timestamp
                            || old.value.to_bits() != sample.value.to_bits()
                    })
                })
            })
            .collect();
        if !changed.is_empty() {
            delta.insert(id, (series.clone(), changed));
        }
    }
    Ok(delta)
}

fn append_delta(
    partition_dir: &Path,
    delta: &BTreeMap<u64, (Series, Vec<Sample>)>,
    max_encoded_chunk_size: u64,
) -> Result<Partition, StoreError> {
    let mut sequence = next_chunk_sequence(partition_dir)?;
    let (chunks_name, index_name) = loop {
        let chunks_name = partition_dir.join(format!("chunks-{sequence:020}.bin"));
        let index_name = partition_dir.join(format!("index-{sequence:020}.bin"));
        if !chunks_name.exists() && !index_name.exists() {
            break (chunks_name, index_name);
        }
        sequence = sequence.saturating_add(1);
    };
    let temp_dir = partition_dir.join(format!(".late-{sequence:020}.tmp"));
    fs::create_dir_all(&temp_dir)?;
    let (chunks, index) = encode_partition_with_limit(delta, sequence, max_encoded_chunk_size)?;
    write_synced(&temp_dir.join(CHUNKS_FILE), &chunks)?;
    write_synced(&temp_dir.join(INDEX_FILE), &index)?;
    sync_directory(&temp_dir)?;
    fs::rename(temp_dir.join(CHUNKS_FILE), &chunks_name)?;
    fs::rename(temp_dir.join(INDEX_FILE), &index_name)?;
    fs::remove_dir_all(&temp_dir)?;
    sync_directory(partition_dir)?;

    read_partition(partition_dir)
}

fn next_chunk_sequence(partition_dir: &Path) -> Result<u64, StoreError> {
    let mut maximum = None;
    for entry in fs::read_dir(partition_dir)? {
        let path = entry?.path();
        let is_chunk = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("chunks") && name.ends_with(".bin"));
        if is_chunk {
            let sequence = read_chunk_sequence(&path)?;
            maximum = Some(maximum.map_or(sequence, |current: u64| current.max(sequence)));
        }
    }
    maximum
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| StoreError::CorruptStorage("chunk sequence exhausted".to_owned()))
}

pub(crate) fn delete_partition(data_root: &Path, start: i64) -> Result<bool, StoreError> {
    let source = partition_path(data_root, start);
    if !source.exists() {
        return Ok(false);
    }
    let tombstone = data_root.join(format!("{start}.deleting"));
    if tombstone.exists() {
        fs::remove_dir_all(&tombstone)?;
    }
    fs::rename(source, &tombstone)?;
    sync_directory(data_root)?;
    if std::env::var_os("GAUGE_STORE_CRASH_AFTER_TOMBSTONE_RENAME").is_some() {
        std::process::abort();
    }
    fs::remove_dir_all(tombstone)?;
    sync_directory(data_root)?;
    Ok(true)
}

fn cleanup_transients(data_root: &Path) -> Result<(), StoreError> {
    for entry in fs::read_dir(data_root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name.ends_with(".deleting") {
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

fn partition_matches(
    partition: &Partition,
    expected: &BTreeMap<u64, (Series, Vec<Sample>)>,
) -> Result<bool, StoreError> {
    if partition.series.len() != expected.len() {
        return Ok(false);
    }
    for (id, (series, samples)) in expected {
        let Some(stored) = partition.series.get(id) else {
            return Ok(false);
        };
        if stored.series != *series || !samples_bits_equal(&read_series_samples(stored)?, samples) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn samples_bits_equal(left: &[Sample], right: &[Sample]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.timestamp == right.timestamp && left.value.to_bits() == right.value.to_bits()
        })
}

fn encode_partition_with_limit(
    series: &BTreeMap<u64, (Series, Vec<Sample>)>,
    sequence: u64,
    max_encoded_chunk_size: u64,
) -> Result<EncodedPartition, StoreError> {
    let mut chunks = Vec::new();
    chunks.extend_from_slice(CHUNKS_MAGIC);
    chunks.push(FORMAT_VERSION);
    put_u64(&mut chunks, sequence);
    chunks.extend_from_slice(&(series.len() as u32).to_le_bytes());
    let mut index = PartitionIndex::default();
    for (&id, (series, samples)) in series {
        let offset = chunks.len() as u64;
        let encoded =
            codec::encode(samples).map_err(|error| StoreError::Codec(error.to_string()))?;
        let encoded_length = u64::try_from(encoded.len()).map_err(|_| {
            StoreError::Codec("encoded chunk length does not fit on disk".to_owned())
        })?;
        if encoded_length > max_encoded_chunk_size {
            return Err(StoreError::Codec(format!(
                "encoded chunk exceeds maximum size ({max_encoded_chunk_size} bytes)"
            )));
        }
        let encoded_length_u32 = u32::try_from(encoded_length).map_err(|_| {
            StoreError::Codec("encoded chunk length exceeds on-disk field".to_owned())
        })?;
        put_u64(&mut chunks, id);
        put_string(&mut chunks, &series.name);
        put_u32(&mut chunks, series.labels.len() as u32);
        index
            .labels
            .entry(("__name__".to_owned(), series.name.clone()))
            .or_default()
            .insert(id);
        for (name, value) in &series.labels {
            put_string(&mut chunks, name);
            put_string(&mut chunks, value);
            index
                .labels
                .entry((name.clone(), value.clone()))
                .or_default()
                .insert(id);
        }
        put_u32(&mut chunks, encoded_length_u32);
        let encoded_offset = chunks.len() as u64;
        chunks.extend_from_slice(&encoded);
        let length = chunks.len() as u64 - offset;
        index.offsets.insert(id, (offset, length));
        index.entries.entry(id).or_default().push(IndexedChunk {
            series: series.clone(),
            sequence,
            record_offset: offset,
            record_length: length,
            encoded_offset,
            encoded_length,
            latest_timestamp: samples.last().map(|sample| sample.timestamp),
        });
    }
    let index_bytes = encode_index(&index);
    Ok((chunks, index_bytes))
}

fn encode_index(index: &PartitionIndex) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(INDEX_MAGIC);
    bytes.push(FORMAT_VERSION);
    put_u32(&mut bytes, index.offsets.len() as u32);
    for (&id, &(offset, length)) in &index.offsets {
        put_u64(&mut bytes, id);
        put_u64(&mut bytes, offset);
        put_u64(&mut bytes, length);
        let entries = index.entries.get(&id).map_or(&[][..], Vec::as_slice);
        put_u32(&mut bytes, entries.len() as u32);
        for entry in entries {
            put_u64(&mut bytes, entry.sequence);
            put_u64(&mut bytes, entry.record_offset);
            put_u64(&mut bytes, entry.record_length);
            put_u64(&mut bytes, entry.encoded_offset);
            put_u64(&mut bytes, entry.encoded_length);
            put_optional_i64(&mut bytes, entry.latest_timestamp);
            put_string(&mut bytes, &entry.series.name);
            put_u32(&mut bytes, entry.series.labels.len() as u32);
            for (name, value) in &entry.series.labels {
                put_string(&mut bytes, name);
                put_string(&mut bytes, value);
            }
        }
    }
    put_u32(&mut bytes, index.labels.len() as u32);
    for ((name, value), ids) in &index.labels {
        put_string(&mut bytes, name);
        put_string(&mut bytes, value);
        put_u32(&mut bytes, ids.len() as u32);
        for id in ids {
            put_u64(&mut bytes, *id);
        }
    }
    bytes
}

fn decode_index(bytes: &[u8]) -> Result<PartitionIndex, StoreError> {
    let mut reader = BinaryReader::new(bytes);
    if reader.take(8)? != INDEX_MAGIC || reader.byte()? != FORMAT_VERSION {
        return Err(StoreError::CorruptStorage(
            "invalid index header".to_owned(),
        ));
    }
    let offsets = reader.u32()? as usize;
    if offsets > reader.remaining() / 28 {
        return Err(StoreError::CorruptStorage(
            "index offset count exceeds encoded bytes".to_owned(),
        ));
    }
    let mut index = PartitionIndex::default();
    for _ in 0..offsets {
        let id = reader.u64()?;
        let offset = reader.u64()?;
        let length = reader.u64()?;
        index.offsets.insert(id, (offset, length));
        let entries = reader.u32()? as usize;
        if entries > reader.remaining() / 49 {
            return Err(StoreError::CorruptStorage(
                "index entry count exceeds encoded bytes".to_owned(),
            ));
        }
        let mut decoded_entries = Vec::new();
        decoded_entries
            .try_reserve_exact(entries)
            .map_err(|_| StoreError::CorruptStorage("index entry allocation failed".to_owned()))?;
        for _ in 0..entries {
            let sequence = reader.u64()?;
            let record_offset = reader.u64()?;
            let record_length = reader.u64()?;
            let encoded_offset = reader.u64()?;
            let encoded_length = reader.u64()?;
            let latest_timestamp = reader.optional_i64()?;
            let name = reader.string()?;
            let labels = reader.u32()? as usize;
            if labels > reader.remaining() / 8 {
                return Err(StoreError::CorruptStorage(
                    "index label count exceeds encoded bytes".to_owned(),
                ));
            }
            let mut label_map = BTreeMap::new();
            for _ in 0..labels {
                label_map.insert(reader.string()?, reader.string()?);
            }
            decoded_entries.push(IndexedChunk {
                series: Series::new(name, label_map),
                sequence,
                record_offset,
                record_length,
                encoded_offset,
                encoded_length,
                latest_timestamp,
            });
        }
        index.entries.insert(id, decoded_entries);
    }
    let labels = reader.u32()? as usize;
    if labels > reader.remaining() / 8 {
        return Err(StoreError::CorruptStorage(
            "index label count exceeds encoded bytes".to_owned(),
        ));
    }
    for _ in 0..labels {
        let key = (reader.string()?, reader.string()?);
        let ids = reader.u32()? as usize;
        if ids > reader.remaining() / 8 {
            return Err(StoreError::CorruptStorage(
                "index series count exceeds encoded bytes".to_owned(),
            ));
        }
        let values = index.labels.entry(key).or_default();
        for _ in 0..ids {
            values.insert(reader.u64()?);
        }
    }
    if !reader.is_empty() {
        return Err(StoreError::CorruptStorage(
            "trailing bytes in index".to_owned(),
        ));
    }
    Ok(index)
}

fn read_partition(path: &Path) -> Result<Partition, StoreError> {
    cleanup_partition_transients(path)?;
    let chunk_paths = discover_chunk_paths(path)?;
    let mut persisted_index = PartitionIndex::default();
    for entry in fs::read_dir(path)? {
        let candidate = entry?.path();
        let is_index = candidate
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("index") && name.ends_with(".bin"));
        if is_index {
            let mut index_bytes = Vec::new();
            File::open(candidate)?.read_to_end(&mut index_bytes)?;
            let decoded = decode_index(&index_bytes)?;
            persisted_index.offsets.extend(decoded.offsets);
            for (key, ids) in decoded.labels {
                persisted_index.labels.entry(key).or_default().extend(ids);
            }
            for (id, entries) in decoded.entries {
                persisted_index
                    .entries
                    .entry(id)
                    .or_default()
                    .extend(entries);
            }
        }
    }

    let paths_by_sequence: BTreeMap<_, _> = chunk_paths.iter().cloned().collect();
    let indexed_sequences: std::collections::BTreeSet<_> = persisted_index
        .entries
        .values()
        .flat_map(|entries| entries.iter().map(|entry| entry.sequence))
        .collect();
    let chunk_sequences: std::collections::BTreeSet<_> =
        chunk_paths.iter().map(|(sequence, _)| *sequence).collect();
    if !persisted_index.entries.is_empty() && indexed_sequences == chunk_sequences {
        let mut series = BTreeMap::new();
        for (&id, entries) in &persisted_index.entries {
            let mut chunks = Vec::new();
            chunks.try_reserve_exact(entries.len()).map_err(|_| {
                StoreError::CorruptStorage("chunk reference allocation failed".to_owned())
            })?;
            let mut series_definition = None;
            for entry in entries {
                let Some(path) = paths_by_sequence.get(&entry.sequence) else {
                    return Err(StoreError::CorruptStorage(
                        "index references missing chunk sequence".to_owned(),
                    ));
                };
                if series_definition
                    .as_ref()
                    .is_some_and(|definition: &Series| definition != &entry.series)
                {
                    return Err(StoreError::CorruptStorage(
                        "series differs across immutable chunks".to_owned(),
                    ));
                }
                series_definition = Some(entry.series.clone());
                chunks.push(ChunkRef {
                    path: path.clone(),
                    record_offset: entry.record_offset,
                    record_length: entry.record_length,
                    encoded_offset: entry.encoded_offset,
                    encoded_length: entry.encoded_length,
                    sequence: entry.sequence,
                });
            }
            let Some(series_definition) = series_definition else {
                return Err(StoreError::CorruptStorage(
                    "index contains an empty series entry".to_owned(),
                ));
            };
            series.insert(
                id,
                StoredSeries {
                    series: series_definition,
                    chunks,
                    latest_timestamp: entries
                        .iter()
                        .filter_map(|entry| entry.latest_timestamp)
                        .max(),
                },
            );
        }
        return Ok(Partition {
            series,
            index: persisted_index,
        });
    }

    // A crash can publish one side of a delta before the other. Rebuild the
    // metadata from chunk records until the serialized index catches up.
    read_partition_from_chunks(chunk_paths)
}

fn cleanup_partition_transients(path: &Path) -> Result<(), StoreError> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() && entry.file_name().to_string_lossy().starts_with('.') {
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

fn discover_chunk_paths(path: &Path) -> Result<Vec<(u64, PathBuf)>, StoreError> {
    let mut chunk_paths: Vec<(u64, PathBuf)> = fs::read_dir(path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("chunks") && name.ends_with(".bin"))
        })
        .map(|path| read_chunk_sequence(&path).map(|sequence| (sequence, path)))
        .collect::<Result<_, _>>()?;
    chunk_paths.sort_by_key(|(sequence, _)| *sequence);
    if chunk_paths.is_empty() {
        return Err(StoreError::CorruptStorage(format!(
            "no chunk file in {}",
            path.display()
        )));
    }
    for pair in chunk_paths.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(StoreError::CorruptStorage(
                "duplicate chunk sequence".to_owned(),
            ));
        }
    }
    Ok(chunk_paths)
}

fn read_partition_from_chunks(chunk_paths: Vec<(u64, PathBuf)>) -> Result<Partition, StoreError> {
    let mut series = BTreeMap::new();
    let mut derived = PartitionIndex::default();
    for (_, chunk_path) in chunk_paths {
        let mut bytes = Vec::new();
        File::open(&chunk_path)?.read_to_end(&mut bytes)?;
        decode_chunk_file(&bytes, &chunk_path, &mut series, &mut derived)?;
    }
    Ok(Partition {
        series,
        index: derived,
    })
}

fn read_chunk_sequence(path: &Path) -> Result<u64, StoreError> {
    let mut file = File::open(path)?;
    let mut header = [0_u8; 17];
    file.read_exact(&mut header)?;
    if &header[..8] != CHUNKS_MAGIC || header[8] != FORMAT_VERSION {
        return Err(StoreError::CorruptStorage(format!(
            "invalid chunk header in {}",
            path.display()
        )));
    }
    Ok(u64::from_le_bytes(
        header[9..17].try_into().expect("eight-byte sequence"),
    ))
}

fn decode_chunk_file(
    bytes: &[u8],
    path: &Path,
    series: &mut BTreeMap<u64, StoredSeries>,
    index: &mut PartitionIndex,
) -> Result<(), StoreError> {
    let mut reader = BinaryReader::new(bytes);
    if reader.take(8)? != CHUNKS_MAGIC || reader.byte()? != FORMAT_VERSION {
        return Err(StoreError::CorruptStorage(
            "invalid chunk header".to_owned(),
        ));
    }
    let sequence = reader.u64()?;
    let count = reader.u32()? as usize;
    if count > reader.remaining() / 20 {
        return Err(StoreError::CorruptStorage(
            "chunk series count exceeds encoded bytes".to_owned(),
        ));
    }
    for _ in 0..count {
        let offset = reader.position as u64;
        let id = reader.u64()?;
        let name = reader.string()?;
        let labels = reader.u32()? as usize;
        if labels > reader.remaining() / 8 {
            return Err(StoreError::CorruptStorage(
                "chunk label count exceeds encoded bytes".to_owned(),
            ));
        }
        let mut label_map = BTreeMap::new();
        for _ in 0..labels {
            let label = reader.string()?;
            let value = reader.string()?;
            if label == "__name__" {
                if value != name {
                    return Err(StoreError::CorruptStorage(
                        "synthetic name label disagrees with series name".to_owned(),
                    ));
                }
            } else {
                index
                    .labels
                    .entry((label.clone(), value.clone()))
                    .or_default()
                    .insert(id);
                label_map.insert(label, value);
            }
        }
        index
            .labels
            .entry(("__name__".to_owned(), name.clone()))
            .or_default()
            .insert(id);
        let encoded_len = reader.u32()? as usize;
        let encoded_offset = reader.position as u64;
        reader.take(encoded_len)?;
        let length = reader.position as u64 - offset;
        index.offsets.insert(id, (offset, length));
        let entry = series.entry(id).or_insert_with(|| StoredSeries {
            series: Series::new(name.clone(), label_map.clone()),
            chunks: Vec::new(),
            latest_timestamp: None,
        });
        if entry.series != Series::new(name, label_map) {
            return Err(StoreError::CorruptStorage(
                "series differs across immutable chunks".to_owned(),
            ));
        }
        let entry_series = entry.series.clone();
        entry.chunks.push(ChunkRef {
            path: path.to_owned(),
            record_offset: offset,
            record_length: length,
            encoded_offset,
            encoded_length: encoded_len as u64,
            sequence,
        });
        index.entries.entry(id).or_default().push(IndexedChunk {
            series: entry_series,
            sequence,
            record_offset: offset,
            record_length: length,
            encoded_offset,
            encoded_length: encoded_len as u64,
            latest_timestamp: None,
        });
    }
    if !reader.is_empty() {
        return Err(StoreError::CorruptStorage(
            "trailing bytes in chunk file".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn open_series(
    stored: &StoredSeries,
    files: &mut BTreeMap<PathBuf, Arc<File>>,
) -> Result<OpenStoredSeries, StoreError> {
    let chunks = stored
        .chunks
        .iter()
        .map(|chunk| {
            let file = if let Some(file) = files.get(&chunk.path) {
                Arc::clone(file)
            } else {
                let file = Arc::new(File::open(&chunk.path)?);
                files.insert(chunk.path.clone(), Arc::clone(&file));
                file
            };
            Ok(OpenChunkRef {
                file,
                record_offset: chunk.record_offset,
                record_length: chunk.record_length,
                encoded_offset: chunk.encoded_offset,
                encoded_length: chunk.encoded_length,
                sequence: chunk.sequence,
            })
        })
        .collect::<Result<_, std::io::Error>>()?;
    Ok(OpenStoredSeries { chunks })
}

pub(crate) fn read_series_samples(stored: &StoredSeries) -> Result<Vec<Sample>, StoreError> {
    let mut chunks = stored.chunks.clone();
    chunks.sort_by_key(|chunk| chunk.sequence);
    let mut samples = BTreeMap::new();
    for chunk in chunks {
        let file = File::open(chunk.path)?;
        read_chunk_samples(
            &file,
            chunk.record_offset,
            chunk.record_length,
            chunk.encoded_offset,
            chunk.encoded_length,
            &mut samples,
        )?;
    }
    Ok(samples
        .into_iter()
        .map(|(timestamp, value)| Sample::new(timestamp, value))
        .collect())
}

pub(crate) fn read_open_series_samples(
    stored: &mut OpenStoredSeries,
) -> Result<Vec<Sample>, StoreError> {
    stored.chunks.sort_by_key(|chunk| chunk.sequence);
    let mut samples = BTreeMap::new();
    for chunk in &stored.chunks {
        read_chunk_samples(
            &chunk.file,
            chunk.record_offset,
            chunk.record_length,
            chunk.encoded_offset,
            chunk.encoded_length,
            &mut samples,
        )?;
    }
    Ok(samples
        .into_iter()
        .map(|(timestamp, value)| Sample::new(timestamp, value))
        .collect())
}

fn read_chunk_samples(
    file: &File,
    record_offset: u64,
    record_length: u64,
    encoded_offset: u64,
    encoded_length: u64,
    samples: &mut BTreeMap<i64, f64>,
) -> Result<(), StoreError> {
    let length = validate_chunk_bounds(
        file,
        record_offset,
        record_length,
        encoded_offset,
        encoded_length,
    )?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(length)
        .map_err(|_| StoreError::CorruptStorage("encoded chunk allocation failed".to_owned()))?;
    encoded.resize(length, 0);
    read_exact_at(file, &mut encoded, encoded_offset)?;
    let decoded = codec::decode(&encoded)
        .map_err(|error| StoreError::CorruptStorage(format!("invalid encoded chunk: {error}")))?;
    for sample in decoded {
        samples.insert(sample.timestamp, sample.value);
    }
    Ok(())
}

fn validate_chunk_bounds(
    file: &File,
    record_offset: u64,
    record_length: u64,
    encoded_offset: u64,
    encoded_length: u64,
) -> Result<usize, StoreError> {
    let file_length = file.metadata()?.len();
    let record_end = checked_file_range(record_offset, record_length, file_length, "chunk record")?;
    let encoded_end =
        checked_file_range(encoded_offset, encoded_length, file_length, "encoded chunk")?;
    let encoded_length_offset = encoded_offset.checked_sub(4).ok_or_else(|| {
        StoreError::CorruptStorage("encoded length field is before file start".to_owned())
    })?;
    if encoded_length_offset < record_offset || encoded_end > record_end {
        return Err(StoreError::CorruptStorage(
            "encoded range is outside its chunk record".to_owned(),
        ));
    }

    let mut length_bytes = [0_u8; 4];
    read_exact_at(file, &mut length_bytes, encoded_length_offset)?;
    let on_disk_length = u64::from(u32::from_le_bytes(length_bytes));
    if on_disk_length != encoded_length {
        return Err(StoreError::CorruptStorage(
            "index and chunk encoded lengths differ".to_owned(),
        ));
    }
    if encoded_length > MAX_ENCODED_CHUNK_SIZE {
        return Err(StoreError::CorruptStorage(
            "encoded chunk exceeds maximum size".to_owned(),
        ));
    }
    usize::try_from(encoded_length)
        .map_err(|_| StoreError::CorruptStorage("encoded chunk is too large".to_owned()))
}

fn checked_file_range(
    offset: u64,
    length: u64,
    file_length: u64,
    kind: &str,
) -> Result<u64, StoreError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| StoreError::CorruptStorage(format!("{kind} range overflows file size")))?;
    if end > file_length {
        return Err(StoreError::CorruptStorage(format!(
            "{kind} range exceeds file size"
        )));
    }
    Ok(end)
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    file.read_at(buffer, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    file.seek_read(buffer, offset)
}

fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    let mut position = 0;
    while position < buffer.len() {
        let read = read_at(file, &mut buffer[position..], offset + position as u64)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated encoded chunk",
            ));
        }
        position += read;
    }
    Ok(())
}

pub(crate) fn latest_timestamp(stored: &StoredSeries) -> Result<Option<i64>, StoreError> {
    match stored.latest_timestamp {
        Some(timestamp) => Ok(Some(timestamp)),
        None => Ok(read_series_samples(stored)?
            .last()
            .map(|sample| sample.timestamp)),
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_optional_i64(out: &mut Vec<u8>, value: Option<i64>) {
    match value {
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.to_le_bytes());
        }
        None => out.push(0),
    }
}
fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), StoreError> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    Ok(())
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8], StoreError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| StoreError::CorruptStorage("length overflow".to_owned()))?;
        if end > self.bytes.len() {
            return Err(StoreError::CorruptStorage(
                "truncated storage file".to_owned(),
            ));
        }
        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }
    fn byte(&mut self) -> Result<u8, StoreError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, StoreError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four-byte slice"),
        ))
    }
    fn u64(&mut self) -> Result<u64, StoreError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight-byte slice"),
        ))
    }
    fn optional_i64(&mut self) -> Result<Option<i64>, StoreError> {
        match self.byte()? {
            0 => Ok(None),
            1 => Ok(Some(i64::from_le_bytes(
                self.take(8)?.try_into().expect("eight-byte slice"),
            ))),
            _ => Err(StoreError::CorruptStorage(
                "invalid optional timestamp marker".to_owned(),
            )),
        }
    }
    fn string(&mut self) -> Result<String, StoreError> {
        let length = self.u32()? as usize;
        if length > MAX_STRING_SIZE {
            return Err(StoreError::CorruptStorage("oversized string".to_owned()));
        }
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| StoreError::CorruptStorage("invalid UTF-8".to_owned()))
    }
    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_contains_metric_name_and_labels() {
        let series = BTreeMap::from([(
            4,
            (
                Series::from_labels("up", [("job", "a")]),
                vec![Sample::new(1, 1.0)],
            ),
        )]);
        let (_, index) = encode_partition_with_limit(&series, 0, MAX_ENCODED_CHUNK_SIZE).unwrap();
        let index = decode_index(&index).unwrap();
        assert!(
            index
                .labels
                .contains_key(&("__name__".to_owned(), "up".to_owned()))
        );
        assert!(
            index
                .labels
                .contains_key(&("job".to_owned(), "a".to_owned()))
        );
    }
}
