use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use gauge_store::{
    GaugeStore, Matcher, Sample, Series, StoreConfig, StoreError, decode_samples, encode_samples,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_store() -> std::path::PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("gauge-round1-{}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn series(job: &str) -> Series {
    Series::from_labels("cpu", [("job", job), ("instance", "local")])
}

fn same_samples(left: &[Sample], right: &[Sample]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.timestamp == right.timestamp && left.value.to_bits() == right.value.to_bits()
        })
}

#[cfg(unix)]
fn open_fd_count() -> usize {
    fs::read_dir("/dev/fd")
        .or_else(|_| fs::read_dir("/proc/self/fd"))
        .unwrap()
        .count()
}

fn read_u32_at(bytes: &[u8], offset: &mut usize) -> u32 {
    let value = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    value
}

fn read_u64_at(bytes: &[u8], offset: &mut usize) -> u64 {
    let value = u64::from_le_bytes(bytes[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    value
}

fn rewrite_encoded_range(
    path: &std::path::Path,
    encoded_offset: u64,
    encoded_length: u64,
    chunk_header_length: Option<u32>,
) {
    let index_path = path.join("data").join("0").join("index.bin");
    let mut index = fs::read(&index_path).unwrap();
    let mut offset = 9; // magic plus format version
    assert_eq!(read_u32_at(&index, &mut offset), 1);
    offset += 8 + 8 + 8; // series id, record offset, record length
    assert_eq!(read_u32_at(&index, &mut offset), 1);
    offset += 8 + 8 + 8; // sequence, record offset, record length
    let encoded_offset_field = offset;
    let original_encoded_offset = read_u64_at(&index, &mut offset);
    let encoded_length_field = offset;
    let _original_encoded_length = read_u64_at(&index, &mut offset);
    index[encoded_offset_field..encoded_offset_field + 8]
        .copy_from_slice(&encoded_offset.to_le_bytes());
    index[encoded_length_field..encoded_length_field + 8]
        .copy_from_slice(&encoded_length.to_le_bytes());
    fs::write(index_path, index).unwrap();

    if let Some(chunk_header_length) = chunk_header_length {
        let chunks_path = path.join("data").join("0").join("chunks.bin");
        let mut chunks = fs::read(&chunks_path).unwrap();
        let encoded_length_offset = original_encoded_offset as usize - 4;
        chunks[encoded_length_offset..encoded_length_offset + 4]
            .copy_from_slice(&chunk_header_length.to_le_bytes());
        fs::write(chunks_path, chunks).unwrap();
    }
}

fn rewrite_chunk_sample_count(path: &std::path::Path, count: u32) {
    let index_path = path.join("data").join("0").join("index.bin");
    let index = fs::read(&index_path).unwrap();
    let mut offset = 9; // magic plus format version
    assert_eq!(read_u32_at(&index, &mut offset), 1);
    offset += 8 + 8 + 8; // series id, record offset, record length
    assert_eq!(read_u32_at(&index, &mut offset), 1);
    offset += 8 + 8 + 8; // sequence, record offset, record length
    let encoded_offset = read_u64_at(&index, &mut offset);

    let chunks_path = path.join("data").join("0").join("chunks.bin");
    let mut chunks = fs::read(&chunks_path).unwrap();
    let count_offset = encoded_offset as usize + 8;
    chunks[count_offset..count_offset + 4].copy_from_slice(&count.to_le_bytes());
    fs::write(chunks_path, chunks).unwrap();
}

#[test]
fn gorilla_round_trip_and_changing_fixture_stays_under_two_bytes() {
    let mut timestamp = 1_700_000_000_000_i64;
    let mut value = 100.0;
    let mut samples = Vec::new();
    let jitter = [-2, 0, 1, 0, 2, -1, 0, 1];
    for index in 0..5_000_i64 {
        timestamp += 15_000 + jitter[index as usize % jitter.len()];
        value += ((index * 17 % 11) as f64 - 5.0) * 0.25;
        samples.push(Sample::new(timestamp, value));
    }
    let encoded = encode_samples(&samples).unwrap();
    assert!(
        encoded.len() as f64 / samples.len() as f64 <= 2.0,
        "{} bytes/sample",
        encoded.len() as f64 / samples.len() as f64
    );
    assert!(same_samples(&decode_samples(&encoded).unwrap(), &samples));

    let constant: Vec<_> = (0..5_000)
        .map(|index| Sample::new(1_700_000_000_000 + index * 15_000, 42.0))
        .collect();
    let constant_encoded = encode_samples(&constant).unwrap();
    assert!(constant_encoded.len() as f64 / (constant.len() as f64) < 1.0);
    assert!(same_samples(
        &decode_samples(&constant_encoded).unwrap(),
        &constant
    ));
}

#[test]
fn gorilla_codec_handles_single_two_sample_and_control_bucket_edges() {
    for samples in [
        vec![Sample::new(7, 1.0)],
        vec![Sample::new(7, 1.0), Sample::new(22, 2.0)],
    ] {
        assert!(same_samples(
            &decode_samples(&encode_samples(&samples).unwrap()).unwrap(),
            &samples
        ));
    }

    let dod_values = [
        0_i64,
        -63,
        64,
        -64,
        65,
        -255,
        256,
        -256,
        257,
        -2_047,
        2_048,
        -2_048,
        2_049,
        -2_049,
        i32::MAX as i64,
        i32::MIN as i64,
    ];
    for dod in dod_values {
        let first_delta = 4_000_000_000_i64;
        let second_delta = first_delta + dod;
        assert!(second_delta > 0);
        let samples = vec![
            Sample::new(0, 1.0),
            Sample::new(first_delta, 1.5),
            Sample::new(first_delta + second_delta, 2.0),
        ];
        assert!(same_samples(
            &decode_samples(&encode_samples(&samples).unwrap()).unwrap(),
            &samples
        ));
    }

    let base = 0x3ff0_0000_0000_0000_u64;
    let windows: Vec<_> = (0..64)
        .map(|index| Sample::new(index * 15, f64::from_bits(base ^ (1_u64 << index))))
        .collect();
    assert!(same_samples(
        &decode_samples(&encode_samples(&windows).unwrap()).unwrap(),
        &windows
    ));
}

#[test]
fn seeded_randomized_irregular_and_adversarial_samples_round_trip() {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut timestamp = 0_i64;
    let mut walk = 10.0;
    let mut samples = Vec::new();
    for index in 0..2_000_i64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        timestamp += 1 + (state % 10_000) as i64;
        let value = match index % 17 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => -0.0,
            4 => f64::from_bits(1),
            _ => {
                walk += ((state >> 16) % 11) as f64 - 5.0;
                walk * 0.25
            }
        };
        samples.push(Sample::new(timestamp, value));
    }
    assert!(same_samples(
        &decode_samples(&encode_samples(&samples).unwrap()).unwrap(),
        &samples
    ));
}

#[test]
fn adversarial_float_property_fixture_round_trips_bit_for_bit() {
    let values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        -1.25,
        1.25,
        0.0,
    ];
    let samples: Vec<_> = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| Sample::new(index as i64 * 15_001, value))
        .collect();
    let decoded = decode_samples(&encode_samples(&samples).unwrap()).unwrap();
    assert!(same_samples(&decoded, &samples));
}

#[test]
fn regex_matchers_are_fully_anchored() {
    let path = temp_store();
    let store = GaugeStore::open(&path).unwrap();
    store.append(series("foo"), Sample::new(1, 1.0)).unwrap();
    store.append(series("foobar"), Sample::new(1, 2.0)).unwrap();
    assert_eq!(
        store
            .select(&[Matcher::regex("job", "foo")], 0, 2)
            .unwrap()
            .len(),
        1
    );
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn indexed_exact_and_regex_selection_returns_only_matching_series() {
    let path = temp_store();
    let store = GaugeStore::open(&path).unwrap();
    for (job, value) in [("api-a", 1.0), ("api-b", 2.0), ("worker", 3.0)] {
        store.append(series(job), Sample::new(1, value)).unwrap();
    }
    store.shutdown(1).unwrap();
    let selected = store
        .select(&[Matcher::regex("job", "api-.")], 0, 2)
        .unwrap();
    assert_eq!(selected.len(), 2);
    let selected = store
        .select(
            &[
                Matcher::exact("__name__", "cpu"),
                Matcher::exact("job", "worker"),
            ],
            0,
            2,
        )
        .unwrap();
    assert_eq!(selected.len(), 1);
    fs::remove_dir_all(path).unwrap();
}

#[cfg(unix)]
#[test]
fn broad_query_opens_each_physical_chunk_file_once() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(0);
    let store = GaugeStore::open_with_config(&path, config).unwrap();
    for index in 0..300_i64 {
        let series = Series::from_labels(
            "cpu",
            [
                ("job".to_owned(), format!("job-{index}")),
                ("instance".to_owned(), "local".to_owned()),
            ],
        );
        store.append(series, Sample::new(1, index as f64)).unwrap();
    }
    store.shutdown(100).unwrap();
    let chunk_file_count = fs::read_dir(path.join("data").join("0"))
        .unwrap()
        .filter(|entry| {
            entry.as_ref().is_ok_and(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.starts_with("chunks") && name.ends_with(".bin")
            })
        })
        .count();
    assert_eq!(chunk_file_count, 1);

    let baseline = open_fd_count();
    let mut during_snapshot = 0;
    let selected = store
        .reader()
        .select_with_hook(&[], 0, 2, || {
            during_snapshot = open_fd_count();
        })
        .unwrap();
    assert_eq!(during_snapshot - baseline, chunk_file_count);
    assert_eq!(selected.len(), 300);
    assert!(selected.iter().all(|row| row.samples.len() == 1));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn oversized_encoded_length_is_reported_without_allocation_abort() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(0);
    let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
    store
        .append(series("corrupt"), Sample::new(1, 7.0))
        .unwrap();
    store.shutdown(100).unwrap();
    drop(store);

    let oversized_length = u64::from(u32::MAX);
    rewrite_encoded_range(&path, oversized_length, oversized_length, Some(u32::MAX));

    let reopened = GaugeStore::open_with_config(&path, config).unwrap();
    let result = reopened.select(&[], 0, 2);
    assert!(matches!(result, Err(StoreError::CorruptStorage(_))));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn malformed_chunk_offsets_are_reported_without_allocation() {
    for (name, offset_kind, length) in [
        ("near-eof", 0_u8, 1_u64),
        ("past-eof", 1_u8, 1_u64),
        ("overflow", 2_u8, u64::MAX),
    ] {
        let path = temp_store();
        let config = StoreConfig::default()
            .with_partition_duration_ms(100)
            .with_out_of_order_tolerance_ms(0);
        let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
        store.append(series(name), Sample::new(1, 7.0)).unwrap();
        store.shutdown(100).unwrap();
        drop(store);
        let file_length = fs::metadata(path.join("data").join("0").join("chunks.bin"))
            .unwrap()
            .len();
        let encoded_offset = match offset_kind {
            0 => file_length - 1,
            1 => file_length + 1,
            _ => u64::MAX - 1,
        };
        rewrite_encoded_range(&path, encoded_offset, length, None);
        let reopened = GaugeStore::open_with_config(&path, config).unwrap();
        let result = reopened.select(&[], 0, 2);
        assert!(matches!(result, Err(StoreError::CorruptStorage(_))));
        fs::remove_dir_all(path).unwrap();
    }
}

#[test]
fn corrupt_sample_count_is_reported_without_allocation_abort() {
    let mut bytes = b"GAUGOR01".to_vec();
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.push(0);
    assert!(matches!(decode_samples(&bytes), Err(StoreError::Codec(_))));

    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(0);
    let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
    store
        .append(series("corrupt-count"), Sample::new(1, 7.0))
        .unwrap();
    store.shutdown(100).unwrap();
    drop(store);
    rewrite_chunk_sample_count(&path, u32::MAX);

    let reopened = GaugeStore::open_with_config(&path, config).unwrap();
    assert!(matches!(
        reopened.select(&[], 0, 2),
        Err(StoreError::CorruptStorage(_))
    ));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn out_of_order_tolerance_accepts_boundary_and_counts_rejection() {
    let path = temp_store();
    let config = StoreConfig::default().with_out_of_order_tolerance_ms(10);
    let store = GaugeStore::open_with_config(&path, config).unwrap();
    store
        .append(series("ordered"), Sample::new(100, 1.0))
        .unwrap();
    store
        .append(series("ordered"), Sample::new(90, 2.0))
        .unwrap();
    assert!(
        store
            .append(series("ordered"), Sample::new(89, 3.0))
            .is_err()
    );
    assert_eq!(store.stats().unwrap().rejected_out_of_order, 1);
    let samples = store.select(&[], 0, 200).unwrap()[0].samples.clone();
    assert!(same_samples(
        &samples,
        &[Sample::new(90, 2.0), Sample::new(100, 1.0)]
    ));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn retention_zero_never_deletes_the_active_partition() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(10)
        .with_retention_ms(0);
    let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
    store
        .append(series("active"), Sample::new(50, 9.0))
        .unwrap();
    store.shutdown(50).unwrap();
    drop(store);
    let reopened = GaugeStore::open_with_config(&path, config).unwrap();
    assert_eq!(
        reopened.select(&[], 0, 100).unwrap()[0].samples[0].value,
        9.0
    );
    assert!(path.join("data").join("0").exists());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn late_samples_use_immutable_delta_chunks_after_a_partition_is_published() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(10);
    let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
    store.append(series("late"), Sample::new(1, 1.0)).unwrap();
    store.append(series("late"), Sample::new(90, 9.0)).unwrap();
    store.shutdown(100).unwrap();
    let original = fs::read(path.join("data").join("0").join("chunks.bin")).unwrap();
    store.append(series("late"), Sample::new(85, 8.5)).unwrap();
    store.shutdown(100).unwrap();
    assert_eq!(
        fs::read(path.join("data").join("0").join("chunks.bin")).unwrap(),
        original
    );
    assert!(
        fs::read_dir(path.join("data").join("0"))
            .unwrap()
            .any(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("chunks-")
            })
    );
    drop(store);
    let reopened = GaugeStore::open_with_config(&path, config).unwrap();
    assert_eq!(reopened.select(&[], 0, 100).unwrap()[0].samples.len(), 3);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn reopen_strips_synthetic_name_and_replays_late_overwrite_in_sequence_order() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(10);
    let original = series("same");
    let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
    store
        .append(original.clone(), Sample::new(90, 1.0))
        .unwrap();
    store.shutdown(100).unwrap();
    drop(store);

    let reopened = GaugeStore::open_with_config(&path, config.clone()).unwrap();
    reopened
        .append(original.clone(), Sample::new(90, 2.0))
        .unwrap();
    reopened.shutdown(100).unwrap();
    drop(reopened);

    let final_store = GaugeStore::open_with_config(&path, config).unwrap();
    let selected = final_store
        .select(&[Matcher::exact("job", "same")], 0, 100)
        .unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].series, original);
    assert_eq!(selected[0].samples, vec![Sample::new(90, 2.0)]);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn atomic_directory_publication_recovers_from_a_real_child_process_crash() {
    let path = temp_store();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child_flush", "--nocapture"])
        .env("GAUGE_CRASH_CHILD_PATH", &path)
        .env("GAUGE_STORE_CRASH_AFTER_TEMP_SYNC", "1")
        .status()
        .unwrap();
    assert!(!child.success());
    let reopened = GaugeStore::open_with_config(
        &path,
        StoreConfig::default().with_partition_duration_ms(100),
    )
    .unwrap();
    assert_eq!(
        reopened.select(&[], 0, 100).unwrap()[0].samples[0].value,
        7.0
    );
    assert!(fs::read_dir(path.join("data")).unwrap().all(|entry| {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        !name.starts_with('.') && !name.ends_with(".deleting")
    }));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn tombstone_deletion_resumes_after_a_real_child_process_crash() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(0)
        .with_retention_ms(0);
    let store = GaugeStore::open_with_config(&path, config.clone()).unwrap();
    store
        .append(series("expired"), Sample::new(0, 1.0))
        .unwrap();
    store.shutdown(0).unwrap();
    drop(store);
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child_tombstone", "--nocapture"])
        .env("GAUGE_TOMBSTONE_CHILD_PATH", &path)
        .env("GAUGE_STORE_CRASH_AFTER_TOMBSTONE_RENAME", "1")
        .status()
        .unwrap();
    assert!(!child.success());
    let reopened = GaugeStore::open_with_config(&path, config).unwrap();
    assert!(reopened.select(&[], 0, 10).unwrap().is_empty());
    assert!(fs::read_dir(path.join("data")).unwrap().all(|entry| {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        !name.ends_with(".deleting")
    }));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn reader_keeps_persisted_chunks_alive_through_retention_delete() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(0)
        .with_retention_ms(1_000);
    let store = GaugeStore::open_with_config(&path, config).unwrap();
    for timestamp in 0..100_i64 {
        store
            .append(
                series("retained-reader"),
                Sample::new(timestamp, timestamp as f64),
            )
            .unwrap();
    }
    store.shutdown(100).unwrap();
    assert!(path.join("data").join("0").exists());

    let reader = store.reader();
    let snapshot_ready = Arc::new(Barrier::new(2));
    let release_reader = Arc::new(Barrier::new(2));
    let snapshot_ready_thread = Arc::clone(&snapshot_ready);
    let release_reader_thread = Arc::clone(&release_reader);
    let reader_thread = thread::spawn(move || {
        let rows = reader
            .select_with_hook(&[], 0, 100, move || {
                snapshot_ready_thread.wait();
                release_reader_thread.wait();
            })
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].samples.len(), 100);
        assert!(
            rows[0]
                .samples
                .iter()
                .enumerate()
                .all(|(index, sample)| sample == &Sample::new(index as i64, index as f64))
        );
    });

    // The reader has opened every persisted chunk before this barrier. The
    // retention flush can therefore unlink the partition while the reader's
    // decode remains backed by its open file descriptor.
    snapshot_ready.wait();
    store.flush(1_100).unwrap();
    assert!(!path.join("data").join("0").exists());
    release_reader.wait();
    reader_thread.join().unwrap();
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn crash_child_flush() {
    let Ok(path) = std::env::var("GAUGE_CRASH_CHILD_PATH") else {
        return;
    };
    let config = StoreConfig::default().with_partition_duration_ms(100);
    let store = GaugeStore::open_with_config(path, config).unwrap();
    store.append(series("crash"), Sample::new(1, 7.0)).unwrap();
    store.flush(100_000).unwrap();
}

#[test]
fn crash_child_tombstone() {
    let Ok(path) = std::env::var("GAUGE_TOMBSTONE_CHILD_PATH") else {
        return;
    };
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(0)
        .with_retention_ms(0);
    let store = GaugeStore::open_with_config(path, config).unwrap();
    store.flush(100).unwrap();
}

#[test]
fn concurrent_reader_can_iterate_while_flush_swaps_storage() {
    let path = temp_store();
    let config = StoreConfig::default()
        .with_partition_duration_ms(100)
        .with_out_of_order_tolerance_ms(0);
    let store = GaugeStore::open_with_config(&path, config).unwrap();
    for timestamp in 0..100_i64 {
        store
            .append(
                series("concurrent"),
                Sample::new(timestamp, timestamp as f64),
            )
            .unwrap();
    }
    let reader = store.reader();
    let reader_snapshotted = Arc::new(Barrier::new(2));
    let overlap = Arc::new(Barrier::new(3));
    let reader_snapshotted_thread = Arc::clone(&reader_snapshotted);
    let overlap_reader = Arc::clone(&overlap);
    let reader_thread = thread::spawn(move || {
        let rows = reader
            .select_with_hook(&[], 0, 1_000, move || {
                reader_snapshotted_thread.wait();
                overlap_reader.wait();
            })
            .unwrap();
        assert_eq!(rows[0].samples.len(), 100);
    });
    // The read has copied its head/partition sources and is waiting just
    // before decoding. The flush can now publish and enter its pre-swap
    // hook, making the overlap with the actual read deterministic.
    reader_snapshotted.wait();
    let flush_store = store.clone();
    let overlap_flush = Arc::clone(&overlap);
    let flush_thread = thread::spawn(move || {
        flush_store
            .flush_with_hook(100, move || {
                overlap_flush.wait();
            })
            .unwrap();
    });
    overlap.wait();
    flush_thread.join().unwrap();
    reader_thread.join().unwrap();
    assert_eq!(store.select(&[], 0, 1_000).unwrap()[0].samples.len(), 100);
    fs::remove_dir_all(path).unwrap();
}
