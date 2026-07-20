use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::model::{Sample, Series};

const RECORD_VERSION: u8 = 1;
const MAX_RECORD_SIZE: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct WalRecord {
    pub(crate) series: Series,
    pub(crate) sample: Sample,
}

pub(crate) struct Wal {
    path: PathBuf,
    file: File,
}

impl Wal {
    pub(crate) fn open(path: impl AsRef<Path>) -> io::Result<(Self, Vec<WalRecord>)> {
        let path = path.as_ref().to_owned();
        let mut bytes = Vec::new();
        if path.exists() {
            File::open(&path)?.read_to_end(&mut bytes)?;
        }

        let (records, valid_len) = decode_records(&bytes)?;
        if valid_len != bytes.len() {
            let file = OpenOptions::new().write(true).open(&path)?;
            file.set_len(valid_len as u64)?;
            file.sync_all()?;
        }

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        sync_parent(&path)?;
        Ok((Self { path, file }, records))
    }

    pub(crate) fn append(&mut self, records: &[WalRecord]) -> io::Result<()> {
        for record in records {
            let payload = encode_record(record);
            self.file.write_all(&(payload.len() as u32).to_le_bytes())?;
            self.file.write_all(&payload)?;
        }
        // A write is acknowledged only after the complete batch reaches the
        // filesystem. sync_all is intentional: the WAL is the durability
        // boundary, not merely a buffered append log.
        self.file.sync_all()
    }

    pub(crate) fn rewrite(&mut self, records: &[WalRecord]) -> io::Result<()> {
        let temp_path = self.path.with_extension("log.rewrite");
        {
            let mut temp = File::create(&temp_path)?;
            for record in records {
                let payload = encode_record(record);
                temp.write_all(&(payload.len() as u32).to_le_bytes())?;
                temp.write_all(&payload)?;
            }
            temp.sync_all()?;
        }
        std::fs::rename(&temp_path, &self.path)?;
        sync_parent(&self.path)?;
        self.file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&self.path)?;
        Ok(())
    }
}

fn encode_record(record: &WalRecord) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(RECORD_VERSION);
    put_string(&mut out, &record.series.name);
    put_u32(&mut out, record.series.labels.len() as u32);
    for (name, value) in &record.series.labels {
        put_string(&mut out, name);
        put_string(&mut out, value);
    }
    out.extend_from_slice(&record.sample.timestamp.to_le_bytes());
    out.extend_from_slice(&record.sample.value.to_bits().to_le_bytes());
    out
}

fn decode_records(bytes: &[u8]) -> io::Result<(Vec<WalRecord>, usize)> {
    let mut records = Vec::new();
    let mut position = 0;
    while position < bytes.len() {
        let record_start = position;
        if bytes.len() - position < 4 {
            return Ok((records, record_start));
        }
        let length = u32::from_le_bytes(
            bytes[position..position + 4]
                .try_into()
                .expect("four-byte slice"),
        ) as usize;
        position += 4;
        if length > MAX_RECORD_SIZE || length > bytes.len() - position {
            return Ok((records, record_start));
        }
        let end = position + length;
        let mut reader = Reader {
            bytes: &bytes[position..end],
            position: 0,
        };
        let version = reader.byte()?;
        if version != RECORD_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported WAL record version",
            ));
        }
        let name = reader.string()?;
        let label_count = reader.u32()? as usize;
        if label_count > 100_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unreasonable WAL label count",
            ));
        }
        let mut labels = std::collections::BTreeMap::new();
        for _ in 0..label_count {
            labels.insert(reader.string()?, reader.string()?);
        }
        let timestamp = reader.i64()?;
        let value = f64::from_bits(reader.u64()?);
        if !reader.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing bytes in WAL record",
            ));
        }
        records.push(WalRecord {
            series: Series::new(name, labels),
            sample: Sample::new(timestamp, value),
        });
        position = end;
    }
    Ok((records, position))
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn sync_parent(path: &Path) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "WAL path has no parent directory",
        )
    })?;
    File::open(parent)?.sync_all()
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> io::Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "WAL length overflow"))?;
        if end > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated WAL record",
            ));
        }
        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }

    fn byte(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four-byte slice"),
        ))
    }

    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("eight-byte slice"),
        ))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight-byte slice"),
        ))
    }

    fn string(&mut self) -> io::Result<String> {
        let length = self.u32()? as usize;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 in WAL"))
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn partial_tail_is_discarded() {
        let dir = tempfile_dir();
        let path = dir.join("wal.log");
        let (mut wal, _) = Wal::open(&path).unwrap();
        wal.append(&[WalRecord {
            series: Series::new("up", BTreeMap::new()),
            sample: Sample::new(1, 1.0),
        }])
        .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[1, 2, 3])
            .unwrap();
        drop(wal);
        let (_, records) = Wal::open(&path).unwrap();
        assert_eq!(records.len(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn tempfile_dir() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("gauge-wal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
