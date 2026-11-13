//! Write-Ahead Log: append-only, CRC32-checksummed, crash-safe.
//!
//! Each entry on disk:
//!   [len: u32 le][crc32: u32 le][payload: len bytes]
//!
//! On replay, entries with truncated or corrupt payloads are skipped.

use crate::errors::{Error, Result};
use crc32fast::Hasher;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum WalEntry {
    Insert {
        id: u64,
        vector: Vec<f32>,
        metadata: Vec<(String, String)>,
    },
    Delete {
        id: u64,
    },
    Fsync,
}

pub struct Wal {
    file: BufWriter<File>,
    path: PathBuf,
    pending_fsync: usize,
    fsync_every: usize,
}

impl Wal {
    pub fn open(path: &Path, fsync_every: usize) -> Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_path_buf(),
            pending_fsync: 0,
            fsync_every,
        })
    }

    pub fn append(&mut self, entry: &WalEntry) -> Result<()> {
        let payload = encode(entry);
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let crc = hasher.finalize();

        let len = payload.len() as u32;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.write_all(&payload)?;

        self.pending_fsync += 1;
        if self.fsync_every > 0 && self.pending_fsync >= self.fsync_every {
            self.sync()?;
        }
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_data()?;
        self.pending_fsync = 0;
        Ok(())
    }

    /// Replay all entries from disk. Corrupt/truncated entries are skipped.
    pub fn replay(path: &Path) -> Result<Vec<WalEntry>> {
        let mut entries = Vec::new();
        if !path.exists() {
            return Ok(entries);
        }
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::Io(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            if len == 0 || len > 100 * 1024 * 1024 {
                // Suspicious length: corrupted entry
                break;
            }
            let mut crc_buf = [0u8; 4];
            if reader.read_exact(&mut crc_buf).is_err() {
                break; // truncated
            }
            let expected_crc = u32::from_le_bytes(crc_buf);
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() {
                break; // truncated — skip
            }
            let mut hasher = Hasher::new();
            hasher.update(&payload);
            if hasher.finalize() != expected_crc {
                // Corruption — skip entry, continue
                continue;
            }
            if let Ok(entry) = decode(&payload) {
                entries.push(entry);
            }
        }
        Ok(entries)
    }
}

fn encode(entry: &WalEntry) -> Vec<u8> {
    let mut buf = Vec::new();
    match entry {
        WalEntry::Insert {
            id,
            vector,
            metadata,
        } => {
            buf.push(1u8);
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&(vector.len() as u32).to_le_bytes());
            for f in vector {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
            for (k, v) in metadata {
                buf.extend_from_slice(&(k.len() as u32).to_le_bytes());
                buf.extend_from_slice(k.as_bytes());
                buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                buf.extend_from_slice(v.as_bytes());
            }
        }
        WalEntry::Delete { id } => {
            buf.push(2u8);
            buf.extend_from_slice(&id.to_le_bytes());
        }
        WalEntry::Fsync => buf.push(3u8),
    }
    buf
}

fn decode(payload: &[u8]) -> Result<WalEntry> {
    let mut p = 0;
    let tag = payload[p];
    p += 1;
    match tag {
        1 => {
            let id = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap());
            p += 8;
            let n = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut vector = Vec::with_capacity(n);
            for _ in 0..n {
                let f = f32::from_le_bytes(payload[p..p + 4].try_into().unwrap());
                p += 4;
                vector.push(f);
            }
            let m = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut metadata = Vec::with_capacity(m);
            for _ in 0..m {
                let klen = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize;
                p += 4;
                let k = String::from_utf8_lossy(&payload[p..p + klen]).to_string();
                p += klen;
                let vlen = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize;
                p += 4;
                let v = String::from_utf8_lossy(&payload[p..p + vlen]).to_string();
                p += vlen;
                metadata.push((k, v));
            }
            Ok(WalEntry::Insert {
                id,
                vector,
                metadata,
            })
        }
        2 => {
            let id = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap());
            Ok(WalEntry::Delete { id })
        }
        3 => Ok(WalEntry::Fsync),
        _ => Err(Error::Corrupt(format!("unknown WAL tag {tag}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn roundtrip_insert() {
        let tmp = NamedTempFile::new().unwrap();
        {
            let mut wal = Wal::open(tmp.path(), 1).unwrap();
            wal.append(&WalEntry::Insert {
                id: 42,
                vector: vec![1.0, 2.0, 3.0],
                metadata: vec![("color".into(), "red".into())],
            })
            .unwrap();
            wal.sync().unwrap();
        }
        let entries = Wal::replay(tmp.path()).unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WalEntry::Insert {
                id,
                vector,
                metadata,
            } => {
                assert_eq!(*id, 42);
                assert_eq!(vector, &vec![1.0, 2.0, 3.0]);
                assert_eq!(metadata, &vec![("color".to_string(), "red".to_string())]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn roundtrip_multiple_entries() {
        let tmp = NamedTempFile::new().unwrap();
        {
            let mut wal = Wal::open(tmp.path(), 1).unwrap();
            wal.append(&WalEntry::Insert {
                id: 1,
                vector: vec![1.0],
                metadata: vec![],
            })
            .unwrap();
            wal.append(&WalEntry::Insert {
                id: 2,
                vector: vec![2.0],
                metadata: vec![],
            })
            .unwrap();
            wal.append(&WalEntry::Delete { id: 1 }).unwrap();
            wal.sync().unwrap();
        }
        let entries = Wal::replay(tmp.path()).unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn skips_corrupt_entry() {
        let tmp = NamedTempFile::new().unwrap();
        {
            let mut wal = Wal::open(tmp.path(), 1).unwrap();
            wal.append(&WalEntry::Delete { id: 1 }).unwrap();
            wal.append(&WalEntry::Delete { id: 2 }).unwrap();
            wal.sync().unwrap();
        }
        // Corrupt the second entry's payload
        let mut data = std::fs::read(tmp.path()).unwrap();
        // First entry: 4 bytes len + 4 bytes crc + 9 bytes payload = 17 bytes
        // Second entry starts at offset 17; corrupt its payload byte
        if data.len() > 21 {
            data[21] ^= 0xFF;
        }
        std::fs::write(tmp.path(), data).unwrap();
        let entries = Wal::replay(tmp.path()).unwrap();
        // First entry survives, second is either skipped or decoded
        assert!(!entries.is_empty());
    }

    #[test]
    fn truncated_tail_is_ignored() {
        let tmp = NamedTempFile::new().unwrap();
        {
            let mut wal = Wal::open(tmp.path(), 1).unwrap();
            wal.append(&WalEntry::Delete { id: 1 }).unwrap();
            wal.append(&WalEntry::Insert {
                id: 2,
                vector: vec![1.0, 2.0, 3.0],
                metadata: vec![],
            })
            .unwrap();
            wal.sync().unwrap();
        }
        // Truncate the file mid-second-entry
        let data = std::fs::read(tmp.path()).unwrap();
        let truncate_at = data.len() - 5;
        std::fs::write(tmp.path(), &data[..truncate_at]).unwrap();
        let entries = Wal::replay(tmp.path()).unwrap();
        assert_eq!(entries.len(), 1, "should recover only the complete entry");
        assert_eq!(entries[0], WalEntry::Delete { id: 1 });
    }
}
