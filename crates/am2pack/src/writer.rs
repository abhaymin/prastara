// SPDX-License-Identifier: AGPL-3.0-only
//! The container writer: a header, sections of bytes, then the index. It writes to `<path>.part` and renames at the
//! end, so a pack that exists is a whole one.

use crate::{hex, up, Index, Result, Section, HEADER_LEN, MAGIC, VERSION};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub struct Writer {
    f: BufWriter<File>,
    path: PathBuf,
    part: PathBuf,
    pos: u64,
    sections: Vec<Section>,
    /// the open section's hash
    open: Option<Sha256>,
    done: bool,
}

const ZEROS: [u8; 4096] = [0; 4096];

impl Writer {
    pub fn create(path: &Path) -> Result<Writer> {
        let mut part = path.as_os_str().to_owned();
        part.push(".part");
        let part = PathBuf::from(part);
        let f = File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        let mut w = Writer { f: BufWriter::with_capacity(8 << 20, f), path: path.to_path_buf(), part, pos: 0, sections: Vec::new(), open: None, done: false };
        w.raw(&ZEROS[..HEADER_LEN as usize])?;
        Ok(w)
    }

    fn raw(&mut self, b: &[u8]) -> Result<()> {
        self.f.write_all(b).map_err(|e| format!("{}: {e}", self.part.display()))?;
        self.pos += b.len() as u64;
        Ok(())
    }

    fn zeros(&mut self, mut n: u64, hashed: bool) -> Result<()> {
        while n > 0 {
            let k = n.min(ZEROS.len() as u64) as usize;
            if hashed {
                self.write(&ZEROS[..k])?;
            } else {
                self.raw(&ZEROS[..k])?;
            }
            n -= k as u64;
        }
        Ok(())
    }

    /// Starts a section on a multiple of `align`; returns its index. The padding before it is zeros, outside every
    /// section.
    pub fn begin(&mut self, kind: &str, name: &str, align: u64) -> Result<usize> {
        if self.open.is_some() {
            return Err("a section is still open".into());
        }
        self.zeros(up(self.pos, align) - self.pos, false)?;
        self.sections.push(Section { kind: kind.into(), name: name.into(), off: self.pos, len: 0, sha256: String::new() });
        self.open = Some(Sha256::new());
        Ok(self.sections.len() - 1)
    }

    /// Bytes into the open section.
    pub fn write(&mut self, b: &[u8]) -> Result<()> {
        self.open.as_mut().ok_or("no section is open")?.update(b);
        self.raw(b)
    }

    /// Pads the open section with zeros to a multiple of `align` (of the file offset); returns the offset reached.
    pub fn align(&mut self, align: u64) -> Result<u64> {
        self.zeros(up(self.pos, align) - self.pos, true)?;
        Ok(self.pos)
    }

    /// The file offset the next byte goes to.
    pub fn pos(&self) -> u64 {
        self.pos
    }

    pub fn end(&mut self) -> Result<()> {
        let h = self.open.take().ok_or("no section is open")?;
        let s = self.sections.last_mut().unwrap();
        s.len = self.pos - s.off;
        s.sha256 = hex(&h.finalize());
        Ok(())
    }

    /// One whole section from a slice.
    pub fn section(&mut self, kind: &str, name: &str, align: u64, bytes: &[u8]) -> Result<usize> {
        let i = self.begin(kind, name, align)?;
        self.write(bytes)?;
        self.end()?;
        Ok(i)
    }

    /// Writes the index (its `sections` are set here) and the header, syncs, and renames the file into place.
    pub fn finish(mut self, index: &mut Index) -> Result<u64> {
        if self.open.is_some() {
            return Err("a section is still open".into());
        }
        index.format = "am2aipac".into();
        index.version = VERSION;
        index.sections = std::mem::take(&mut self.sections);
        let json = serde_json::to_vec(index).map_err(|e| e.to_string())?;
        self.zeros(up(self.pos, 4096) - self.pos, false)?;
        let index_off = self.pos;
        self.raw(&json)?;
        let file_len = self.pos;
        let head = header(index_off, json.len() as u64, &Sha256::digest(&json), file_len);
        let io = |e: std::io::Error| format!("{}: {e}", self.part.display());
        self.f.flush().map_err(io)?;
        let f = self.f.get_mut();
        f.seek(SeekFrom::Start(0)).map_err(io)?;
        f.write_all(&head).map_err(io)?;
        f.sync_all().map_err(io)?;
        std::fs::rename(&self.part, &self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.done = true;
        Ok(file_len)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.done {
            let _ = std::fs::remove_file(&self.part);
        }
    }
}

/// The header's 104 meaningful bytes; the rest of its 4 KiB is zero.
pub(crate) fn header(index_off: u64, index_len: u64, index_sha: &[u8], file_len: u64) -> Vec<u8> {
    let mut h = Vec::with_capacity(104);
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&VERSION.to_le_bytes());
    h.extend_from_slice(&(HEADER_LEN as u32).to_le_bytes());
    h.extend_from_slice(&index_off.to_le_bytes());
    h.extend_from_slice(&index_len.to_le_bytes());
    h.extend_from_slice(&file_len.to_le_bytes());
    h.extend_from_slice(index_sha);
    let own = Sha256::digest(&h);
    h.extend_from_slice(&own);
    h
}
