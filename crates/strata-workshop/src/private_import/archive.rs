//! Archives in quarantine (zip, tar, tar.gz) unpacked with guards: an entry's name is never used as a path (every
//! item is stored by its SHA-256), and an entry that tries to leave the folder (an absolute name, `..`), a link or a
//! device is refused; the sizes are counted as the bytes come out (never taken from the headers), so a bomb stops at
//! the caps: the total, one file's size, the number of files, and how much one entry may expand.
//!
//! Any guard that trips rejects the whole archive: what it gave so far is removed again.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use sha2::Digest;

/// The caps of one import.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Caps {
    /// all items of the import together
    pub max_total_bytes: u64,
    /// one item
    pub max_file_bytes: u64,
    /// items of the import
    pub max_files: usize,
    /// what one archive entry may expand to, against its compressed size (past the first MiB)
    pub max_ratio: u64,
}

impl Default for Caps {
    fn default() -> Caps {
        Caps { max_total_bytes: 256 << 30, max_file_bytes: 8 << 30, max_files: 200_000, max_ratio: 200 }
    }
}

/// What the archive kinds are, by their first bytes (never by the name alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archive {
    Zip,
    Tar,
    TarGz,
}

pub fn archive_kind(path: &Path) -> Option<Archive> {
    let mut head = [0u8; 512];
    let n = std::fs::File::open(path).and_then(|mut f| read_full(&mut f, &mut head)).ok()?;
    if n >= 4 && &head[..4] == b"PK\x03\x04" {
        return Some(Archive::Zip);
    }
    if n >= 2 && head[0] == 0x1f && head[1] == 0x8b {
        // a gzip: a tar inside it, or one compressed file (not unpacked)
        let name = path.file_name().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
        return (name.ends_with(".tar.gz") || name.ends_with(".tgz") || name.ends_with(".tar.gz.part")).then_some(Archive::TarGz);
    }
    if n >= 262 && &head[257..262] == b"ustar" {
        return Some(Archive::Tar);
    }
    None
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

/// What unpacking gave: the new items (hash, extension, bytes) and the entries refused.
#[derive(Debug, Default)]
pub struct Unpacked {
    pub items: Vec<(String, String, u64)>,
    /// links, devices and other entries that are not plain files
    pub skipped: usize,
}

/// Why an archive was rejected as a whole.
#[derive(Debug, PartialEq)]
pub struct Rejected(pub String);

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The running totals the caps are held against (shared by every archive and file of an import).
#[derive(Debug, Default, Clone, Copy)]
pub struct Totals {
    pub bytes: u64,
    pub files: usize,
}

/// An entry name that would leave the folder (absolute, a drive, `..`), or is empty.
pub fn unsafe_name(name: &str) -> bool {
    let n = name.replace('\\', "/");
    if n.is_empty() || n.starts_with('/') || n.contains('\0') || (n.len() > 1 && n.as_bytes()[1] == b':') {
        return true;
    }
    Path::new(&n).components().any(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_)))
}

/// The extension an item keeps (lower case, letters and digits, at most 8), from the entry's name.
pub fn ext_of(name: &str) -> String {
    let e = Path::new(name).extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
    if !e.is_empty() && e.len() <= 8 && e.chars().all(|c| c.is_ascii_alphanumeric()) { e } else { "bin".into() }
}

/// Copy `r` into `dir/<sha256>.<ext>` under the caps; `compressed` (0: unknown) bounds the expansion. Returns the hash
/// and the bytes, or the guard that tripped (the partial file removed).
pub fn store(r: &mut impl Read, dir: &Path, ext: &str, compressed: u64, caps: &Caps, totals: &mut Totals) -> Result<std::result::Result<(String, u64), Rejected>> {
    if totals.files >= caps.max_files {
        return Ok(Err(Rejected(format!("more than {} files", caps.max_files))));
    }
    let tmp = dir.join(format!(".incoming-{}-{}", std::process::id(), totals.files));
    let mut f = super::create_private(&tmp)?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n: u64 = 0;
    let tripped = loop {
        let k = match r.read(&mut buf) {
            Ok(0) => break None,
            Ok(k) => k,
            Err(e) => break Some(format!("the archive could not be read: {e}")),
        };
        n += k as u64;
        if n > caps.max_file_bytes {
            break Some(format!("a file larger than {} bytes", caps.max_file_bytes));
        }
        if totals.bytes + n > caps.max_total_bytes {
            break Some(format!("more than {} bytes in all", caps.max_total_bytes));
        }
        if compressed > 0 && n > (1 << 20) && n / compressed.max(1) > caps.max_ratio {
            break Some(format!("an entry that expands more than {} times (a decompression bomb)", caps.max_ratio));
        }
        h.update(&buf[..k]);
        f.write_all(&buf[..k]).with_context(|| format!("{}", tmp.display()))?;
    };
    drop(f);
    if let Some(why) = tripped {
        std::fs::remove_file(&tmp).ok();
        return Ok(Err(Rejected(why)));
    }
    let hash: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let dst = dir.join(format!("{hash}.{ext}"));
    if dst.exists() {
        std::fs::remove_file(&tmp).ok(); // the same item twice: kept once
    } else {
        std::fs::rename(&tmp, &dst)?;
    }
    totals.bytes += n;
    totals.files += 1;
    Ok(Ok((hash, n)))
}

/// Unpack `archive` into `dir` (the quarantine's items). On a guard, every item this archive added is removed and the
/// rejection returned.
pub fn unpack(archive: &Path, kind: Archive, dir: &Path, caps: &Caps, totals: &mut Totals) -> Result<std::result::Result<Unpacked, Rejected>> {
    let before: std::collections::BTreeSet<PathBuf> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).collect();
    let saved = *totals;
    let r = match kind {
        Archive::Zip => unzip(archive, dir, caps, totals),
        Archive::Tar => untar(std::fs::File::open(archive)?, dir, caps, totals),
        Archive::TarGz => untar(flate2::read::MultiGzDecoder::new(std::fs::File::open(archive)?), dir, caps, totals),
    };
    match r {
        Ok(Ok(u)) => Ok(Ok(u)),
        other => {
            // what this archive added goes again
            for e in std::fs::read_dir(dir)?.flatten() {
                if !before.contains(&e.path()) {
                    std::fs::remove_file(e.path()).ok();
                }
            }
            *totals = saved;
            match other {
                Ok(Err(r)) => Ok(Err(r)),
                Err(e) => Ok(Err(Rejected(format!("the archive could not be read: {e:#}")))),
                Ok(Ok(_)) => unreachable!(),
            }
        }
    }
}

fn unzip(path: &Path, dir: &Path, caps: &Caps, totals: &mut Totals) -> Result<std::result::Result<Unpacked, Rejected>> {
    let mut z = zip::ZipArchive::new(std::fs::File::open(path)?)?;
    if z.len() > caps.max_files {
        return Ok(Err(Rejected(format!("more than {} files", caps.max_files))));
    }
    let mut out = Unpacked::default();
    for i in 0..z.len() {
        let mut e = z.by_index(i)?;
        let name = e.name().to_string();
        if unsafe_name(&name) {
            return Ok(Err(Rejected("an entry whose name leaves the folder (zip-slip)".into())));
        }
        if e.is_dir() {
            continue;
        }
        // a link (unix mode 0o120000) or anything else that is not a plain file
        if e.unix_mode().is_some_and(|m| m & 0o170000 != 0o100000 && m & 0o170000 != 0) || e.is_symlink() {
            out.skipped += 1;
            continue;
        }
        let compressed = e.compressed_size().max(1);
        let ext = ext_of(&name);
        match store(&mut e, dir, &ext, compressed, caps, totals)? {
            Ok((h, n)) => out.items.push((h, ext, n)),
            Err(r) => return Ok(Err(r)),
        }
    }
    Ok(Ok(out))
}

fn octal(b: &[u8]) -> Option<u64> {
    if b.first().is_some_and(|x| x & 0x80 != 0) {
        // base-256 (GNU, large sizes)
        let mut v: u64 = (b[0] & 0x7f) as u64;
        for x in &b[1..] {
            v = v.checked_mul(256)?.checked_add(*x as u64)?;
        }
        return Some(v);
    }
    let s: String = b.iter().take_while(|c| **c != 0).map(|c| *c as char).collect();
    let s = s.trim();
    if s.is_empty() { Some(0) } else { u64::from_str_radix(s, 8).ok() }
}

fn cstr(b: &[u8]) -> String {
    String::from_utf8_lossy(&b[..b.iter().position(|c| *c == 0).unwrap_or(b.len())]).into_owned()
}

/// A tar stream (ustar, GNU long names, pax sizes and paths): plain files kept, links and devices skipped.
fn untar(mut r: impl Read, dir: &Path, caps: &Caps, totals: &mut Totals) -> Result<std::result::Result<Unpacked, Rejected>> {
    let mut out = Unpacked::default();
    let mut long_name: Option<String> = None;
    let mut pax_size: Option<u64> = None;
    let mut zeros = 0;
    loop {
        let mut h = [0u8; 512];
        let n = read_full(&mut r, &mut h)?;
        if n == 0 {
            break;
        }
        if n < 512 {
            return Ok(Err(Rejected("the tar ends in the middle of a header".into())));
        }
        if h.iter().all(|b| *b == 0) {
            zeros += 1;
            if zeros == 2 { break; }
            continue;
        }
        zeros = 0;
        let size = match pax_size.take() {
            Some(s) => s,
            None => octal(&h[124..136]).ok_or_else(|| anyhow::anyhow!("a tar header with a bad size"))?,
        };
        let flag = h[156];
        let mut name = cstr(&h[0..100]);
        if &h[257..262] == b"ustar" {
            let prefix = cstr(&h[345..500]);
            if !prefix.is_empty() {
                name = format!("{prefix}/{name}");
            }
        }
        if let Some(l) = long_name.take() {
            name = l;
        }
        let padded = size.div_ceil(512) * 512;
        match flag {
            b'L' | b'x' => {
                // GNU long name / pax extended header: read (bounded), then apply to the next entry
                if size > 1 << 20 {
                    return Ok(Err(Rejected("a tar header extension larger than 1 MiB".into())));
                }
                let mut data = vec![0u8; padded as usize];
                if read_full(&mut r, &mut data)? < padded as usize {
                    return Ok(Err(Rejected("the tar ends in the middle of an entry".into())));
                }
                let data = &data[..size as usize];
                if flag == b'L' {
                    long_name = Some(cstr(data));
                } else {
                    for rec in String::from_utf8_lossy(data).lines() {
                        let kv = rec.split_once(' ').map(|x| x.1).unwrap_or("");
                        if let Some(v) = kv.strip_prefix("size=") { pax_size = v.trim().parse().ok(); }
                        if let Some(v) = kv.strip_prefix("path=") { long_name = Some(v.to_string()); }
                    }
                }
                continue;
            }
            b'g' => {
                std::io::copy(&mut (&mut r).take(padded), &mut std::io::sink())?;
                continue;
            }
            _ => {}
        }
        if unsafe_name(&name) {
            return Ok(Err(Rejected("an entry whose name leaves the folder (tar path traversal)".into())));
        }
        match flag {
            b'0' | 0 | b'7' => {
                let ext = ext_of(&name);
                let mut body = (&mut r).take(size);
                match store(&mut body, dir, &ext, 0, caps, totals)? {
                    Ok((hsh, k)) => {
                        if k != size {
                            return Ok(Err(Rejected("the tar ends in the middle of an entry".into())));
                        }
                        out.items.push((hsh, ext, k));
                    }
                    Err(rj) => return Ok(Err(rj)),
                }
                std::io::copy(&mut (&mut r).take(padded - size), &mut std::io::sink())?;
            }
            b'5' => std::io::copy(&mut (&mut r).take(padded), &mut std::io::sink()).map(|_| ())?,
            _ => {
                // hard links, symlinks, devices, fifos: never made
                out.skipped += 1;
                std::io::copy(&mut (&mut r).take(padded), &mut std::io::sink())?;
            }
        }
    }
    Ok(Ok(out))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tar of (name, type flag, body) entries, as a writer would make it (ustar headers, checksums).
    pub fn tar(entries: &[(&str, u8, &[u8])]) -> Vec<u8> {
        let mut out = vec![];
        for (name, flag, body) in entries {
            let mut h = [0u8; 512];
            h[..name.len()].copy_from_slice(name.as_bytes());
            h[100..107].copy_from_slice(b"0000644");
            h[124..135].copy_from_slice(format!("{:011o}", body.len()).as_bytes());
            h[156] = *flag;
            h[257..263].copy_from_slice(b"ustar\0");
            h[263..265].copy_from_slice(b"00");
            h[148..156].copy_from_slice(b"        ");
            let sum: u32 = h.iter().map(|b| *b as u32).sum();
            h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
            out.extend_from_slice(&h);
            out.extend_from_slice(body);
            out.resize(out.len().div_ceil(512) * 512, 0);
        }
        out.resize(out.len() + 1024, 0);
        out
    }

    pub fn zip_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (n, b) in entries {
            w.start_file(*n, o).unwrap();
            w.write_all(b).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pi-arch-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(d.join("items")).unwrap();
        d
    }

    fn items(d: &Path) -> usize {
        std::fs::read_dir(d.join("items")).unwrap().count()
    }

    #[test]
    fn a_zip_and_a_tar_unpack_into_items_named_by_hash() {
        let d = scratch("ok");
        std::fs::write(d.join("a.zip"), zip_of(&[("dir/one.txt", b"lorem ipsum"), ("two.txt", b"dolor sit")])).unwrap();
        assert_eq!(archive_kind(&d.join("a.zip")), Some(Archive::Zip));
        let mut t = Totals::default();
        let u = unpack(&d.join("a.zip"), Archive::Zip, &d.join("items"), &Caps::default(), &mut t).unwrap().unwrap();
        assert_eq!(u.items.len(), 2);
        assert!(d.join("items").join(format!("{}.txt", crate::private_data::sha256_hex(b"lorem ipsum"))).is_file());
        std::fs::write(d.join("b.tar"), tar(&[("x/three.md", b'0', b"amet"), ("link", b'2', b""), ("dev", b'3', b"")])).unwrap();
        assert_eq!(archive_kind(&d.join("b.tar")), Some(Archive::Tar));
        let u = unpack(&d.join("b.tar"), Archive::Tar, &d.join("items"), &Caps::default(), &mut t).unwrap().unwrap();
        assert_eq!((u.items.len(), u.skipped), (1, 2), "links and devices are never made");
        // a gzip'd tar
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(&tar(&[("four.txt", b'0', b"consectetur")])).unwrap();
        std::fs::write(d.join("c.tar.gz"), gz.finish().unwrap()).unwrap();
        assert_eq!(archive_kind(&d.join("c.tar.gz")), Some(Archive::TarGz));
        assert_eq!(unpack(&d.join("c.tar.gz"), Archive::TarGz, &d.join("items"), &Caps::default(), &mut t).unwrap().unwrap().items.len(), 1);
        assert_eq!((t.files, items(&d)), (4, 4));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn zip_slip_and_tar_traversal_reject_the_whole_archive() {
        let d = scratch("slip");
        std::fs::write(d.join("a.zip"), zip_of(&[("fine.txt", b"lorem"), ("../../evil.txt", b"ipsum")])).unwrap();
        let mut t = Totals::default();
        let r = unpack(&d.join("a.zip"), Archive::Zip, &d.join("items"), &Caps::default(), &mut t).unwrap();
        assert!(r.unwrap_err().0.contains("zip-slip"));
        assert_eq!((items(&d), t.files), (0, 0), "what it gave first is removed again");
        assert!(!d.join("evil.txt").exists() && !d.parent().unwrap().join("evil.txt").exists());
        std::fs::write(d.join("b.tar"), tar(&[("/etc/x.txt", b'0', b"lorem")])).unwrap();
        assert!(unpack(&d.join("b.tar"), Archive::Tar, &d.join("items"), &Caps::default(), &mut t).unwrap().is_err());
        for bad in ["../x", "/abs", "a/../../b", "C:\\x", ""] {
            assert!(unsafe_name(bad), "{bad}");
        }
        assert!(!unsafe_name("a/b/c.png"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn bombs_and_too_many_files_stop_at_the_caps() {
        let d = scratch("bomb");
        let zeros = vec![0u8; 8 << 20];
        std::fs::write(d.join("bomb.zip"), zip_of(&[("zeros.txt", &zeros)])).unwrap();
        let mut t = Totals::default();
        let r = unpack(&d.join("bomb.zip"), Archive::Zip, &d.join("items"), &Caps { max_ratio: 100, ..Caps::default() }, &mut t).unwrap();
        assert!(r.unwrap_err().0.contains("bomb"));
        let r = unpack(&d.join("bomb.zip"), Archive::Zip, &d.join("items"), &Caps { max_file_bytes: 1 << 20, ..Caps::default() }, &mut t).unwrap();
        assert!(r.unwrap_err().0.contains("larger than"));
        let r = unpack(&d.join("bomb.zip"), Archive::Zip, &d.join("items"), &Caps { max_total_bytes: 1 << 20, ..Caps::default() }, &mut t).unwrap();
        assert!(r.unwrap_err().0.contains("in all"));
        std::fs::write(d.join("many.zip"), zip_of(&[("1.txt", b"a"), ("2.txt", b"b"), ("3.txt", b"c"), ("4.txt", b"d")])).unwrap();
        let r = unpack(&d.join("many.zip"), Archive::Zip, &d.join("items"), &Caps { max_files: 3, ..Caps::default() }, &mut t).unwrap();
        assert!(r.unwrap_err().0.contains("more than 3 files"));
        // a tar bomb (sizes counted, not trusted)
        std::fs::write(d.join("big.tar"), tar(&[("big.txt", b'0', &zeros)])).unwrap();
        let r = unpack(&d.join("big.tar"), Archive::Tar, &d.join("items"), &Caps { max_file_bytes: 1 << 20, ..Caps::default() }, &mut t).unwrap();
        assert!(r.is_err());
        assert_eq!(items(&d), 0, "nothing of a rejected archive is kept");
        std::fs::remove_dir_all(&d).ok();
    }
}
