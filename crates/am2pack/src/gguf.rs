// SPDX-License-Identifier: AGPL-3.0-only
//! Just enough of GGUF to pack one: the header's scalar keys, the tensor table and where the data begins.

use crate::Result;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// from the start of the data section
    pub offset: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
}

#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
    pub alignment: u64,
    /// the header's length rounded up to the alignment: tensor offsets count from here
    pub data_start: u64,
    /// scalar keys only; arrays (the vocabulary) are skipped
    pub kv: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
}

const NAMES: [&str; 43] = ["F32", "F16", "Q4_0", "Q4_1", "Q4_2", "Q4_3", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K",
    "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32",
    "I64", "F64", "IQ1_M", "BF16", "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "TQ1_0", "TQ2_0", "IQ4_NL_4_4", "IQ4_NL_4_8",
    "IQ4_NL_8_8", "MXFP4", "NVFP4", "Q4_0_8_8", "Q2_0"];

pub fn type_name(id: u32) -> String {
    NAMES.get(id as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("type{id}"))
}

/// (elements per block, bytes per block), None for an encoding this crate does not know (it is then copied by extent).
pub fn block_geometry(name: &str) -> Option<(u64, u64)> {
    Some(match name {
        "F32" | "I32" => (1, 4), "F16" | "BF16" | "I16" => (1, 2), "F64" | "I64" => (1, 8), "I8" => (1, 1),
        "Q4_0" => (32, 18), "Q4_1" => (32, 20), "Q5_0" => (32, 22), "Q5_1" => (32, 24), "Q8_0" => (32, 34), "Q8_1" => (32, 36),
        "Q2_K" => (256, 84), "Q3_K" => (256, 110), "Q4_K" => (256, 144), "Q5_K" => (256, 176), "Q6_K" => (256, 210),
        "Q8_K" => (256, 292), "IQ2_XXS" => (256, 66), "IQ2_XS" => (256, 74), "IQ3_XXS" => (256, 98), "IQ1_S" => (256, 50),
        "IQ4_NL" => (32, 18), "IQ3_S" => (256, 110), "IQ2_S" => (256, 82), "IQ4_XS" => (256, 136), "IQ1_M" => (256, 56),
        "TQ1_0" => (256, 54), "TQ2_0" => (256, 66), "Q2_0" => (64, 18), "MXFP4" => (32, 17), "NVFP4" => (64, 36),
        _ => return None,
    })
}

impl TensorInfo {
    pub fn type_name(&self) -> String {
        type_name(self.ggml_type)
    }
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn expected_bytes(&self) -> Option<u64> {
        let (be, bb) = block_geometry(&self.type_name())?;
        let n = self.elements();
        (n % be == 0).then(|| n / be * bb)
    }
}

impl Header {
    pub fn str(&self, k: &str) -> Option<&str> {
        match self.kv.get(k) { Some(Value::S(s)) => Some(s), _ => None }
    }
    pub fn u64(&self, k: &str) -> Option<u64> {
        match self.kv.get(k) { Some(Value::U(v)) => Some(*v), Some(Value::I(v)) if *v >= 0 => Some(*v as u64), _ => None }
    }
    pub fn architecture(&self) -> Option<&str> {
        self.str("general.architecture")
    }
}

struct Cur<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.b.len()).ok_or("the GGUF header runs past the end of the file")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U(self.take(1)?[0] as u64),
            1 => Value::I(self.take(1)?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::B(self.take(1)?[0] != 0),
            8 => Value::S(self.string()?),
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => return Err(format!("unknown GGUF value type {t}")),
        })
    }
    /// Steps over a value; returns it when it is a scalar.
    fn value(&mut self, t: u32) -> Result<Option<Value>> {
        if t != 9 {
            return self.scalar(t).map(Some);
        }
        let et = self.u32()?;
        let n = self.u64()?;
        let fixed = match et { 0 | 1 | 7 => 1, 2 | 3 => 2, 4..=6 => 4, 10..=12 => 8, _ => 0 };
        if fixed > 0 {
            self.take(n.checked_mul(fixed).ok_or("a GGUF array is too long")? as usize)?;
        } else {
            for _ in 0..n {
                self.value(et)?;
            }
        }
        Ok(None)
    }
}

/// Parses the header at the start of `file` (the whole file's bytes, or at least its header).
pub fn parse(file: &[u8]) -> Result<Header> {
    let mut c = Cur { b: file, at: 0 };
    if c.take(4)? != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = c.u32()?;
    if version < 2 {
        return Err(format!("GGUF version {version} is too old"));
    }
    let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
    let mut kv = HashMap::new();
    for _ in 0..n_kv {
        let k = c.string()?;
        let t = c.u32()?;
        if let Some(v) = c.value(t)? {
            kv.insert(k, v);
        }
    }
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()? as usize;
        let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push(TensorInfo { name, dims, ggml_type, offset });
    }
    let alignment = match kv.get("general.alignment") { Some(Value::U(a)) if *a > 0 => *a, _ => 32 };
    let data_start = (c.at as u64).div_ceil(alignment) * alignment;
    Ok(Header { version, alignment, data_start, kv, tensors })
}
