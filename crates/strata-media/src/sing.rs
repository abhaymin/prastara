//! M1: syllabify Devanagari text into aksharas marked laghu or guru.
//!
//! Pure Rust, no model, no server, no downloads. A word is split on Unicode code points into
//! clusters; each cluster gets a weight from its vowel (long vs short), any anusvara/visarga, or
//! because it is closed. `matras` turns those weights into beat counts.

use anyhow::{bail, Result};
use serde_json::{json, Value};

/// Consonant: U+0915–U+0939 or U+0958–U+095F.
fn is_consonant(c: char) -> bool {
    matches!(c as u32, 0x0915..=0x0939 | 0x0958..=0x095F)
}

/// Virama (the halving sign): U+094D.
fn is_virama(c: char) -> bool {
    c as u32 == 0x094D
}

/// Vowel sign / matra: U+093E–U+094C plus U+0962, U+0963.
fn is_vowel_sign(c: char) -> bool {
    matches!(c as u32, 0x093E..=0x094C | 0x0962..=0x0963)
}

/// Independent vowel: U+0904–U+0914.
fn is_independent_vowel(c: char) -> bool {
    matches!(c as u32, 0x0904..=0x0914)
}

fn is_nasal(c: char) -> bool {
    matches!(c, 'ँ' | 'ं' | 'ः')
}

/// A vowel sign that makes a syllable long.
fn is_long_vowel_sign(c: char) -> bool {
    matches!(c as u32, 0x093E | 0x0940 | 0x0942 | 0x0944 | 0x0947 | 0x0948 | 0x094B | 0x094C)
}

/// A word boundary: space, दंड (।), अर्द्दंड (॥), comma, period.
fn is_word_break(c: char) -> bool {
    c == ' ' || c == '\u{0964}' || c == '\u{0965}' || c == ',' || c == '.'
}

/// A syllable's weight for scansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Weight {
    #[serde(rename = "L")]
    Laghu,
    #[serde(rename = "G")]
    Guru,
}

impl Weight {
    fn is_guru(self) -> bool {
        matches!(self, Weight::Guru)
    }
}

/// One akshara: the closed-syllable split of a word's text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Syllable {
    pub text: String,
    pub latin: String,
    pub weight: Weight,
    pub word: usize,
}

fn syllable_matras(s: &Syllable) -> usize {
    if s.weight.is_guru() {
        2
    } else {
        1
    }
}

/// The matra count for a line of syllables.
pub fn matras(s: &[Syllable]) -> usize {
    s.iter().map(syllable_matras).sum()
}

fn long_independent_vowel(c: char) -> bool {
    matches!(c as u32, 0x093A | 0x0945 | 0x0946)
}

/// The weight of one syllable's text.
pub fn weight(text: &str) -> Weight {
    let cps: Vec<char> = text.chars().collect();
    if text == "ॐ" {
        return Weight::Guru; // o + m: a long vowel closed by the nasal
    }
    for &c in &cps {
        if is_long_vowel_sign(c) || long_independent_vowel(c) {
            return Weight::Guru;
        }
    }
    for &c in &cps {
        if c == 'ँ' {
            continue; // chandrabindu only nasalises, does not make the syllable guru
        }
        if matches!(c, 'ं' | 'ः') {
            return Weight::Guru;
        }
    }
    if cps.len() >= 2 && is_consonant(cps[cps.len() - 2]) && is_virama(cps[cps.len() - 1]) {
        return Weight::Guru;
    }
    Weight::Laghu
}

/// Collect the trailing coda of a syllable starting its coda at index `j`: a conjunct
/// (consonant + virata) that closes it, then any trailing nasal. Returns the next index.
fn end_coda(cps: &[char], n: usize, mut j: usize) -> usize {
    while j < n {
        if is_consonant(cps[j]) && j + 1 < n && is_virama(cps[j + 1]) {
            j += 2;
        } else if is_nasal(cps[j]) {
            j += 1;
        } else {
            break;
        }
    }
    j
}

/// Split a vowelless run `cps[start..n]` into syllables: a virata-conjunct cluster is the onset,
/// and the first consonant of the following conjunct closes it as coda (word-initial all-onset,
/// and word-final consonant + virata joining the syllable before it).
fn split_vowelless(cps: &[char], start: usize, n: usize, out: &mut Vec<String>) {
    let mut i = start;
    while i < n {
        let mut j = i + 1;
        while j < n && is_virama(cps[j]) && j + 1 < n && is_consonant(cps[j + 1]) {
            j += 2;
        }
        let k = j; // cps[i..j] is the virata-conjunct onset cluster
        if k < n && is_consonant(cps[k]) && k + 1 < n && is_virama(cps[k + 1]) {
            out.push(cps[i..k + 2].iter().collect()); // first-closes prev as coda
            i = k + 2;
        } else {
            let mut jj = j;
            while jj < n && is_nasal(cps[jj]) {
                jj += 1;
            }
            out.push(cps[i..jj].iter().collect());
            i = jj;
        }
    }
}

/// Split one word into akshara texts.
fn split_word(word: &str) -> Vec<String> {
    let cps: Vec<char> = word.chars().collect();
    let n = cps.len();
    if n == 1 && cps[0] == 'ॐ' {
        return vec![word.to_string()];
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cps[i];
        if is_independent_vowel(c) {
            let j = end_coda(&cps, n, i + 1);
            out.push(cps[i..j].iter().collect());
            i = j;
            continue;
        }
        // A consonant whose next character is an explicit vowel sign.
        if i + 1 < n && is_vowel_sign(cps[i + 1]) {
            let j = end_coda(&cps, n, i + 2);
            out.push(cps[i..j].iter().collect());
            i = j;
            continue;
        }
        // A consonant followed by a conjunct leading to an explicit nucleus: the whole conjunct
        // cluster is the onset of that akshara.
        if i + 2 < n && is_virama(cps[i + 1]) && is_consonant(cps[i + 2]) {
            let mut j = i + 1;
            while j < n && is_virama(cps[j]) && j + 1 < n && is_consonant(cps[j + 1]) {
                j += 2;
            }
            let cluster_end = j;
            let mut e = cluster_end;
            while e < n && !(is_vowel_sign(cps[e]) || is_independent_vowel(cps[e])) {
                e += 1;
            }
            if e < n {
                let end = end_coda(&cps, n, e + 1);
                out.push(cps[i..end].iter().collect());
                i = end;
                continue;
            }
        }
        // Is the remainder vowelless? Then split it with the conjunct rules.
        let mut e = i;
        while e < n && !(is_vowel_sign(cps[e]) || is_independent_vowel(cps[e])) {
            e += 1;
        }
        if e >= n {
            split_vowelless(&cps, i, n, &mut out);
            return out;
        }
        // A standalone inherent-a consonant standing before a later explicit vowel.
        let j = end_coda(&cps, n, i + 1);
        out.push(cps[i..j].iter().collect());
        i = j;
    }
    out
}

/// The words of `text`, in order (delimited by spaces and दंड/ardhadanda/comma/period).
fn words_of(text: &str) -> Vec<&str> {
    text.split(|c| is_word_break(c)).filter(|w| !w.is_empty()).collect()
}

/// The romanisation of a single consonant, for captions. Simple, not IAST.
fn consonant_latin(c: char) -> &'static str {
    match c as u32 {
        0x0915 => "k",
        0x0916 => "kh",
        0x0917 => "g",
        0x0918 => "gh",
        0x0919 => "c",
        0x091A => "ch",
        0x091B => "j",
        0x091C => "jh",
        0x091D => "d",
        0x091E => "dh",
        0x091F => "t",
        0x0920 => "th",
        0x0921 => "d",
        0x0922 => "dh",
        0x0923 => "n",
        0x0924 => "t",
        0x0925 => "th",
        0x0926 => "d",
        0x0927 => "dh",
        0x0928 => "n",
        0x0929 => "p",
        0x092A => "ph",
        0x092B => "b",
        0x092C => "bh",
        0x092D => "m",
        0x092E => "m",
        0x092F => "y",
        0x0930 => "r",
        0x0931 => "l",
        0x0932 => "l",
        0x0933 => "l",
        0x0935 => "v",
        0x0936 => "z",
        0x0937 => "sh",
        0x0938 => "s",
        0x0939 => "h",
        0x0958 => "d",
        0x0959 => "dh",
        0x095A => "d",
        0x095B => "dh",
        0x095C => "l",
        0x095D => "th",
        _ => "k",
    }
}

/// The romanisation of a single vowel sign or independent vowel.
fn vowel_latin(c: char) -> &'static str {
    if is_independent_vowel(c) {
        match c as u32 {
            0x0904 => "ri",
            0x0905 => "a",
            0x0906 => "e",
            0x0907 => "i",
            0x0908 => "o",
            0x0909 => "u",
            0x090A => "aa",
            0x090B => "ii",
            0x090C => "ri",
            _ => "a",
        }
    } else if is_vowel_sign(c) {
        match c as u32 {
            0x093E => "aa",
            0x093F => "i",
            0x0940 => "ii",
            0x0941 => "u",
            0x0942 => "uu",
            0x0943 => "ri",
            0x0944 => "rr",
            0x0945 => "ii",
            0x0946 => "uu",
            0x0947 => "e",
            0x0948 => "ai",
            0x0949 => "ai",
            0x094A => "ai",
            0x094B => "o",
            0x094C => "au",
            0x0962 => "ri",
            0x0963 => "uu",
            _ => "a",
        }
    } else {
        "a"
    }
}

/// A simple ASCII romanisation of a syllable's text, for captions.
fn latin(text: &str) -> String {
    let cps: Vec<char> = text.chars().collect();
    let n = cps.len();
    let mut out = String::new();
    let mut i = 0;
    while i < n {
        let c = cps[i];
        if is_consonant(c) {
            out.push_str(consonant_latin(c));
            if i + 1 < n && (is_vowel_sign(cps[i + 1]) || is_independent_vowel(cps[i + 1])) {
                out.push_str(vowel_latin(cps[i + 1]));
                i += 2;
            } else {
                out.push('a');
                i += 1;
            }
        } else if is_virama(c) {
            i += 1;
        } else if c == 'ँ' {
            out.push('n');
            i += 1;
        } else if c == 'ं' {
            out.push('m');
            i += 1;
        } else if c == 'ः' {
            out.push('h');
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Split Devanagari `text` into syllables, each marked laghu or guru.
pub fn syllables(text: &str) -> Vec<Syllable> {
    let mut result = Vec::new();
    let mut word_index = 0usize;
    for w in words_of(text) {
        for t in split_word(w) {
            result.push(Syllable {
                text: t.clone(),
                latin: latin(&t),
                weight: weight(&t),
                word: word_index,
            });
        }
        word_index += 1;
    }
    result
}

/// A score event: a syllable with text, or a rest / breath of some length.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Syllable(SyllableEvent),
    Rest(usize),
    Breath(usize),
}

impl Event {
    /// The event's text, if it is a syllable.
    pub fn text(&self) -> Option<&str> {
        match self {
            Event::Syllable(e) => Some(e.text.as_str()),
            _ => None,
        }
    }

    /// The event's weight, if it is a syllable.
    pub fn weight(&self) -> Option<Weight> {
        match self {
            Event::Syllable(e) => Some(e.weight),
            _ => None,
        }
    }
}

impl<'de> serde::Deserialize<'de> for Event {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(obj) = value.as_object() {
            if obj.contains_key("text") {
                let se = serde_json::from_value(value).map_err(Error::custom)?;
                return Ok(Event::Syllable(se));
            }
            if let Some(n) = obj.get("rest").and_then(|v| v.as_u64()) {
                return Ok(Event::Rest(n as usize));
            }
            if let Some(n) = obj.get("breath").and_then(|v| v.as_u64()) {
                return Ok(Event::Breath(n as usize));
            }
        }
        Err(Error::custom("score event has no text, rest or breath"))
    }
}

/// A syllable row of a score. `beats` is the note's length; `note` the swara (relative to Sa); `glide_to` a target the vowel
/// slides to. Unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct SyllableEvent {
    pub text: String,
    #[serde(default)]
    pub latin: Option<String>,
    pub weight: Weight,
    #[serde(default)]
    pub beats: Option<f64>,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub glide_to: Option<String>,
}

impl SyllableEvent {
    /// The syllable's length in beats when the score omits it (a laghu is one beat, a guru two).
    pub fn beats(&self) -> f64 { self.beats.unwrap_or_else(|| match self.weight { Weight::Guru => 2.0, Weight::Laghu => 1.0 }) }
}

/// A word's events (indices into `Score.events`): membership used to place the word gap after its last syllable.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Word {
    pub text: String,
    #[serde(default)]
    pub latin: Option<String>,
    #[serde(default)]
    pub events: Vec<usize>,
}

/// The gap after each word's last syllable: a quarter beat here so words never run together.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct WordGap {
    pub beats: f64,
}

/// A singing score: its ordered events, tempo (bpm), the words and the gap between them. Unknown top-level fields are ignored.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Score {
    #[serde(default)]
    pub events: Vec<Event>,
    #[serde(default)]
    pub bpm: f64,
    #[serde(default)]
    pub words: Vec<Word>,
    #[serde(default)]
    pub word_gap: Option<WordGap>,
}

impl Score {
    /// The tempo in beats per minute (168 is the example's pace), so one beat is 60/bpm seconds.
    pub fn bpm(&self) -> f64 { if self.bpm > 0.0 { self.bpm } else { 168.0 } }
}

impl Score {
    /// The events that have a `text`, in order.
    pub fn syllables(&self) -> Vec<&Event> {
        self.events.iter().filter(|e| e.text().is_some()).collect()
    }
}

/// Load and parse a score from `path`.
pub fn load_score(path: &str) -> anyhow::Result<Score> {
    let data = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}

// M0: `sing-check` — measure one sung take against its score, per syllable. Pure Rust, no model, no FFT: it reuses the
// pitch picker from analyze.rs (a YIN-style autocorrelation) and measures loudness with plain frame RMS. It turns the
// score into time slots, lines up each voice's fundamental against the note the score asks for, times the vowel onto the
// beat and flags where the singer left a silence inside a word or cut a syllable instead of fading it (SINGING-PLAN: the
// voice is linked, and every word is clear).

/// The pitch-track frame length in seconds (analyze.rs's 10 ms frame).
const FRAME_S: f64 = 0.01;
/// The loudness frame length in seconds for the quiet-run and cut checks (20 ms).
const FRAME_RMS_S: f64 = 0.02;

/// A swara name's semitone above Sa (Sa 0, re 1, Re 2, ga 3, Ga 4, ma 5, Ma 6, Pa 7, dha 8, Dha 9, ni 10, Ni 11); a
/// trailing `_` lowers it an octave (`Dha_`), matching the score's note names.
fn swara_semitone(name: &str) -> i32 {
    let (base, down) = match name.strip_suffix('_') { Some(r) => (r, true), None => (name, false) };
    let semi = match base {
        "Sa" => 0, "re" => 1, "Re" => 2, "ga" => 3, "Ga" => 4, "ma" => 5, "Ma" => 6,
        "Pa" => 7, "dha" => 8, "Dha" => 9, "ni" => 10, "Ni" => 11, _ => 0,
    };
    if down { semi - 12 } else { semi }
}

/// A note's frequency in hertz: the swara's semitone above `tonic_hz` (Sa), an octave lower when its name ends in `_`.
fn note_frequency(name: &str, tonic_hz: f64) -> f64 { tonic_hz * 2.0f64.powf(swara_semitone(name) as f64 / 12.0) }

/// The cents of a frequency relative to `reference` (positive = sharp).
fn cents_above(freq: f64, reference: f64) -> f64 { 1200.0 * (freq / reference).log2() }

/// A syllable's target pitch error is measured against its note for the first half of the slot and against `glide_to` for
/// the last quarter, linear between — so a take that glides exactly to the written note reads zero cents throughout.
fn glide_ref_cents(glide_to: Option<&str>, note: &str, frac: f64) -> f64 {
    let Some(gt) = glide_to else { return 0.0 };
    let g = 120.0 * (note_frequency(gt, 1.0) / note_frequency(note, 1.0)).log2(); // tonic-independent ratio
    if frac <= 0.5 { 0.0 } else if frac <= 0.75 { g * (frac - 0.5) / 0.25 } else { g }
}

/// The start and end (seconds into the take) of every event's slot, in order.
///
/// Events start at `start_s`; each syllable lasts its beats and each rest/breath its count, one beat being 60/bpm seconds.
/// A word gap is added only after a word's LAST event, so it never moves the events inside that word — every later event
/// (the next word and on) starts later by the accumulated gap. Hence for event i:
///   t0_i = start_s + sum_{j<i} duration_j + word_gap_beats * beat * (#words ending strictly before i),
/// a constant advance per boundary, which the walk below realises as it goes (bpm defaults to 168 like the example).
fn event_windows(score: &Score, start_s: f64) -> Vec<(f64, f64)> {
    let beat = 60.0 / score.bpm().max(f64::MIN_POSITIVE); // one beat is 60/bpm seconds, so slot times are in real seconds
    let mut word_last = vec![false; score.events.len()];
    for w in &score.words { if let Some(&last) = w.events.iter().max() { word_last[last] = true; } }
    let gap_s = score.word_gap.as_ref().map(|g| g.beats * beat).unwrap_or(0.0);
    let mut wins = Vec::with_capacity(score.events.len());
    let mut t = start_s;
    for (i, e) in score.events.iter().enumerate() {
        let dur = match e { Event::Syllable(ev) => ev.beats() * beat, Event::Rest(n) | Event::Breath(n) => *n as f64 * beat };
        wins.push((t, t + dur));
        t += dur;
        if word_last[i] { t += gap_s; }
    }
    wins
}

/// The RMS of each 20 ms frame across the whole take: `(center_time_s, rms)`, mono samples.
fn rms_frames(pcm: &[f32], rate: f64) -> Vec<(f64, f32)> {
    let hop = (rate * FRAME_RMS_S).round().max(1.0) as usize;
    if pcm.len() < hop { return vec![]; }
    (0..pcm.len() / hop)
        .map(|i| (i as f64 * hop as f64 / rate, rms_of(&pcm[i * hop..((i + 1) * hop).min(pcm.len())])))
        .collect()
}

fn rms_of(w: &[f32]) -> f32 { (w.iter().map(|v| v * v).sum::<f32>() / w.len().max(1) as f32).sqrt() }

/// The p-th percentile of an f32 slice, sorted on entry.
fn percentile_f32(mut x: Vec<f32>, p: f64) -> f32 {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if x.is_empty() { return 0.0; }
    let rank = p * (x.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi { x[lo] } else { x[lo] + (x[hi] - x[lo]) * ((rank - lo as f64) as f32) }
}

/// The median of an f64 slice, sorted on entry (the cents / onset errors).
fn median_f64(x: &mut [f64]) -> f64 {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = x.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 { x[n / 2] } else { (x[n / 2 - 1] + x[n / 2]) / 2.0 }
}

fn r2(x: f64) -> f64 { (x * 100.0).round() / 100.0 }

/// Check one take against its score: per-syllable pitch error, vowel onset and voiced share, plus the faults that break the
/// link between words (SINGING-PLAN). Returns the events, the faults and a summary (`median_abs_cents`, `median_abs_onset_ms`,
/// fault count, syllable count). Runs clean on an empty or short take.
pub fn check(pcm: &[f32], rate: u32, score: &Score, start_s: f64, tonic_hz: f64) -> Value {
    let rate = rate as f64;
    if pcm.is_empty() {
        return json!({"events": Vec::<Value>::new(), "faults": Vec::<Value>::new(),
            "summary": {"median_abs_cents": 0.0, "median_abs_onset_ms": 0.0, "faults": 0, "syllables": 0}});
    }
    let wins = event_windows(score, start_s);
    // the pitch track (fundamental per 10 ms) and loudness frames (RMS per 20 ms) over the whole take; a frame's time is its
    // center, so slots (absolute seconds into the take) line up with them.
    let frames = crate::analyze::fundamental_frames(pcm, rate);   // (Option<hz>, rms) at FRAME_S
    let rms = rms_frames(pcm, rate);                              // (t, rms) at FRAME_RMS_S

    // a frame is "quiet" when its RMS is under 10% of the take's loudness (the 95th percentile): that is what splits a real
    // silence from a held note for the inside-a-word gap check.
    let quiet_thr = 0.1 * percentile_f32(rms.iter().map(|(_, r)| *r).collect::<Vec<f32>>(), 0.95);

    let mut rows = Vec::new();
    let mut faults = Vec::new();
    let mut pitch_errs = Vec::new();     // every middle-60% frame's |cents error|, for the summary
    let mut onsets = Vec::new();
    let mut syllable_count = 0usize;

    for (i, e) in score.events.iter().enumerate() {
        if e.text().is_none() { continue; }     // no row for a rest or a breath
        let ev = match e { Event::Syllable(s) => s, _ => unreachable!() };
        syllable_count += 1;
        let (t0, t1) = wins[i];
        let dur = (t1 - t0).max(FRAME_S);
        let note_hz = note_frequency(&ev.note, tonic_hz);

        // walk the pitch frames that fall in this slot: measure the ones in the middle 60% against the written pitch (which
        // glides), take the first voiced frame near the beat for the onset, and count how much of the slot is voiced.
        let n = frames.len();
        let mut mid_cents = Vec::new();
        let mut voiced_in_slot = 0usize;
        let mut total_frames = 0usize;
        let mut onset_ms: Option<f64> = None;
        let mut med_rms = 0.0f64;
        {
            let mut slot_rms = Vec::new();
            for fi in 0..n {
                let tc = (fi as f64 + 0.5) * FRAME_S;      // frame center, absolute seconds into the take
                if !(t0 <= tc && tc < t1) { continue; }
                total_frames += 1;
                slot_rms.push(rms.iter().find_map(|(t, r)| (*t >= t0 && *t < t1).then_some(*r)).unwrap_or(0.0));
                if let Some(freq) = frames[fi].0 {
                    voiced_in_slot += 1;
                    let frac = (tc - t0) / dur;
                    if 0.2 <= frac && frac < 0.8 {      // the middle 60% of the slot
                        let err = cents_above(freq, note_hz).abs() - glide_ref_cents(ev.glide_to.as_deref(), &ev.note, frac).abs();
                        mid_cents.push(err.max(0.0));
                    }
                }
            }
            if !slot_rms.is_empty() {
                let mut m = slot_rms.iter().map(|v| *v as f64).collect::<Vec<f64>>();
                med_rms = median_f64(&mut m);
            }
        }
        // the first voiced frame within +/-200 ms of the beat, minus t0 (None if none is there)
        if onset_ms.is_none() {
            for fi in 0..n {
                let tc = (fi as f64 + 0.5) * FRAME_S;
                if !(t0 - 0.2 <= tc && tc <= t0 + 0.2) { continue; }
                if frames[fi].0.is_some() { onset_ms = Some((tc - t0) * 1000.0); break; }
            }
        }
        let voiced_share = if total_frames > 0 { voiced_in_slot as f64 / total_frames as f64 } else { 0.0 };
        let pitch_cents = if mid_cents.is_empty() { Value::Null } else { json!(r2(median_f64(&mut mid_cents))) };
        pitch_errs.extend(mid_cents);
        onsets.push(onset_ms.map(|v| v.abs()).unwrap_or(0.0));

        // a syllable with less than thirty percent of its slot voiced is effectively silent — the vowel was never sung.
        if voiced_share < 0.3 {
            faults.push(json!({"kind": "silent_syllable", "t": r2(t0), "text": ev.text}));
        }

        rows.push(json!({"text": ev.text, "t0": r2(t0), "t1": r2(t1), "note_hz": r2(note_hz),
            "pitch_cents": pitch_cents, "onset_ms": onset_ms.map(|v| r2(v)), "voiced_share": r2(voiced_share)}));

        // word-level faults: measured against the last syllable of this word (its neighbor into the gap).
        let wlast = win_last_index(&score, i);
        if wlast {
            let med_last = med_rms;
            // hard cut: loud enough right up to the end and then under 5% a frame (20 ms) later — it should have faded.
            for k in rms_range(&rms, t0, t1) {
                if k + 1 < rms.len() && rms[k].1 > 0.5 * med_last as f32 && rms[k + 1].1 < 0.05 * med_last as f32 {
                    faults.push(json!({"kind": "hard_cut", "t": r2(rms[k].0), "text": word_text(&score, i)}));
                    break;
                }
            }
            // a word gap with no clear dip: the quietest 40 ms across the boundary must be at least 9 dB below the two neighbours.
            if score.word_gap.is_some() {
                let next = win_first_index(&score, i);
                if next < wins.len() {
                    let (gstart, gend) = (t1, wins[next].0);
                    let med_neigh = rms_median_between(&rms, t0, t1).max(1e-9) + rms_median_between(&rms, wins[next].0, wins[next].1).max(1e-9);
                    let quietest = rms_quietest_40ms(&rms, gstart, gend);
                    if med_neigh > 0.0 && (20.0 * ((quietest as f64 / med_neigh).max(1e-9)).log10()) > -9.0 {
                        faults.push(json!({"kind": "no_word_gap", "t": r2(gstart), "text": word_text(&score, i)}));
                    }
                }
            }
        }
    }

    // inside-a-word gap: a quiet run longer than 60 ms between a word's first and last syllable.
    for w in &score.words {
        if w.events.is_empty() { continue; }
        let first = w.events[0];
        let last = *w.events.iter().max().unwrap();
        let (a, b) = (wins[first].0, wins[last].1);
        if let Some((rt, run)) = longest_quiet_run(&rms, a, b, quiet_thr) {
            if run > 0.06 { faults.push(json!({"kind": "gap_in_word", "t": r2(rt), "text": w.text.clone()})); }
        }
    }

    json!({"events": rows, "faults": faults, "summary": {
        "median_abs_cents": r2(median_f64(&mut pitch_errs)),
        "median_abs_onset_ms": r2(median_f64(&mut onsets)),
        "faults": faults.len(),
        "syllables": syllable_count}})
}

/// Whether event index `i` is the last event of its word (used to place the word gap after it).
fn win_last_index(score: &Score, i: usize) -> bool { score.words.iter().any(|w| w.events.last() == Some(&i)) }

/// The index of the first event of the word that follows `i` (the next word's first syllable), for the neighbor RMS.
fn win_first_index(score: &Score, i: usize) -> usize {
    score.words.iter()
        .filter(|w| w.events.iter().all(|&e| e > i))   // words whose events all come after `i`
        .map(|w| *w.events.first().expect("word has events"))
        .min()
        .unwrap_or(i + 1)
}

/// The text of the word that owns event index `i` (for a fault reported at one of its boundaries).
fn word_text(score: &Score, i: usize) -> String {
    score.words.iter().find(|w| w.events.contains(&i)).map(|w| w.text.clone()).unwrap_or_else(|| score.events[i].text().unwrap_or("").to_string())
}

/// The rms-frame indices whose center lies in [lo, hi].
fn rms_range(rms: &[(f64, f32)], lo: f64, hi: f64) -> Vec<usize> {
    (0..rms.len()).filter(|&k| lo <= rms[k].0 && rms[k].0 < hi).collect()
}

/// The median RMS over frames whose center lies in [lo, hi].
fn rms_median_between(rms: &[(f64, f32)], lo: f64, hi: f64) -> f64 {
    let mut v = rms_range(rms, lo, hi).iter().map(|&k| rms[k].1 as f64).collect::<Vec<f64>>();
    median_f64(&mut v)
}

/// The quietest 40 ms (mean of two adjacent frames) entirely within [lo, hi], for the word-gap dip check.
fn rms_quietest_40ms(rms: &[(f64, f32)], lo: f64, hi: f64) -> f32 {
    let mut best = f32::MAX;
    for k in 0..rms.len().saturating_sub(1) {
        let c0 = rms[k].0;
        if c0 >= lo && (rms[k + 1].0) <= hi {
            let mean = (rms[k].1 + rms[k + 1].1) / 2.0;
            if mean < best { best = mean; }
        }
    }
    if best == f32::MAX { 0.0 } else { best }
}

/// The (center, length_s) of the longest run of consecutive quiet frames (< `thr`) within [lo, hi], or None.
fn longest_quiet_run(rms: &[(f64, f32)], lo: f64, hi: f64, thr: f32) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    let mut run_start: Option<f64> = None;
    let mut run_len = 0.0f64;
    for &(t, r) in rms {
        if t >= lo && t < hi && r < thr {
            if run_start.is_none() { run_start = Some(t); }
            run_len += FRAME_RMS_S;
            let len = run_len;
            if let Some((_, bl)) = best { if len > bl { best = Some((run_start.unwrap(), len)); } }
        } else { run_start = None; run_len = 0.0; }
    }
    best
}

// M2: split a spoken word into its syllables, and tell a steady hum from speech. It finds exactly one energy peak
// (vowel nucleus) per text with the boundaries between them in the quiet dip (the consonant) that separates two vowels,
// and scores how steady a take is so a hum can be retaken. Pure Rust, no model, no FFT: it reuses analyze.rs's pitch
// picker for voicing and takes loudness and zero crossings from plain frames.

/// One syllable of a word as split from its audio: where the syllable starts and ends, and the voiced vowel between them.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cut {
    pub text: String,
    pub start_s: f64,
    pub vowel_start_s: f64,
    pub vowel_end_s: f64,
    pub end_s: f64,
}

/// A frame is silence below this fraction of the word's loudest frame, so the word's own span starts at its first loud frame.
const SILENCE_FRAC: f64 = 0.03;
/// How far a hum's per-frame pitch may sit from its median (cents) to still count as held.
const HUM_PITCH_CENTS: f64 = 30.0;
/// How much a hum's per-frame loudness may sit from its median ratio to still count as steady.
const HUM_AMPLITUDE_FRAC: f64 = 0.25;

/// The loudness frame used to sample a hum's pitch and crossing rate (20 ms).
const HUM_FRAME_S: f64 = 0.02;

/// Vowel-ness per 10 ms frame: the loudness between 250 Hz and 1.5 kHz, weighted by its share of the frame's loudness, smoothed over 30 ms.
/// A vowel is loud and low; a fricative or a burst is mostly high; a closure is quiet. It needs no pitch, so a creaky vowel
/// (which Parler voices often end in, and which a pitch tracker drops) still counts.
fn vowelness(pcm: &[f32], rate: f64) -> Vec<f64> {
    // two one-pole low-passes twice each: below 1.5 kHz, and below 250 Hz (the voice's fundamental and room rumble)
    let low = |fc: f64| {
        let a = (-2.0 * std::f64::consts::PI * fc / rate).exp();
        let mut y: Vec<f64> = pcm.iter().map(|&v| v as f64).collect();
        for _ in 0..2 {
            let mut p = 0.0;
            for v in y.iter_mut() { p = (1.0 - a) * *v + a * p; *v = p; }
        }
        y
    };
    let l15 = low(1500.0);
    let l25 = low(250.0);
    // the vowel band, 250-1500 Hz: a vowel's first formants; a nasal's murmur and a breath's rumble lie below it
    let lp: Vec<f64> = l15.iter().zip(&l25).map(|(a, b)| a - b).collect();
    let hop = ((rate * FRAME_S).round() as usize).max(1);
    let raw: Vec<f64> = (0..pcm.len() / hop).map(|i| {
        let (mut el, mut eh) = (0.0, 0.0);
        for j in i * hop..(i + 1) * hop {
            el += lp[j] * lp[j];
            let h = pcm[j] as f64 - l15[j];
            eh += h * h;
        }
        let (el, eh) = ((el / hop as f64).sqrt(), (eh / hop as f64).sqrt());
        el * el / (el + eh + 1e-9)
    }).collect();
    (0..raw.len()).map(|i| {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(raw.len().saturating_sub(1));
        raw[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64
    }).collect()
}

/// How strongly the expected place of each vowel (by syllable weight) counts against loudness when choosing nuclei.
const LAMBDA: f64 = 0.5;
/// A vowel nucleus is at least this share of the word's loudest vowel.
const NUCLEUS_FLOOR: f64 = 0.1;

/// The local maxima of `v` in [s0, s1] at or above `floor` of the peak, each with its prominence: how far it stands above the
/// higher of the lowest points separating it from a taller peak (or from the edges).
fn nucleus_candidates(v: &[f64], s0: usize, s1: usize, floor: f64) -> Vec<(usize, f64)> {
    let pk = v.iter().cloned().fold(0f64, f64::max);
    let mut c = Vec::new();
    for i in s0..=s1 {
        let l = if i > s0 { v[i - 1] } else { 0.0 };
        let r = if i < s1 { v[i + 1] } else { 0.0 };
        if v[i] < floor * pk || v[i] < l || v[i] <= r { continue; }
        let side = |dir: i64| {
            let (mut j, mut m) = (i as i64, v[i]);
            loop {
                j += dir;
                if j < s0 as i64 || j > s1 as i64 { return 0.0; }
                let x = v[j as usize];
                if x > v[i] { return m; }
                m = m.min(x);
            }
        };
        c.push((i, v[i] - side(-1).max(side(1))));
    }
    c
}

/// Exactly `weights.len()` vowel nuclei in time order, at least 50 ms apart, chosen by dynamic programming to be prominent
/// and near where the syllable weights expect them (laghu 1, guru 2, spread over the word's loud part: up to where 97% of
/// its vowel energy has passed, so a faint tail neither holds a nucleus nor stretches the places). None when there are too few.
fn pick_nuclei(v: &[f64], s0: usize, s1: usize, weights: &[f64], lambda: f64, floor: f64) -> Option<Vec<usize>> {
    let n = weights.len();
    let pk = v.iter().cloned().fold(0f64, f64::max).max(1e-12);
    let total: f64 = v[s0..=s1].iter().sum::<f64>().max(1e-12);
    let mut acc = 0.0;
    let mut e1 = s1;
    for i in s0..=s1 { acc += v[i]; if acc >= 0.97 * total { e1 = i; break; } }
    let tot_w: f64 = weights.iter().sum::<f64>().max(1e-9);
    let mut cum = 0.0;
    let exp: Vec<f64> = weights.iter().map(|w| { let e = s0 as f64 + (e1 - s0) as f64 * (cum + w / 2.0) / tot_w; cum += w; e }).collect();
    let span = ((e1 - s0) as f64).max(1.0);
    let c = nucleus_candidates(v, s0, e1, floor);
    let m = c.len();
    if m < n { return None; }
    const NEG: f64 = -1e18;
    let mut dp = vec![vec![NEG; m]; n];
    let mut back = vec![vec![usize::MAX; m]; n];
    for (j, &(i, p)) in c.iter().enumerate() { dp[0][j] = p / pk - lambda * (i as f64 - exp[0]).abs() / span; }
    for k in 1..n {
        for (j, &(i, p)) in c.iter().enumerate() {
            let mut best = (NEG, usize::MAX);
            for jj in 0..j {
                if c[jj].0 + 5 <= i && dp[k - 1][jj] > best.0 { best = (dp[k - 1][jj], jj); }
            }
            if best.1 != usize::MAX { dp[k][j] = best.0 + p / pk - lambda * (i as f64 - exp[k]).abs() / span; back[k][j] = best.1; }
        }
    }
    let (mut j, best) = dp[n - 1].iter().enumerate().fold((0, NEG), |a, (j, &x)| if x > a.1 { (j, x) } else { a });
    if best <= NEG / 2.0 { return None; }
    let mut nu = vec![0; n];
    for k in (0..n).rev() { nu[k] = c[j].0; j = back[k][j]; }
    Some(nu)
}

/// How far to trust a split, 0..1: the share of nuclei that stay within 20 ms when the choice's two settings are moved
/// (lambda 0.3/0.8, floor 0.07/0.15), times whether the prominent peaks found without knowing the count (prominence over a
/// quarter of the peak) are as many as the syllables. A take below 0.6 is better spoken again than sung.
fn split_confidence(v: &[f64], s0: usize, s1: usize, weights: &[f64], nuclei: &[usize]) -> f64 {
    let pk = v.iter().cloned().fold(0f64, f64::max).max(1e-12);
    let mut same = 0usize;
    let mut tried = 0usize;
    for (l, f) in [(0.3, NUCLEUS_FLOOR), (0.8, NUCLEUS_FLOOR), (LAMBDA, 0.07), (LAMBDA, 0.15)] {
        tried += nuclei.len();
        if let Some(o) = pick_nuclei(v, s0, s1, weights, l, f) {
            same += o.iter().zip(nuclei).filter(|(a, b)| a.abs_diff(**b) <= 2).count();
        }
    }
    let stable = same as f64 / tried.max(1) as f64;
    let free = nucleus_candidates(v, s0, s1, NUCLEUS_FLOOR).iter().filter(|(_, p)| *p >= 0.25 * pk).count();
    let count = if free == nuclei.len() { 1.0 } else { 0.75 };
    (stable * count * 100.0).round() / 100.0
}

/// A syllable's boundaries in frames; its vowel is the voiced stretch around its nucleus above half peak.
fn build_cuts(pcm: &[f32], rate: f64, texts: &[&str], weights: &[f64]) -> Result<Split> {
    let n = texts.len();
    if n == 0 { bail!("split: no syllables to find"); }
    let v = vowelness(pcm, rate);
    if v.len() < 3 { bail!("split: the audio is too short to split"); }
    let peak = v.iter().cloned().fold(0f64, f64::max);
    if peak <= 1e-6 { bail!("split: no voice in the audio to find syllables"); }
    let silent = peak * SILENCE_FRAC;
    let start_frame = v.iter().position(|&e| e > silent).unwrap_or(0);
    let end_frame = v.iter().rposition(|&e| e > silent).unwrap_or(v.len() - 1).max(start_frame);
    let en = &v;

    let nuclei = pick_nuclei(&v, start_frame, end_frame, weights, LAMBDA, NUCLEUS_FLOOR)
        .ok_or_else(|| anyhow::anyhow!("split: found fewer vowels than the {n} syllables"))?;
    // each vowel: the run around its nucleus above half its own peak
    let mut vowels: Vec<(usize, usize)> = Vec::new();
    for (k, &nu) in nuclei.iter().enumerate() {
        let lo_lim = if k > 0 { nuclei[k - 1] + 1 } else { start_frame };
        let hi_lim = if k + 1 < n { nuclei[k + 1] - 1 } else { end_frame };
        let half = 0.5 * v[nu];
        let mut vs = nu;
        while vs > lo_lim && v[vs - 1] > half { vs -= 1; }
        let mut ve = nu;
        while ve < hi_lim && v[ve + 1] > half { ve += 1; }
        vowels.push((vs, ve));
    }
    // vowels that touch share their boundary at the lowest point between the two nuclei
    for k in 0..n.saturating_sub(1) {
        if vowels[k].1 + 1 >= vowels[k + 1].0 {
            let (a, b) = (nuclei[k], nuclei[k + 1]);
            let m = (a..=b).min_by(|&x, &y| en[x].partial_cmp(&en[y]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
            vowels[k].1 = m.saturating_sub(1).max(vowels[k].0);
            vowels[k + 1].0 = (m + 1).min(vowels[k + 1].1);
        }
    }
    // a syllable starts where the vowel before it ends: its consonant is in front of its own vowel
    let time = |i: usize| r2(i as f64 * FRAME_S);
    let mut cuts = Vec::new();
    for k in 0..n {
        let s0 = if k == 0 { start_frame } else { vowels[k - 1].1 + 1 };
        let s1 = if k + 1 == n { end_frame + 1 } else { vowels[k].1 + 1 };
        cuts.push(Cut { text: texts[k].to_string(), start_s: time(s0), vowel_start_s: time(vowels[k].0),
                        vowel_end_s: time(vowels[k].1 + 1), end_s: time(s1) });
    }
    let confidence = split_confidence(&v, start_frame, end_frame, weights, &nuclei);
    Ok(Split { cuts, confidence })
}

/// A spoken word split into syllables, and how far to trust the split (0..1; below `SPLIT_TRUST` speak the word again).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Split {
    pub cuts: Vec<Cut>,
    pub confidence: f64,
}

/// The split confidence below which a take is better spoken again (another sub-seed) than sung.
pub const SPLIT_TRUST: f64 = 0.6;

/// Split a spoken word's audio into its syllables (`texts`): one vowel nucleus per text, chosen to be prominent in the vowel
/// band (250-1500 Hz) and near where the syllable weights expect it; each syllable starts where the vowel before it ends,
/// so its consonant is in front of its own vowel. Errors when there is no voice or fewer vowels than syllables.
pub fn split(pcm: &[f32], rate: u32, texts: &[&str]) -> Result<Split> {
    if pcm.is_empty() { bail!("split: the audio is empty"); }
    let weights: Vec<f64> = texts.iter().map(|t| match weight(t) { Weight::Guru => 2.0, Weight::Laghu => 1.0 }).collect();
    build_cuts(pcm, rate as f64, texts, &weights)
}

/// The 20 ms frames of a take: each frame's pitch (if voiced), its loudness and its zero-crossing rate.
fn hum_frames(pcm: &[f32], rate: f64) -> Vec<(Option<f64>, f64, f64)> {
    let hop = (rate * HUM_FRAME_S).round() as usize;
    if pcm.len() < hop || hop == 0 { return vec![]; }
    (0..pcm.len() / hop)
        .map(|i| {
            let w = &pcm[i * hop..((i + 1) * hop).min(pcm.len())];
            let rms = rms_of(w) as f64;
            let zc: usize = w.windows(2).filter(|p| (p[0] >= 0.0) != (p[1] >= 0.0)).count();
            let pitch = crate::analyze::fundamental_frames(w, rate).into_iter().rev().find_map(|(f, _)| f);
            (pitch, rms, zc as f64 / w.len().max(1) as f64)
        })
        .collect()
}

/// How steady the take is as a hum: the share of frames that hold one pitch and one loudness. A spoken word with consonants
/// (unvoiced gaps, shifting pitch and volume) scores low; a single held note scores near one.
pub fn hum_score(pcm: &[f32], rate: u32) -> f64 {
    let frames = hum_frames(pcm, rate as f64);
    if frames.is_empty() { return 0.0; }
    let voiced: Vec<(f64, f64)> = frames.iter().filter_map(|(p, e, _)| p.map(|hz| (hz, *e))).collect();
    if voiced.is_empty() { return 0.0; }
    let mut pitches: Vec<f64> = voiced.iter().map(|(h, _)| *h).collect();
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = pitches[pitches.len() / 2];
    let amps: Vec<f64> = voiced.iter().map(|(_, e)| *e).collect();
    let a_med = median_clamp(&amps);
    // how much the pitch spreads across the whole take (a steady hum barely shifts; a word steps between vowels).
    let min_p = pitches.first().copied().unwrap();
    let max_p = pitches.last().copied().unwrap();
    let pitch_spread = cents_above(max_p, med).abs().max(cents_above(min_p, med).abs());
    // a frame counts when its pitch is within one tolerance of the median and its loudness barely shifts.
    let tr = HUM_PITCH_CENTS.max(pitch_spread * 0.25);
    let ok = frames.iter().filter(|(p, e, _)| {
        let Some(hz) = p else { return false; };
        cents_above(*hz, med).abs() <= tr && ((e / a_med.max(1e-9)) - 1.0).abs() <= HUM_AMPLITUDE_FRAC
    }).count();
    (ok as f64 / frames.len() as f64).clamp(0.0, 1.0)
}

/// Whether the take is a hum: most of its frames hold one pitch and one loudness with no consonant gaps to split it.
/// Measured on real Parler takes (2026-11-21): six hums scored 0.78-0.90 and nine spoken words 0.23 or less.
pub fn is_hum(pcm: &[f32], rate: u32) -> bool { hum_score(pcm, rate) >= HUM_AT }

/// The hum score at and above which a take is a hum.
pub const HUM_AT: f64 = 0.5;

fn median_clamp(x: &[f64]) -> f64 {
    let mut y = x.to_vec();
    y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if y.is_empty() { 0.0 } else if y.len() % 2 == 1 { y[y.len() / 2] } else { (y[y.len() / 2 - 1] + y[y.len() / 2]) / 2.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact (text, weight) pairs the mantra must produce, in order.
    fn expected_pairs() -> Vec<(&'static str, Weight)> {
        vec![
            ("\u{950}", Weight::Guru),
            ("\u{924}\u{94D}\u{930}\u{94D}\u{92F}\u{92E}\u{94D}", Weight::Guru),
            ("\u{92C}", Weight::Laghu),
            ("\u{915}\u{902}", Weight::Guru),
            ("\u{92F}", Weight::Laghu),
            ("\u{91C}\u{93E}", Weight::Guru),
            ("\u{92E}", Weight::Laghu),
            ("\u{939}\u{947}", Weight::Guru),
            ("\u{938}\u{941}", Weight::Laghu),
            ("\u{917}\u{928}\u{94D}", Weight::Guru),
            ("\u{927}\u{93F}\u{902}", Weight::Guru),
            ("\u{92A}\u{941}\u{937}\u{94D}", Weight::Guru),
            ("\u{91F}\u{93F}", Weight::Laghu),
            ("\u{935}\u{930}\u{94D}", Weight::Guru),
            ("\u{927}", Weight::Laghu),
            ("\u{928}\u{92E}\u{94D}", Weight::Guru),
            ("\u{909}\u{930}\u{94D}", Weight::Guru),
            ("\u{935}\u{93E}", Weight::Guru),
            ("\u{930}\u{941}", Weight::Laghu),
            ("\u{915}", Weight::Laghu),
            ("\u{92E}\u{93F}", Weight::Laghu),
            ("\u{935}", Weight::Laghu),
            ("\u{92C}\u{928}\u{94D}", Weight::Guru),
            ("\u{927}", Weight::Laghu),
            ("\u{928}\u{93E}\u{928}\u{94D}", Weight::Guru),
            ("\u{92E}\u{943}\u{924}\u{94D}", Weight::Guru),
            ("\u{92F}\u{94B}\u{930}\u{94D}", Weight::Guru),
            ("\u{92E}\u{941}\u{915}\u{94D}", Weight::Guru),
            ("\u{937}\u{940}", Weight::Guru),
            ("\u{92F}", Weight::Laghu),
            ("\u{92E}\u{93E}", Weight::Guru),
            ("\u{92E}\u{943}", Weight::Laghu),
            ("\u{924}\u{93E}\u{924}\u{94D}", Weight::Guru),
        ]
    }

    const MANTRA: &str = concat!(
        "\u{950} \u{924}\u{94D}\u{930}\u{94D}\u{92F}\u{92E}\u{94D}\u{92C}",
        "\u{915}\u{902} \u{92F}\u{91C}\u{93E}\u{92E}\u{939}\u{947} \u{938}",
        "\u{941}\u{917}\u{928}\u{94D}\u{927}\u{93F}\u{902} \u{92A}\u{941}",
        "\u{937}\u{94D}\u{91F}\u{93F}\u{935}\u{930}\u{94D}\u{927}\u{928}",
        "\u{92E}\u{94D} \u{964} \u{909}\u{930}\u{94D}\u{935}\u{93E}\u{930}",
        "\u{941}\u{915}\u{92E}\u{93F}\u{935} \u{92C}\u{928}\u{94D}\u{927}",
        "\u{928}\u{93E}\u{928}\u{94D} \u{92E}\u{943}\u{924}\u{94D}\u{92F}",
        "\u{94B}\u{930}\u{94D}\u{92E}\u{941}\u{915}\u{94D}\u{937}\u{940}",
        "\u{92F} \u{92E}\u{93E}\u{92E}\u{943}\u{924}\u{93E}\u{924}\u{94D}",
        " \u{965}"
    );

    #[test]
    fn mantra_scansion() {
        let got = syllables(MANTRA);
        let exp = expected_pairs();
        assert_eq!(got.len(), exp.len(), "syllable count");
        for (s, pair) in got.iter().zip(exp.iter()) {
            let (t, w) = *pair;
            assert_eq!(s.text.as_str(), t, "text mismatch at weight {:?}", w);
            assert_eq!(s.weight, w);
        }
    }

    #[test]
    fn score_matches_scansion() {
        let path = format!("{}/tests/data/score-mahamrityunjaya-A.json", env!("CARGO_MANIFEST_DIR"));
        let score = load_score(&path).expect("load score");
        let scored: Vec<(&str, Weight)> = score
            .syllables()
            .iter()
            .map(|e| (e.text().unwrap(), e.weight().unwrap()))
            .collect();
        let mantra_vec = syllables(MANTRA);
        let mantra_pairs: Vec<(&str, Weight)> =
            mantra_vec.iter().map(|s| (s.text.as_str(), s.weight)).collect();
        assert_eq!(scored, mantra_pairs, "score events must match the mantra scansion");
    }

    // First chaupai line.
    const C1: &str = concat!(
        "\u{91C}\u{92F} \u{939}\u{928}\u{941}\u{92E}\u{93E}",
        "\u{928} \u{91C}\u{94D}\u{91E}\u{93E}\u{928} \u{917}\u{941}\u{928}",
        " \u{938}\u{93E}\u{917}\u{930}"
    );
    // Second chaupai line.
    const C2: &str = concat!(
        "\u{91C}\u{92F} \u{915}\u{92A}\u{940}\u{938} \u{924}",
        "\u{93F}\u{939}\u{941}\u{901} \u{932}\u{94B}\u{915} \u{909}\u{91C}",
        "\u{93E}\u{917}\u{930}"
    );

    #[test]
    fn chaupai_lines_are_16_matras() {
        assert_eq!(matras(&syllables(C1)), 16, "first line: {:?}", syllables(C1));
        assert_eq!(matras(&syllables(C2)), 16, "second line: {:?}", syllables(C2));
    }

    #[test]
    fn chandrabindu_is_laghu() {
        // Second chaupai has ह+ु+chandrabindu (\u{901}); it only nasalises, so laghu.
        let s = syllables(C2);
        assert!(s.iter().any(|x| x.text == "\u{939}\u{941}\u{901}" && x.weight == Weight::Laghu));
    }

    #[test]
    fn syllables_join_back_to_their_word() {
        for w in [
            "\u{950}",
            concat!(
                "\u{924}\u{94D}\u{930}\u{94D}\u{92F}\u{92E}\u{94D}\u{92C}\u{915}",
                "\u{902}"
            ),
            "\u{92F}\u{91C}\u{93E}\u{92E}\u{939}\u{947}",
            "\u{938}\u{941}\u{917}\u{928}\u{94D}\u{927}\u{93F}\u{902}",
            concat!(
                "\u{92A}\u{941}\u{937}\u{94D}\u{91F}\u{93F}\u{935}\u{930}",
                "\u{94D}\u{927}\u{928}\u{92E}\u{94D}"
            ),
            concat!(
                "\u{909}\u{930}\u{94D}\u{935}\u{93E}\u{930}\u{941}\u{915}",
                "\u{92E}\u{93F}\u{935}"
            ),
            "\u{92C}\u{928}\u{94D}\u{927}\u{928}\u{93E}\u{928}\u{94D}",
            concat!(
                "\u{92E}\u{943}\u{924}\u{94D}\u{92F}\u{94B}\u{930}\u{94D}",
                "\u{92E}\u{941}\u{915}\u{94D}\u{937}\u{940}\u{92F}"
            ),
            "\u{92E}\u{93E}\u{92E}\u{943}\u{924}\u{93E}\u{924}\u{94D}",
        ] {
            let joined: String = syllables(w).iter().map(|x| x.text.as_str()).collect();
            assert_eq!(joined, w, "round trip failed for {:?}", w);
        }
    }

    #[test]
    fn word_indices_advance() {
        let s = syllables("\u{92C}\u{928}\u{94D} \u{915}\u{902}");
        assert!(s.iter().map(|x| x.word).any(|w| w == 0));
        assert!(s.iter().map(|x| x.word).any(|w| w == 1));
    }

    #[test]
    fn weight_rules() {
        assert_eq!(weight("\u{924}"), Weight::Laghu); // short, open
        assert_eq!(weight("\u{92E}\u{93E}"), Weight::Guru); // long vowel
        assert_eq!(weight("\u{915}\u{902}"), Weight::Guru); // anusvara
        assert_eq!(weight("\u{939}\u{941}\u{901}"), Weight::Laghu); // chandrabindu only nasalises
        assert_eq!(weight("\u{92C}\u{928}\u{94D}"), Weight::Guru); // closed
    }

    // ---- M2: split and hum ---------------------------------------------------------------------------

    /// Sample rate for the synthetic takes (harmonics stay under Nyquist, pitch inside analyze's 70..=500 Hz picker band).
    const SR: f32 = 8_000.0;

    fn pulse_train(freq: f32, dur_s: f32) -> Vec<f32> {
        let n = (dur_s * SR) as usize;
        (0..n).map(|i| {
            let t = i as f32 / SR;
            let mut s = 0.0f64;
            for h in 1..=6u8 { s += (1.0 / h as f64) * (2.0f32 * std::f32::consts::PI * freq * h as f32 * t).sin() as f64; }
            (s * 0.25) as f32 // a clean voiced tone: the vowel nucleus
        }).collect()
    }

    fn noise_burst(dur_s: f32) -> Vec<f32> {
        let n = (dur_s * SR) as usize;
        (0..n).map(|i| {
            let v = i.wrapping_mul(1_103_515_125usize).wrapping_add(12_345) >> 16;
            let x = (v as i32 % 1000 - 500) as f32 / 1000.0 * 0.3f32; // quiet, unvoiced: the consonant between two vowels
            x
        }).collect()
    }

    /// Three voiced vowels (180/240/180 Hz pulse trains, 150 ms each) with 60 ms noise bursts between them and a tail of
    /// silence — one word with three syllables.
    fn synthetic_word() -> Vec<f32> {
        let mut v = Vec::new();
        v.extend(pulse_train(180.0, 0.150));
        v.extend(noise_burst(0.060));
        v.extend(pulse_train(240.0, 0.150));
        v.extend(noise_burst(0.060));
        v.extend(pulse_train(180.0, 0.150));
        v.extend(std::iter::repeat(0f32).take((0.10 * SR) as usize)); // tail silence
        v
    }

    /// A single steady note, constant amplitude throughout: a hum.
    fn steady_tone() -> Vec<f32> {
        let n = (2.5 * SR) as usize;
        (0..n).map(|i| {
            let t = i as f32 / SR;
            let mut s = 0.0f64;
            for h in 1..=6u8 { s += (1.0 / h as f64) * (2.0f32 * std::f32::consts::PI * 180.0f32 * h as f32 * t).sin() as f64; }
            (s * 0.25) as f32
        }).collect()
    }

    #[test]
    fn a_three_vowel_word_is_split_into_three_with_boundaries_at_the_bursts() {
        let pcm = synthetic_word();
        let texts: Vec<&str> = vec!["a", "b", "c"];
        let cuts = split(&pcm, SR as u32, &texts).expect("three voiced vowels split").cuts;
        assert_eq!(cuts.len(), 3, "one nucleus per text: {cuts:?}");
        // the bursts (150-210 ms and 360-420 ms) start the second and third syllables: each boundary is where the burst
        // begins, and each vowel lies between the bursts, within 20 ms
        let b0 = cuts[0].end_s;
        let b1 = cuts[1].end_s;
        assert_eq!(b0, cuts[1].start_s, "boundaries join the syllables");
        assert_eq!(b1, cuts[2].start_s, "boundaries join the syllables");
        assert!((b0 - 0.150).abs() < 0.02, "boundary one where the first burst begins: {cuts:?}");
        assert!((b1 - 0.360).abs() < 0.02, "boundary two where the second burst begins: {cuts:?}");
        assert!((cuts[1].vowel_start_s - 0.210).abs() < 0.02 && (cuts[2].vowel_start_s - 0.420).abs() < 0.02,
                "each vowel starts after its consonant: {cuts:?}");
        // each syllable's vowel is inside its own span and lasts more than a frame.
        for c in &cuts {
            assert!(c.start_s <= c.vowel_start_s && c.vowel_start_s <= c.vowel_end_s && c.vowel_end_s <= c.end_s, "ordering: {c:?}");
            assert!((c.vowel_end_s - c.vowel_start_s) > 0.05, "a real vowel is voiced for a while: {c:?}");
        }
    }

    #[test]
    fn a_steady_tone_is_a_hum() {
        let pcm = steady_tone();
        assert!(is_hum(&pcm, SR as u32), "a single held note is a hum (score {}):", hum_score(&pcm, SR as u32));
    }

    #[test]
    fn the_synthetic_word_is_not_a_hum() {
        let pcm = synthetic_word();
        assert!(!is_hum(&pcm, SR as u32), "a spoken word with consonants and shifting pitch is not a hum (score {}):", hum_score(&pcm, SR as u32));
    }

    // ---- sing::check synthesis and expectations ------------------------------------------------------

    /// Every slot starts at the take origin (event_windows runs slots from here, which is where frame 0 also lands).
    const TAKE_START_S: f64 = 0.0;
    /// Sample rate: harmonics stay under Nyquist and pitch stays inside analyze's 70..=500 Hz band for the picker.
    const SAMPLE_RATE: f64 = 16000.0;
    /// The take's tonic (A below middle C), the reference every swara name is measured against.
    const TONIC_HZ: f64 = 130.0;

    fn syllable(text: &str, note: &str) -> Event {
        Event::Syllable(SyllableEvent { text: text.to_string(), latin: None, weight: Weight::Guru, beats: Some(4.0), note: note.to_string(), glide_to: None })
    }

    /// Four words (`Sa Ga Ma Pa`), one syllable each, long beats so each event's +-200 ms onset window sees only its own voice.
    fn right_take_score() -> Score {
        Score {
            bpm: 168.0,
            word_gap: Some(WordGap { beats: 2.0 }),
            events: vec![syllable("sa", "Ga"), syllable("ga", "Ma"), syllable("ma", "Pa"), syllable("pa", "Sa")],
            words: vec![
                Word { text: "sa".into(), latin: Some("sa").map(|s| s.to_string()), events: vec![0] },
                Word { text: "ga".into(), latin: Some("ga").map(|s| s.to_string()), events: vec![1] },
                Word { text: "ma".into(), latin: Some("ma").map(|s| s.to_string()), events: vec![2] },
                Word { text: "pa".into(), latin: Some("pa").map(|s| s.to_string()), events: vec![3] },
            ],
        }
    }

    /// Three single-syllable words, used for the continuous-voice / hard-cut case.
    fn continuous_take_score() -> Score {
        Score {
            bpm: 168.0,
            word_gap: Some(WordGap { beats: 2.0 }),
            events: vec![syllable("sa", "Ga"), syllable("ga", "Ma"), syllable("ma", "Pa")],
            words: vec![
                Word { text: "sa".into(), latin: Some("sa").map(|s| s.to_string()), events: vec![0] },
                Word { text: "ga".into(), latin: Some("ga").map(|s| s.to_string()), events: vec![1] },
                Word { text: "ma".into(), latin: Some("ma").map(|s| s.to_string()), events: vec![2] },
            ],
        }
    }

    /// Synthesize each syllable as fundamental + five harmonics, a 30 ms attack, and (when `fade_ms > 0`) a linear fade over the
    /// final `fade_ms` of its slot. Pure tones on pitch; `freq_scale` shifts every note by that ratio. Gaps are left silent.
    fn synthesize_tones(score: &Score, freq_scale: f64, fade_ms: f64) -> Vec<f32> {
        let wins = event_windows(score, TAKE_START_S);
        let end = wins.iter().map(|(_, t)| *t).fold(0f64, f64::max);
        let mut pcm = vec![0.0f32; (end * SAMPLE_RATE) as usize + 1];
        for (i, e) in score.events.iter().enumerate() {
            if let Event::Syllable(s) = e {
                let (t0, t1) = wins[i];
                let base = note_frequency(&s.note, TONIC_HZ) * freq_scale;
                for j in ((t0 * SAMPLE_RATE) as usize)..((t1 * SAMPLE_RATE) as usize) {
                    let tc = (j as f64) / SAMPLE_RATE;
                    let local = (tc - t0).max(0.0);
                    let mut env = (local * 1000.0 / 30.0).min(1.0); // 30 ms attack ramp to full amplitude
                    if fade_ms > 0.0 { env *= ((t1 - tc) * 1000.0 / fade_ms).clamp(0.0, 1.0); } // fade over the final `fade_ms`
                    let phase = tc * base * std::f64::consts::TAU;
                    let mut sum = 0.0f64;
                    for h in 1..=6u8 { sum += (1.0 / h as f64) * (phase * h as f64).sin(); }
                    pcm[j] += (0.5 * env * sum) as f32;
                }
            }
        }
        pcm
    }

    /// A continuous tone across all words at constant amplitude, so every gap stays voiced and no word gap is missing; the tone
    /// drops to silence exactly at the end of `cut_word`'s syllable, leaving a single-frame hard cut there.
    fn synthesize_continuous(score: &Score, cut_word: usize) -> Vec<f32> {
        let wins = event_windows(score, TAKE_START_S);
        let start = wins[0].0;
        let w = cut_word.min(score.events.len() - 1);
        let (t0_w, t1_w) = wins[w]; // the tone runs until this word's syllable ends, then stops — a hard cut there
        let base = note_frequency(&score.events.last().unwrap().text().unwrap(), TONIC_HZ);
        let tail = (t1_w - t0_w) + 1.0; // silence after the cut so the drop spans a real frame
        let len = ((t1_w + tail) * SAMPLE_RATE) as usize + 1;
        let mut pcm = vec![0.0f32; len];
        for j in ((start * SAMPLE_RATE) as usize)..((t1_w * SAMPLE_RATE) as usize) {
            let tc = (j as f64) / SAMPLE_RATE;
            let phase = tc * base * std::f64::consts::TAU;
            let mut sum = 0.0f64;
            for h in 1..=6u8 { sum += (1.0 / h as f64) * (phase * h as f64).sin(); }
            pcm[j] += (0.5 * sum) as f32; // no attack, no fade: the voice fills every gap and only stops at the hard cut
        }
        pcm
    }

    /// A take read exactly on pitch, on the beat, with nothing missing or cut — each word's last syllable fades into its quiet
    /// gap, so there is neither a missing gap nor an abrupt cut.
    #[test]
    fn sing_check_right_take_is_clear() {
        let score = right_take_score();
        let pcm = synthesize_tones(&score, 1.0, 80.0); // pure tones; the last syllable of each word fades into its gap
        let got = check(&pcm, SAMPLE_RATE as u32, &score, TAKE_START_S, TONIC_HZ);
        let s = &got["summary"];
        assert!(s["median_abs_cents"].as_f64().unwrap() < 10.0, "on pitch: {:?}", got);
        assert!(s["median_abs_onset_ms"].as_f64().unwrap() < 25.0, "on the beat: {:?}", got);
        assert_eq!(s["faults"].as_u64().unwrap(), 0, "no missing gaps or cuts: {:?}", got);
        assert_eq!(s["syllables"].as_u64().unwrap(), 4);
    }

    /// The same take sung a semitone sharp: every note is shifted up one half step so the pitch error centers on ~100 cents.
    #[test]
    fn sing_check_a_semitone_sharp_reads_around_100_cents() {
        let score = right_take_score();
        let pcm = synthesize_tones(&score, 2f64.powf(1.0 / 12.0), 80.0); // each note raised a semitone
        let got = check(&pcm, SAMPLE_RATE as u32, &score, TAKE_START_S, TONIC_HZ);
        let cents = got["summary"]["median_abs_cents"].as_f64().unwrap();
        assert!((90.0..=110.0).contains(&cents), "a semitone sharp is about 100 cents: {:?}", got);
    }

    /// A take whose gaps are filled with voice (no fade, so every boundary stays voiced) and where the last word is hard-cut
    /// instead of faded: every voiced boundary is missing its gap, and the one cut word is flagged as a hard cut.
    #[test]
    fn sing_check_continuous_voice_with_one_hard_cut_reports_both() {
        let score = continuous_take_score();
        let pcm = synthesize_continuous(&score, 2); // voice fills every gap; hard-cut the last word
        let got = check(&pcm, SAMPLE_RATE as u32, &score, TAKE_START_S, TONIC_HZ);
        let faults: Vec<&serde_json::Value> = got["faults"].as_array().unwrap().iter().collect();
        assert!(faults.iter().any(|f| f["kind"].as_str() == Some("no_word_gap")), "every voiced boundary is missing its gap: {:?}", got);
        assert!(faults.iter().any(|f| f["kind"].as_str() == Some("hard_cut")), "the one cut word is flagged: {:?}", got);
    }
}
