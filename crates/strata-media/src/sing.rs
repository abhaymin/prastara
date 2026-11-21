//! M1: syllabify Devanagari text into aksharas marked laghu or guru.
//!
//! Pure Rust, no model, no server, no downloads. A word is split on Unicode code points into
//! clusters; each cluster gets a weight from its vowel (long vs short), any anusvara/visarga, or
//! because it is closed. `matras` turns those weights into beat counts.

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
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// A syllable row of a score. Unknown fields (beats, note, glide_to, split, ...) are ignored.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct SyllableEvent {
    pub text: String,
    #[serde(default)]
    pub latin: Option<String>,
    pub weight: Weight,
}

/// A singing score: its ordered events. Unknown top-level fields are ignored.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Score {
    #[serde(default)]
    pub events: Vec<Event>,
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
}
