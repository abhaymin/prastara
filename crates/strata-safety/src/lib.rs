//! The minor-safety rules: a hard check that no setting turns off. No sexual content involving minors, whether from an
//! uncensored model on the server (`am2strata`'s `minor_safety`, which reads every chat with such a model) or in a
//! private dataset the Workshop builds (`strata-workshop`'s `private_data`, which drops a row on a match).
//!
//! **The rules** ([`check`]): the text is lower-cased and split into words (letters and digits; common digit-for-letter
//! swaps undone). It matches when
//! 1. it holds a term that names sexual content of minors outright (`lolicon`, `jailbait`, `child porn`, ...), or
//! 2. a minor term (child, kid, toddler, preteen, underage, schoolgirl, an age under 18 written as "12 year old",
//!    "12yo", "age 12", "twelve-year-old", ...) and a sexual term (sexual, porn, nude, naked, erotic, orgasm,
//!    intercourse, masturbating, genitals, "have sex", ...) are within [`WINDOW`] words of each other.
//!
//! It is strict by design: it can refuse an innocent text that puts the two kinds of word close together. Deterministic
//! rules, no classifier model and nothing downloaded.

/// Words apart at most for a minor term and a sexual term to count as one statement.
pub const WINDOW: usize = 14;

/// Terms that name sexual content of minors by themselves (as words, or two words in a row).
const OUTRIGHT: &[&str] = &[
    "loli", "lolis", "lolicon", "shota", "shotacon", "jailbait", "pedo", "pedophile", "paedophile", "pedophilia", "paedophilia",
    "child porn", "child pornography", "kiddie porn", "kiddy porn", "kid porn", "childporn", "cp porn", "preteen porn", "underage porn",
    "csam", "child sexual", "pthc",
];

/// Words that say the person is a minor.
const MINOR: &[&str] = &[
    "child", "children", "childs", "kid", "kids", "kiddie", "kiddy", "toddler", "toddlers", "infant", "infants", "preteen", "preteens",
    "pre teen", "tween", "tweens", "prepubescent", "pubescent", "underage", "under age", "underaged", "minor", "minors", "juvenile",
    "schoolgirl", "schoolgirls", "schoolboy", "schoolboys", "little girl", "little girls", "little boy", "little boys", "young girl",
    "young girls", "young boy", "young boys", "teen", "teens", "teenage", "teenager", "teenagers", "middle school", "elementary school",
    "grade school", "primary school", "kindergarten", "first grader", "second grader", "third grader", "fourth grader", "fifth grader",
    "sixth grader", "seventh grader", "eighth grader", "ninth grader",
];

/// Words of sexual content.
const SEXUAL: &[&str] = &[
    "sexual", "sexually", "sexy", "sex with", "have sex", "has sex", "had sex", "having sex", "sex scene", "sex act", "sex acts",
    "porn", "porno", "pornographic", "pornography", "nude", "nudes", "nudity", "naked", "erotic", "erotica", "orgasm", "orgasms",
    "intercourse", "masturbate", "masturbates", "masturbating", "masturbation", "genital", "genitals", "penis", "vagina", "vaginal",
    "clitoris", "blowjob", "handjob", "fellatio", "cunnilingus", "oral sex", "anal sex", "aroused", "arousal", "horny", "lewd", "molest",
    "molested", "molesting", "molestation", "fondle", "fondled", "fondling", "grope", "groped", "groping", "seduce", "seduced",
    "seducing", "seduction", "make love", "making love", "lingerie", "fetish", "rape", "raped", "raping", "explicit", "nsfw",
    "undress", "undressed", "undressing", "strip naked", "spread her legs", "spread his legs", "breasts", "nipples", "cum", "cumming",
];

const NUMBERS: &[&str] = &["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                           "fourteen", "fifteen", "sixteen", "seventeen"];

/// Lower case words of `text`, letters and digits only; `0`→o, `1`→i, `3`→e, `4`→a, `5`→s, `@`→a, `$`→s inside a word
/// that has letters (so "l0li" is "loli" while "12" stays 12).
fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.to_lowercase().split(|c: char| !(c.is_alphanumeric() || c == '@' || c == '$')) {
        if raw.is_empty() {
            continue;
        }
        let has_letters = raw.chars().any(|c| c.is_alphabetic());
        let w: String = if has_letters && raw.chars().any(|c| c.is_ascii_digit() || c == '@' || c == '$') && !is_age_word(raw) {
            raw.chars().map(|c| match c { '0' => 'o', '1' => 'i', '3' => 'e', '4' => 'a', '5' => 's', '@' => 'a', '$' => 's', c => c }).collect()
        } else {
            raw.chars().filter(|c| c.is_alphanumeric()).collect()
        };
        if !w.is_empty() {
            out.push(w);
        }
    }
    out
}

/// "12yo", "12yr", "12yrs", "12y": a number with an age suffix.
fn is_age_word(w: &str) -> bool {
    let digits: String = w.chars().take_while(|c| c.is_ascii_digit()).collect();
    !digits.is_empty() && ["yo", "y", "yr", "yrs", "yearold", "yearolds"].contains(&&w[digits.len()..])
}

fn minor_age(n: u32) -> bool {
    (1..18).contains(&n)
}

/// Positions of every match of `terms` (one or two words) in `w`.
fn find(w: &[String], terms: &[&str]) -> Vec<usize> {
    let mut at = Vec::new();
    for (i, x) in w.iter().enumerate() {
        for t in terms {
            let mut parts = t.split(' ');
            let first = parts.next().unwrap_or("");
            let second = parts.next();
            if x == first && second.is_none_or(|s| w.get(i + 1).is_some_and(|y| y == s)) {
                at.push(i);
                break;
            }
        }
    }
    at
}

/// Positions where an age under 18 is written: "12 year old", "12 years old", "12yo", "aged 12", "age 12", "12-year-old",
/// "twelve year old".
fn ages(w: &[String]) -> Vec<usize> {
    let mut at = Vec::new();
    for (i, x) in w.iter().enumerate() {
        let num = x.parse::<u32>().ok().or_else(|| NUMBERS.iter().position(|n| n == x).map(|p| p as u32 + 1));
        if let Some(n) = num {
            let next = w.get(i + 1).map(String::as_str);
            let then = w.get(i + 2).map(String::as_str);
            let yearly = matches!(next, Some("year" | "years" | "yr" | "yrs" | "yo")) && (next == Some("yo") || matches!(then, Some("old" | "olds")));
            let before = i.checked_sub(1).and_then(|j| w.get(j)).map(String::as_str);
            if minor_age(n) && (yearly || matches!(before, Some("age" | "aged"))) {
                at.push(i);
            }
        } else if is_age_word(x) {
            let digits: String = x.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.parse::<u32>().is_ok_and(minor_age) {
                at.push(i);
            }
        }
    }
    at
}

/// What matched, or None. See the module documentation for the rules.
pub fn check(text: &str) -> Option<&'static str> {
    let w = words(text);
    if !find(&w, OUTRIGHT).is_empty() {
        return Some("a term for sexual content of minors");
    }
    let sexual = find(&w, SEXUAL);
    if sexual.is_empty() {
        return None;
    }
    let mut minor = find(&w, MINOR);
    minor.extend(ages(&w));
    let near = minor.iter().any(|m| sexual.iter().any(|s| m.abs_diff(*s) <= WINDOW));
    near.then_some("a minor and sexual content together")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_innocent_pair_of_trigger_words_still_matches() {
        // the rule is about words, not meaning: a minor term and a sexual term close together match
        assert!(check("the juvenile court kept the explicit details private").is_some());
        assert!(check("the court kept the details private").is_none());
    }

    #[test]
    fn far_apart_words_do_not_match() {
        let filler = "the weather was mild and the market sold apples pears and many other fruits all through that long afternoon";
        assert!(check(&format!("a juvenile record. {filler}. the explicit part")).is_none());
    }

    #[test]
    fn ages_are_read() {
        assert!(check("an explicit note about someone aged 15").is_some());
        assert!(check("an explicit note about someone aged 25").is_none());
    }
}

/// The fixed rules of a private-data import (`strata-workshop`'s `private_import`): how a check model's label scores
/// become a decision, and when a whole source is rejected. No setting turns any of them off; a setting can only make
/// them stricter.
pub mod import {
    /// The share of a source's items that may be flagged as involving a minor before the whole import is rejected and
    /// its quarantine deleted. A setting can lower it, never raise it.
    pub const MAX_MINOR_SHARE: f64 = 0.01;
    /// A minor check (image, text, age) flags an item when its flag labels together score at least this. A setting can
    /// lower it (stricter), never raise it above [`MINOR_THRESHOLD_MAX`].
    pub const MINOR_THRESHOLD_DEFAULT: f32 = 0.2;
    pub const MINOR_THRESHOLD_MAX: f32 = 0.5;
    /// The real-person check's default and highest threshold.
    pub const REAL_THRESHOLD_DEFAULT: f32 = 0.5;
    pub const REAL_THRESHOLD_MAX: f32 = 0.9;
    /// An age model that answers in years (one output) flags an item under this age.
    pub const ADULT_AGE: f32 = 18.0;
    /// ... plus at least this margin, for the model's error (a setting can raise it, never lower it).
    pub const AGE_MARGIN_MIN: f32 = 3.0;

    /// The minor-share limit a source runs with: `asked` when it is stricter, else the fixed one.
    pub fn minor_share(asked: Option<f64>) -> f64 {
        match asked {
            Some(a) if a.is_finite() && (0.0..=MAX_MINOR_SHARE).contains(&a) => a,
            _ => MAX_MINOR_SHARE,
        }
    }

    /// Whether `flagged` of `total` items passes the limit (the whole source is rejected). No items: not rejected.
    pub fn source_rejected(flagged: usize, total: usize, share: f64) -> bool {
        total > 0 && flagged as f64 > share * total as f64
    }

    /// A threshold as asked, held inside `(0, max]`; anything else (missing, NaN, above max) is `default`.
    pub fn threshold(asked: Option<f32>, default: f32, max: f32) -> f32 {
        match asked {
            Some(t) if t.is_finite() && t > 0.0 && t <= max => t,
            Some(t) if t.is_finite() && t > max => max,
            _ => default,
        }
    }

    /// An age label as a range of years: "0-2", "10-19", "20 - 29", "more than 70", "70+", "under 18", "12", "(8, 12)".
    /// None when it names no age.
    pub fn age_range(label: &str) -> Option<(f32, f32)> {
        let l = label.to_lowercase();
        let nums: Vec<f32> = l.split(|c: char| !c.is_ascii_digit() && c != '.').filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<f32>().ok()).collect();
        match nums.as_slice() {
            [] => None,
            [n] if l.contains("more") || l.contains("over") || l.contains('+') || l.contains("above") || l.contains("older") => Some((*n, 200.0)),
            [n] if l.contains("under") || l.contains("less") || l.contains("below") || l.contains("younger") || l.contains('<') => Some((0.0, *n)),
            [n] => Some((*n, *n)),
            [a, b, ..] => Some((a.min(*b), a.max(*b))),
        }
    }

    /// Whether an age label may mean someone under 18: any range that reaches below 18 (so "10-19" does).
    pub fn age_label_minor(label: &str) -> Option<bool> {
        age_range(label).map(|(lo, _)| lo < ADULT_AGE)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn settings_only_make_the_rules_stricter() {
            assert_eq!(minor_share(None), MAX_MINOR_SHARE);
            assert_eq!(minor_share(Some(0.5)), MAX_MINOR_SHARE, "a higher share is not taken");
            assert_eq!(minor_share(Some(0.0)), 0.0);
            assert_eq!(minor_share(Some(f64::NAN)), MAX_MINOR_SHARE);
            assert_eq!(threshold(Some(0.9), MINOR_THRESHOLD_DEFAULT, MINOR_THRESHOLD_MAX), MINOR_THRESHOLD_MAX);
            assert_eq!(threshold(Some(0.05), MINOR_THRESHOLD_DEFAULT, MINOR_THRESHOLD_MAX), 0.05);
            assert_eq!(threshold(Some(0.0), MINOR_THRESHOLD_DEFAULT, MINOR_THRESHOLD_MAX), MINOR_THRESHOLD_DEFAULT);
            assert_eq!(threshold(None, MINOR_THRESHOLD_DEFAULT, MINOR_THRESHOLD_MAX), MINOR_THRESHOLD_DEFAULT);
        }

        #[test]
        fn a_source_over_the_share_is_rejected() {
            assert!(!source_rejected(1, 100, 0.01), "1 % is the limit, not over it");
            assert!(source_rejected(2, 100, 0.01));
            assert!(source_rejected(1, 10, 0.01));
            assert!(source_rejected(1, 1000, 0.0));
            assert!(!source_rejected(0, 0, 0.01));
        }

        #[test]
        fn age_labels_are_read_and_a_range_reaching_under_18_is_minor() {
            assert_eq!(age_label_minor("0-2"), Some(true));
            assert_eq!(age_label_minor("10-19"), Some(true));
            assert_eq!(age_label_minor("20-29"), Some(false));
            assert_eq!(age_label_minor("more than 70"), Some(false));
            assert_eq!(age_label_minor("70+"), Some(false));
            assert_eq!(age_label_minor("under 18"), Some(true));
            assert_eq!(age_label_minor("(15, 20)"), Some(true));
            assert_eq!(age_label_minor("adult"), None);
        }
    }
}
