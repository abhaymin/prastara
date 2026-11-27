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
