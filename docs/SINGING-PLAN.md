# Singing: turning clear speech into song with a score

**Request (2026-11-21):** songs and chants, mantras above all, need a singing voice. Singing is not speech: the voice is
modulated differently.

## Why not a song model alone

**ACE-Step** makes good instrumental beds. Its sung words, though, follow its own timing, and in Sanskrit they blur:
- nine chant takes were checked word by word;
- every one swallowed or reordered words;
- the start-to-start gaps between words varied by 57–81%.

**No open singing model sings Hindi or Sanskrit with note-level control** (survey, 2026-11-21):
- The score-driven singers (SoulX-Singer, DiffSinger voicebanks, VISinger2, NNSVS) are trained on Chinese, English,
  Japanese and Korean.
- The only Hindi DiffSinger work is one hour of one male singer, under an unclear licence.

## The approach

1. **Speech:** a text-to-speech voice (Indic Parler) speaks each word clearly. This is where the words come from.
2. **A score** gives every syllable:
   - its text;
   - its weight: laghu = 1 beat, guru = 2;
   - its note (a swara relative to Sa) and an optional glide;
   - the rests and breaths.

   Score files are JSON (`score-*.json`).
3. **`media sing`** turns the speech into singing, syllable by syllable:
   - the consonant is kept as spoken and placed just before the beat;
   - the vowel is stretched to the note's length and set to its pitch by pitch-synchronous overlap-add (PSOLA, in Rust);
   - then come glides, accents on the guru syllables, and short crisp endings.
4. **The dub** lays the voice over the instrumental bed, made at the same tempo and key.
5. **`media sing-check`** measures a take against its score, per syllable:
   - pitch error in cents;
   - vowel start against the beat, in ms;
   - length error;
   - silence where no breath is written.

## How singing differs from speech, and who controls it

| Part | Speech | Singing (the score) |
|---|---|---|
| Pitch | drifts all the time | held on the note; short glides between notes |
| Timing | the word's own rhythm | the vowel on the beat, about 80–90% of the syllable |
| Length | even | laghu 1, guru 2, phrase ends held longer |
| Vibrato | none | none for Vedic chant; light, on long notes, for bhajans |
| Loudness | flat | guru accented; an arc over each phrase |
| Breath | anywhere | only where the score says |

## The voice is linked, and every word is clear

The owner's direction (2026-11-21):
- **Inside a word,** the syllables flow together with no break.
- **Between words** there is a quarter-beat gap (`word_gap` in the score), so no word runs into the next. The last
  vowel rings and fades into the gap, never cut hard, and the next word starts on a clean consonant.
- **Om** is one ringing sound, o into a long m hum, and carries into the next word.
- **Breaths** are not cuts: the note rings and fades, and the next phrase grows out of it.
- **Speech clean-up:** `media sing` never uses the speech clean-ups that trim tails and silences.
- **`sing-check` faults:** a gap of more than 60 ms inside a word, and a word boundary without a clear dip.

## Rules

- **Synthetic voices only** (VOICES-PLAN): no real person's voice is cloned. Any voice-conversion model is trained only
  on a synthetic voice, and only with the owner's decision.
- **All in Rust,** with no WORLD and no Python. Any voice-conversion model runs through ONNX (`ort`).
- **Each part is a UI panel and a CLI:**
  - `am2strata-run media sing` and `sing-check`;
  - `POST /v1/media/sing`;
  - a Sing panel in the Audio tab.

## Steps

| Step | What | Gate |
|---|---|---|
| M0 | the score format and `sing-check` | flags a take a semitone off; the octave is checked on a known female clip |
| M1 | the syllabifier: Devanagari in, syllables with laghu/guru out | the 32 syllables of RV 7.59.12 match the hand scansion; also scans 16-matra chaupai lines |
| M2 | speech word mode (`"words": […]`, retaken on a hum) and the vowel-centre split | every syllable found; cuts within 30 ms of hand marks |
| M3 | `media sing` (PSOLA retime and re-pitch, glides, accents) and the dub | median pitch error under 50 cents and vowel timing under 40 ms; the words still heard; the owner's ear |
| M4 | optional voice conversion (RVC, ONNX), only if M3 sounds robotic | the owner's yes to the download and the voice |
| M5 | presets (Vedic, bhajan), the UI panel, `docs/MEDIA.md` | the owner's ear on a full piece |

## Status and findings

- **M1 and M0** are done: the syllabifier and `sing-check`.
- **Divya's voice** sits at about 120–185 Hz. The spectrum confirms that this is her true pitch, so Sa = C#3 suits her.
- **M2:** word mode and `media syllables` work.
  - **The hum detector** was checked on real takes: six hums scored 0.78–0.90, and nine spoken words scored 0.23 or less.
  - **The split is approximate.** It finds the right number of vowels, but some land tens of ms off, and a few are wrong
    where vowels join through r, v, y or a nasal. Its confidence does not yet tell the good splits from the bad ones.
  - **Parler ends words in a creaky voice.** A pitch tracker drops creak, so the split uses band loudness, not pitch.
- **For M3:** anchor each vowel nucleus on its beat, and stretch each frame in proportion to how vowel-like it is. That way a
  boundary a few tens of ms off moves where the stretch happens, but it never stretches a consonant.
- **A robust split** needs a phoneme aligner, for example a wav2vec2 phoneme CTC model (Apache-2.0) run through ONNX. That
  is a download, so it needs the owner's yes.
