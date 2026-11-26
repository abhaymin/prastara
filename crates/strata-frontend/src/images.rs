//! Pictures in a prompt: their sources in prompt order, and each one's placeholder expanded to its token count.
//! Port of serve/frontend.py `images_of` and the expansion in serve/server.py `Service.prepare`.

use serde_json::Value;

/// The image sources of the rendered conversation, in prompt order (one image item per picture).
pub fn images_of(messages: &[Value]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("image"))
        .map(|item| item.get("source").and_then(Value::as_str).unwrap_or("").to_string())
        .collect()
}

/// Where a picture is in a video: the server's media.rs marks each frame (and its timestamp text) with
/// `"video_frame": {"v": <video>, "i": <frame>, "t": <seconds>}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTag {
    pub video: usize,
    pub frame: usize,
    pub t: f64,
}

pub fn frame_tag(item: &Value) -> Option<FrameTag> {
    let v = item.get("video_frame")?;
    Some(FrameTag { video: v.get("v")?.as_u64()? as usize, frame: v.get("i")?.as_u64()? as usize, t: v.get("t").and_then(Value::as_f64).unwrap_or(0.0) })
}

/// The pictures of the rendered conversation in prompt order, as `images_of`, each with its place in a video.
pub fn image_items(messages: &[Value]) -> Vec<(String, Option<FrameTag>)> {
    messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("image"))
        .map(|item| (item.get("source").and_then(Value::as_str).unwrap_or("").to_string(), frame_tag(item)))
        .collect()
}

/// One `pad` per picture -> one per image token. Only a pad right after `start` (the marker the template writes
/// for an image) is a picture; a pad elsewhere (the text of a message that quotes the marker) becomes the ids of
/// its text, `literal`. `counts`: each picture's token count, in prompt order.
pub fn expand_images(ids: &[u32], pad: u32, start: Option<u32>, literal: &[u32], counts: &[usize])
    -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut k = 0;
    for (j, &t) in ids.iter().enumerate() {
        let is_image = t == pad
            && match start {
                None => k < counts.len(),
                Some(s) => j > 0 && ids[j - 1] == s,
            };
        if is_image {
            if k == counts.len() {
                return Err("the prompt and its images do not match".into());
            }
            out.extend(std::iter::repeat_n(pad, counts[k]));
            k += 1;
        } else if t == pad {
            out.extend_from_slice(literal);
        } else {
            out.push(t);
        }
    }
    if k != counts.len() {
        return Err("the prompt and its images do not match".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_only_marked_pads() {
        // 5 = vision_start, 9 = image_pad; the second 9 is quoted text (not after a start)
        let ids = [1, 5, 9, 6, 2, 9, 3];
        let out = expand_images(&ids, 9, Some(5), &[70, 71], &[3]).unwrap();
        assert_eq!(out, vec![1, 5, 9, 9, 9, 6, 2, 70, 71, 3]);
        assert!(expand_images(&ids, 9, Some(5), &[70], &[3, 4]).is_err());
    }

    #[test]
    fn finds_sources_in_order() {
        let m = serde_json::json!([
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image", "source": "data:1"}]},
            {"role": "tool", "content": [{"type": "image", "source": "data:2"}]}]);
        assert_eq!(images_of(m.as_array().unwrap()), vec!["data:1", "data:2"]);
    }

    #[test]
    fn video_frames_keep_their_place() {
        let m = serde_json::json!([{"role": "user", "content": [
            {"type": "image", "source": "data:a"},
            {"type": "text", "text": "[00:01]", "video_frame": {"v": 0, "i": 1, "t": 1.0}},
            {"type": "image", "source": "data:b", "video_frame": {"v": 0, "i": 1, "t": 1.0}}]}]);
        let items = image_items(m.as_array().unwrap());
        assert_eq!(items, vec![("data:a".to_string(), None), ("data:b".to_string(), Some(FrameTag { video: 0, frame: 1, t: 1.0 }))]);
    }
}
