/// Lager overlappende kontekstvinduer med påfølgende token som fasit.
pub fn sliding_windows(
    tokens: &[u32],
    sequence_length: usize,
) -> impl Iterator<Item = (&[u32], u32)> {
    tokens
        .windows(sequence_length + 1)
        .map(|window| window.split_last().expect("window is never empty"))
        .map(|(&target, sequence)| (sequence, target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_overlapping_sequences_with_next_token_as_target() {
        let tokens = [1, 2, 3, 4, 5];

        let windows: Vec<_> = sliding_windows(&tokens, 3).collect();

        assert_eq!(windows, vec![(&tokens[0..3], 4), (&tokens[1..4], 5)]);
    }

    #[test]
    fn returns_no_windows_when_sequence_is_too_short() {
        let tokens = [1, 2, 3];

        assert_eq!(sliding_windows(&tokens, 3).count(), 0);
    }
}
