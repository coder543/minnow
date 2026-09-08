use candle_core::{Device, Tensor};
use minnow::decode::{Options, SpecialTokens, apply_edits, decode_block};
use rand::{SeedableRng, rngs::StdRng};

const S: SpecialTokens = SpecialTokens {
    mask: 7,
    delete: 8,
    split: 9,
    eos: 10,
};

fn logits(ids: &[u32]) -> Tensor {
    let mut data = vec![-5f32; ids.len() * 11];
    for (i, &t) in ids.iter().enumerate() {
        data[i * 11 + t as usize] = 5.0;
        data[i * 11 + 2] = 4.0;
    }
    Tensor::from_vec(data, (ids.len(), 11), &Device::Cpu).unwrap()
}

#[test]
fn edit_tracking_follows_carried_tokens_and_new_masks() {
    let (ids, tracking) = apply_edits(&[1, 8, 9, 4], &[1, 2, 3, 4], &[false, true, true, false], S);
    assert_eq!(ids, vec![1, 7, 3, 4]);
    assert_eq!(tracking, vec![false, false, true, false]);
    let (ids, tracking) = apply_edits(&[1, 8, 8, 4], &[1, 2, 3, 4], &[false, true, true, false], S);
    assert_eq!(ids, vec![1, 4, 7, 7]);
    assert_eq!(tracking, vec![false, false, false, false]);
}

#[test]
fn final_round_suppresses_edits_and_preserves_partial_prompt() {
    let opts = Options {
        max_post_steps: 0,
        ..Options::default()
    };
    let (ids, steps) = decode_block(
        vec![3, 7, 7, 7],
        &opts,
        S,
        &mut StdRng::seed_from_u64(42),
        |_| Ok(logits(&[8, 8, 9, 1])),
    )
    .unwrap();
    assert_eq!(ids, vec![3, 2, 2, 1]);
    assert_eq!(steps, 1);
}

#[test]
fn new_masks_after_delete_are_resolved_in_final_post_step() {
    let opts = Options {
        max_post_steps: 1,
        ..Options::default()
    };
    let mut calls = 0;
    let (ids, steps) = decode_block(
        vec![3, 7, 7, 7],
        &opts,
        S,
        &mut StdRng::seed_from_u64(42),
        |input| {
            calls += 1;
            if calls == 1 {
                Ok(logits(&[1, 8, 1, 1]))
            } else {
                assert_eq!(input, &[3, 1, 1, 7]);
                Ok(logits(&[1, 1, 1, 9]))
            }
        },
    )
    .unwrap();
    assert_eq!(ids, vec![3, 1, 1, 2]);
    assert_eq!(steps, 2);
}

#[test]
fn schedule_forces_progress_below_threshold() {
    let opts = Options {
        steps: 2,
        threshold: 1.0,
        editing_threshold: 1.0,
        max_post_steps: 0,
        ..Options::default()
    };
    let (ids, steps) = decode_block(vec![7; 4], &opts, S, &mut StdRng::seed_from_u64(42), |_| {
        Ok(logits(&[1; 4]))
    })
    .unwrap();
    assert_eq!(ids, vec![1; 4]);
    assert_eq!(steps, 2);
}

#[test]
fn exhausted_budget_reports_nonconvergence() {
    let opts = Options {
        steps: 32,
        threshold: 1.0,
        max_steps_per_block: 1,
        ..Options::default()
    };
    let err = decode_block(vec![7; 4], &opts, S, &mut StdRng::seed_from_u64(42), |_| {
        Ok(logits(&[1; 4]))
    })
    .unwrap_err();
    assert!(err.to_string().contains("did not converge"));
}

#[test]
fn reserved_prompt_edits_are_rejected() {
    assert!(
        decode_block(
            vec![8, 7],
            &Options::default(),
            S,
            &mut StdRng::seed_from_u64(42),
            |_| unreachable!()
        )
        .is_err()
    );
}

#[test]
fn zero_steps_resolves_all_masks_without_division_by_zero() {
    let opts = Options {
        steps: 0,
        max_post_steps: 0,
        ..Options::default()
    };
    let (ids, steps) = decode_block(vec![7; 4], &opts, S, &mut StdRng::seed_from_u64(42), |_| {
        Ok(logits(&[1; 4]))
    })
    .unwrap();
    assert_eq!(ids, vec![1; 4]);
    assert_eq!(steps, 1);
}
