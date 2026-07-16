use super::codec::TernaryCodecError;
use super::codec::TernaryPackedTensor;
use super::pack::{pack_ternary_codes, unpack_ternary_codes, validate_no_reserved_codes};
use super::reference::ternary_gemv_reference;

#[test]
fn test_ternary_pack_unpack_roundtrip() {
    let values: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1];
    let packed = pack_ternary_codes(&values).unwrap();
    let unpacked = unpack_ternary_codes(&packed, values.len()).unwrap();
    assert_eq!(values, unpacked);

    // Non-multiple-of-4
    let values2: Vec<i8> = vec![-1, 1, 1, -1, 0];
    let packed2 = pack_ternary_codes(&values2).unwrap();
    let unpacked2 = unpack_ternary_codes(&packed2, values2.len()).unwrap();
    assert_eq!(values2, unpacked2);

    // Single value
    let values3: Vec<i8> = vec![1];
    let packed3 = pack_ternary_codes(&values3).unwrap();
    let unpacked3 = unpack_ternary_codes(&packed3, 1).unwrap();
    assert_eq!(values3, unpacked3);
}

#[test]
fn test_ternary_rejects_reserved_code_11() {
    // Manually construct a byte with code 0b11 in position 0
    let bad_byte: u8 = 0b11; // bits 0-1 = 11
    let result = unpack_ternary_codes(&[bad_byte], 1);
    assert!(matches!(result, Err(TernaryCodecError::ReservedCode11)));

    // validate_no_reserved_codes should also reject it
    let result = validate_no_reserved_codes(&[bad_byte], 1);
    assert!(matches!(result, Err(TernaryCodecError::ReservedCode11)));

    // Reject invalid weight values at pack time (2, -2, etc.)
    assert!(matches!(
        pack_ternary_codes(&[2i8]),
        Err(TernaryCodecError::InvalidWeight(2))
    ));
    assert!(matches!(
        pack_ternary_codes(&[-2i8]),
        Err(TernaryCodecError::InvalidWeight(-2))
    ));
}

#[test]
fn test_ternary_payload_size_matches_layout() {
    let rows = 4usize;
    let cols = 64usize;
    let group_size = 16usize;
    let groups_per_row = cols.div_ceil(group_size); // 4
    let bytes_per_group = (group_size + 3) / 4; // 4
    let total_values = rows * cols; // 256
    let expected_bytes = rows * groups_per_row * bytes_per_group; // 4*4*4 = 64

    let values: Vec<i8> = (0..total_values)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let packed = pack_ternary_codes(&values).unwrap();
    assert_eq!(packed.len(), expected_bytes);

    // Verify bytes_per_group
    assert_eq!(bytes_per_group, 4);
    // groups_per_row * bytes_per_group = 16 bytes per row
    assert_eq!(groups_per_row * bytes_per_group, 16);

    // Build a TernaryPackedTensor matching this layout
    let scales: Vec<half::f16> = (0..rows * groups_per_row)
        .map(|_| half::f16::from_f32(1.0))
        .collect();

    let tensor = TernaryPackedTensor {
        rows,
        cols,
        group_size,
        groups_per_row,
        bytes_per_group,
        codes: packed,
        scales,
    };

    assert_eq!(tensor.codes.len(), expected_bytes);
    assert_eq!(tensor.scales.len(), rows * groups_per_row);
    assert_eq!(tensor.groups_per_row, groups_per_row);
    assert_eq!(tensor.bytes_per_group, bytes_per_group);
}

#[test]
fn test_ternary_gemv_reference_matches_raw_reconstruction() {
    // Small matrix: 2 rows, 8 cols, group_size=4
    let rows = 2usize;
    let cols = 8usize;
    let group_size = 4usize;

    // Row 0 weights: [-1, 0, 1, -1,  0, 1, -1, 0]
    // Row 1 weights: [ 1, 1, 0, -1, -1, 0,  1, 1]
    let row0: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1, 0];
    let row1: Vec<i8> = vec![1, 1, 0, -1, -1, 0, 1, 1];
    let all_weights: Vec<i8> = row0.iter().chain(row1.iter()).copied().collect();

    let packed = pack_ternary_codes(&all_weights).unwrap();

    // Per-group scales (2 groups per row, 2 rows = 4 scales)
    let scales: Vec<f32> = vec![2.0, 0.5, 3.0, 1.5];

    // Activations: 8 elements
    let activations: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let result =
        ternary_gemv_reference(&activations, &packed, &scales, rows, cols, group_size).unwrap();

    // Manual computation:
    // Row 0:
    //   Group 0: weights [-1,0,1,-1] * acts [1,2,3,4] = -1*1 + 0*2 + 1*3 + (-1)*4 = -1+0+3-4 = -2
    //            scaled by 2.0 -> -4
    //   Group 1: weights [0,1,-1,0] * acts [5,6,7,8] = 0*5 + 1*6 + (-1)*7 + 0*8 = 6-7 = -1
    //            scaled by 0.5 -> -0.5
    //   Row 0 total: -4.5
    // Row 1:
    //   Group 0: weights [1,1,0,-1] * acts [1,2,3,4] = 1*1 + 1*2 + 0*3 + (-1)*4 = 1+2+0-4 = -1
    //            scaled by 3.0 -> -3
    //   Group 1: weights [-1,0,1,1] * acts [5,6,7,8] = -1*5 + 0*6 + 1*7 + 1*8 = -5+0+7+8 = 10
    //            scaled by 1.5 -> 15
    //   Row 1 total: 12

    let expected: Vec<f32> = vec![-4.5, 12.0];
    assert_eq!(result.len(), expected.len());
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: result={a}, expected={b}");
    }
}

#[test]
fn test_ternary_group_size_padding_is_deterministic() {
    // When values.len() is not a multiple of 4, padding must be deterministic.
    let values: Vec<i8> = vec![-1, 1]; // 2 values -> fits in 1 byte, 2 padding nibbles
    let packed1 = pack_ternary_codes(&values).unwrap();
    let packed2 = pack_ternary_codes(&values).unwrap();
    assert_eq!(packed1, packed2);

    // Unpacking should return only the original 2 values despite padding
    let unpacked = unpack_ternary_codes(&packed1, 2).unwrap();
    assert_eq!(unpacked, values);

    // validate_no_reserved_codes should accept padding
    validate_no_reserved_codes(&packed1, 2).unwrap();
}

#[test]
fn test_ternary_rejects_invalid_weight() {
    assert!(matches!(
        pack_ternary_codes(&[2i8]),
        Err(TernaryCodecError::InvalidWeight(2))
    ));
    assert!(matches!(
        pack_ternary_codes(&[-2i8]),
        Err(TernaryCodecError::InvalidWeight(-2))
    ));
    assert!(matches!(
        pack_ternary_codes(&[3i8]),
        Err(TernaryCodecError::InvalidWeight(3))
    ));
}
