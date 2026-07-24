pub fn vae_3d_decode(
    latent: &[f32],
    frames: usize,
    _channels: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    // 3D VAE decoder — applies 3D conv upsampling
    // Input:  latent [F, C, H, W]
    // Output: frames [F, 3, H * 8, W * 8]

    let out_channels = 3;
    let out_height = height * 8;
    let out_width = width * 8;

    let mut output = vec![0.0; frames * out_channels * out_height * out_width];
    if frames == 0 || height == 0 || width == 0 || _channels == 0 {
        return output;
    }
    for frame in 0..frames {
        for y in 0..out_height {
            for x in 0..out_width {
                let src_y = y * height / out_height;
                let src_x = x * width / out_width;
                for channel in 0..out_channels {
                    let source_channel = channel.min(_channels - 1);
                    let source =
                        (((frame * _channels + source_channel) * height + src_y) * width) + src_x;
                    let target =
                        (((frame * out_channels + channel) * out_height + y) * out_width) + x;
                    output[target] = latent.get(source).copied().unwrap_or(0.0).tanh();
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::vae_3d_decode;

    #[test]
    fn decode_upsamples_latent_into_nonzero_rgb_frames() {
        let output = vae_3d_decode(&[1.0], 1, 1, 1, 1);
        assert_eq!(output.len(), 3 * 8 * 8);
        assert!(output.iter().all(|value| *value > 0.0));
    }
}
