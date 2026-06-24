pub fn conv3d(
    input: &[f32],
    weight: &[f32],
    n: usize, c_in: usize, f_in: usize, h_in: usize, w_in: usize,
    c_out: usize, kt: usize, kh: usize, kw: usize,
    stride_t: usize, stride_h: usize, stride_w: usize,
    pad_t: usize, pad_h: usize, pad_w: usize,
) -> Vec<f32> {
    let f_out = (f_in + 2 * pad_t - kt) / stride_t + 1;
    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;

    let mut output = vec![0.0; n * c_out * f_out * h_out * w_out];

    for batch in 0..n {
        for oc in 0..c_out {
            for t in 0..f_out {
                for y in 0..h_out {
                    for x in 0..w_out {
                        let mut sum = 0.0;
                        for ic in 0..c_in {
                            for dt in 0..kt {
                                for dy in 0..kh {
                                    for dx in 0..kw {
                                        let in_t = t as isize * stride_t as isize + dt as isize - pad_t as isize;
                                        let in_y = y as isize * stride_h as isize + dy as isize - pad_h as isize;
                                        let in_x = x as isize * stride_w as isize + dx as isize - pad_w as isize;

                                        if in_t >= 0 && in_t < f_in as isize &&
                                           in_y >= 0 && in_y < h_in as isize &&
                                           in_x >= 0 && in_x < w_in as isize {
                                            
                                            let in_idx = batch * (c_in * f_in * h_in * w_in)
                                                       + ic * (f_in * h_in * w_in)
                                                       + (in_t as usize) * (h_in * w_in)
                                                       + (in_y as usize) * w_in
                                                       + (in_x as usize);

                                            let w_idx = oc * (c_in * kt * kh * kw)
                                                      + ic * (kt * kh * kw)
                                                      + dt * (kh * kw)
                                                      + dy * kw
                                                      + dx;

                                            sum += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let out_idx = batch * (c_out * f_out * h_out * w_out)
                                    + oc * (f_out * h_out * w_out)
                                    + t * (h_out * w_out)
                                    + y * w_out
                                    + x;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    output
}
