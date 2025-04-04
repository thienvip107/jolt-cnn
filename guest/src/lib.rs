#![no_main]

extern crate alloc;

use alloc::vec::Vec;

type Fixed = i32;
const FP_SHIFT: u32 = 8;
const FP_ONE: Fixed = 1 << FP_SHIFT; // 256
// Mô hình rút gọn:
// conv: 4 bộ lọc, kernel 3×3 → 4×3×3 = 36 trọng số.
// FC: 10 neuron, mỗi neuron nhận 676 đầu vào → 10×676 = 6760 trọng số.
// Tổng trọng số = 36 + 6760 = 6796.
const TOTAL_WEIGHTS: usize = 6796;
use serde::{Serialize, Deserialize};
// Multiply Q8.8
fn fixed_mul(a: Fixed, b: Fixed) -> Fixed {
    let tmp = (a as i64) * (b as i64);
    (tmp >> FP_SHIFT) as Fixed
}

// Convert pixel [0..255] => Q8.8
fn fixed_from_pixel(px: u8) -> Fixed {
    (px as i32 * FP_ONE) / 255
}
struct Model {
    conv: [[[Fixed; 3]; 3]; 4],
    fc: [[Fixed; 676]; 10],
}

impl Model {
    fn new() -> Self {
        Model {
            conv: [[[1; 3]; 3]; 4],
            fc: [[1; 676]; 10],
        }
    }

    /// load_weights: nhận đầu vào là slice `[u8]` (độ dài = TOTAL_WEIGHTS * 4),
    /// chuyển sang Vec<i32> và nạp vào các trường của mô hình.
    fn load_weights(&mut self, input: &[u8]) {
        assert!(
            input.len() == TOTAL_WEIGHTS * 4,
            "Weights must be {} bytes",
            TOTAL_WEIGHTS * 4
        );
        let input_i32: Vec<i32> = input
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let mut idx = 0;
        // Conv: 4×3×3 = 36 phần tử.
        for f in 0..4 {
            for i in 0..3 {
                for j in 0..3 {
                    self.conv[f][i][j] = input_i32[idx];
                    idx += 1;
                }
            }
        }
        // FC: 10×676 = 6760 phần tử.
        for i in 0..10 {
            for j in 0..676 {
                self.fc[i][j] = input_i32[idx];
                idx += 1;
            }
        }
    }


    /// forward: Thực hiện forward pass cho 1 ảnh (784 byte, 28×28).
    /// Các bước:
    /// 1. Chuyển đổi ảnh thành mảng 28×28 dạng Q8.8.
    /// 2. Convolution: Với 4 bộ lọc, kernel 3×3, valid convolution → output kích thước 26×26.
    /// 3. Pooling: 2×2 max pooling → output kích thước 13×13 cho mỗi bộ lọc.
    /// 4. Flatten: Nối tất cả các output pooling thành vector 4×13×13 = 676 phần tử.
    /// 5. Fully Connected: Tính logits từ 676 đầu vào cho 10 neuron (fc).
    fn forward(&self, image: &[u8]) -> (Vec<Fixed>, Vec<Vec<Fixed>>) {
        // 1) Chuyển ảnh 784 byte thành mảng 28×28 dạng Q8.8.
        let mut input = [[0; 28]; 28];
        for i in 0..28 {
            for j in 0..28 {
                input[i][j] = fixed_from_pixel(image[i * 28 + j]);
            }
        }
        // 2) Convolution: Với 4 bộ lọc, kernel 3×3, output: 26×26.
        let mut conv_out: Vec<Vec<Fixed>> = vec![vec![0; 26 * 26]; 4];
        for f in 0..4 {
            for r in 0..26 {
                for c in 0..26 {
                    let mut sum = 0;
                    for kr in 0..3 {
                        for kc in 0..3 {
                            sum += fixed_mul(self.conv[f][kr][kc], input[r + kr][c + kc]);
                        }
                    }
                    if sum < 0 { sum = 0; }
                    conv_out[f][r * 26 + c] = sum;
                }
            }
        }
        // 3) Pooling: 2×2 max pooling trên mỗi conv_out, output: 13×13 cho mỗi filter.
        let mut pool_out: Vec<Vec<Fixed>> = vec![vec![0; 13 * 13]; 4];
        for f in 0..4 {
            for r in 0..13 {
                for c in 0..13 {
                    let mut max_val = 0;
                    for dr in 0..2 {
                        for dc in 0..2 {
                            let idx = (r * 2 + dr) * 26 + (c * 2 + dc);
                            if conv_out[f][idx] > max_val {
                                max_val = conv_out[f][idx];
                            }
                        }
                    }
                    pool_out[f][r * 13 + c] = max_val;
                }
            }
        }
        // 4) Flatten: Nối output của 4 bộ lọc → 4×13×13 = 676 phần tử.
        let mut flat = vec![0; 676];
        for f in 0..4 {
            for i in 0..(13 * 13) {
                flat[f * 169 + i] = pool_out[f][i];
            }
        }
        // 5) Fully Connected: Tính logits từ flat cho 10 neuron.
        let mut logits = vec![0; 10];
        let mut fc_out = vec![0; 10];
        for i in 0..10 {
            let mut sum = 0;
            for j in 0..676 {
                sum += fixed_mul(self.fc[i][j], flat[j]);
            }
            fc_out[i] = if sum > 0 { sum } else { 0 };
            logits[i] = fc_out[i];
        }
        (logits, vec![flat, fc_out])
    }

    /// backward: Tính gradient cho lớp FC sử dụng loss MSE (one-hot target: FP_ONE tại vị trí nhãn).
    fn backward(&mut self, _image: &[u8], label: u8, logits: Vec<Fixed>, intermediates: Vec<Vec<Fixed>>) {
        let flat = &intermediates[0]; // flat có độ dài 676.
        let mut target = vec![0; 10];
        if (label as usize) < 10 {
            target[label as usize] = FP_ONE;
        }
        let error: Vec<Fixed> = logits.iter().zip(target.iter()).map(|(&l, &t)| l - t).collect();
        let lr: Fixed = 1; // learning rate.
        // Cập nhật trọng số của FC (fc): kích thước 10×676.
        for i in 0..10 {
            for j in 0..676 {
                let grad = fixed_mul(error[i], flat[j]);
                self.fc[i][j] -= fixed_mul(lr, grad);
            }
        }
    }

    fn train_image(&mut self, image: &[u8], label: u8) {
        let (logits, intermediates) = self.forward(image);
        self.backward(image, label, logits, intermediates);
    }

    /// train_batch: Huấn luyện trên một batch (ảnh lưu liên tiếp, mỗi ảnh 784 byte).
    fn train_batch(&mut self, images: &[u8], labels: &[u8]) {
        let n = labels.len();
        for i in 0..n {
            let start = i * 784;
            let end = start + 784;
            let img = &images[start..end];
            self.train_image(img, labels[i]);
        }
    }

    /// flatten_weights: Gom tất cả trọng số của mô hình thành một Vec<i32>.
    /// Mô hình có trọng số = conv (4×3×3 = 36) nối với fc (10×676 = 6760) = 36 + 6760 = 6796.
    fn flatten_weights(&self) -> Vec<Fixed> {
        let mut w = Vec::with_capacity(36 + 6760);
        for f in 0..4 {
            for i in 0..3 {
                for j in 0..3 {
                    w.push(self.conv[f][i][j]);
                }
            }
        }
        for i in 0..10 {
            for j in 0..676 {
                w.push(self.fc[i][j]);
            }
        }
        w
    }
}

#[jolt::provable(stack_size = 90000000, memory_size = 900000000000, max_input_size = 9000000, max_output_size = 9000000)]
fn fib(
    images: &[u8],    // n ảnh, mỗi ảnh 784 byte (28×28)
    labels: &[u8],    // n nhãn
    weights: &[u8] // Mảng trọng số đầu vào, kích thước phải bằng TOTAL_WEIGHTS (8790)
) -> Vec<u8> {
    let mut model = Model::new();
    model.load_weights(weights);

    // Huấn luyện model trên toàn bộ ảnh của batch.
    model.train_image(images, labels[0]);

    // Lấy trọng số mới từ mô hình.
    let new_weights = model.flatten_weights();
    // Chuyển Vec<i32> thành Vec<u8> (mỗi i32 thành 4 byte little-endian).
    let new_weights_bytes: Vec<u8> = new_weights.iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    new_weights_bytes
}