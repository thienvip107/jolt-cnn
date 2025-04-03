#![no_main]

extern crate alloc;

use alloc::vec::Vec;

type Fixed = i32;
const FP_SHIFT: u32 = 8;
const FP_ONE: Fixed = 1 << FP_SHIFT; // 256
const TOTAL_WEIGHTS: usize = 8790;
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
    conv1: [[[Fixed; 5]; 5]; 10],
    fc1: [[Fixed; 864]; 10],
}

impl Model {
    fn new() -> Self {
        Model {
            conv1: [[[1; 5]; 5]; 10],
            fc1: [[1; 864]; 10],
        }
    }

    /// load_weights: Sử dụng slice input (độ dài = TOTAL_WEIGHTS) để khởi tạo các trọng số của mô hình.
    fn load_weights(&mut self, input: &[i32]) {
        let mut idx = 0;
        // conv1: 10×5×5 = 250 phần tử.
        for f in 0..10 {
            for i in 0..5 {
                for j in 0..5 {
                    self.conv1[f][i][j] = input[idx];
                    idx += 1;
                }
            }
        }
        // fc1: 10×864 = 8640 phần tử.
        for i in 0..10 {
            for j in 0..864 {
                self.fc1[i][j] = input[idx];
                idx += 1;
            }
        }
    }

    /// forward: Thực hiện forward pass cho 1 ảnh (784 byte, 28×28).
    /// Các bước:
    /// 1. Chuyển đổi ảnh thành mảng 28×28 dạng Q8.8.
    /// 2. Convolution: Với 10 bộ lọc, kernel 5×5 → output 24×24 cho mỗi bộ lọc.
    /// 3. Pooling: 2×2 max pooling → output 12×12 cho mỗi bộ lọc.
    /// 4. Flatten: Giả sử sử dụng 6 bộ lọc đầu → 6×12×12 = 864 phần tử.
    /// 5. Fully Connected: Tính logits từ 864 đầu vào cho 10 neuron (fc1).
    fn forward(&self, image: &[u8]) -> (Vec<Fixed>, Vec<Vec<Fixed>>) {
        // 1) Chuyển ảnh 784 byte thành mảng 28×28 dạng Q8.8.
        let mut input = [[0; 28]; 28];
        for i in 0..28 {
            for j in 0..28 {
                input[i][j] = fixed_from_pixel(image[i * 28 + j]);
            }
        }
        // 2) Convolution: Với 10 bộ lọc, mỗi output kích thước 24×24.
        let mut conv_out: Vec<Vec<Fixed>> = vec![vec![0; 24 * 24]; 10];
        for f in 0..10 {
            for r in 0..24 {
                for c in 0..24 {
                    let mut sum = 0;
                    for kr in 0..5 {
                        for kc in 0..5 {
                            sum += fixed_mul(self.conv1[f][kr][kc], input[r + kr][c + kc]);
                        }
                    }
                    if sum < 0 { sum = 0; }
                    conv_out[f][r * 24 + c] = sum;
                }
            }
        }
        // 3) Pooling: 2×2 max pooling → output kích thước 12×12 cho mỗi bộ lọc.
        let mut pool_out: Vec<Vec<Fixed>> = vec![vec![0; 12 * 12]; 10];
        for f in 0..10 {
            for r in 0..12 {
                for c in 0..12 {
                    let mut max_val = 0;
                    for dr in 0..2 {
                        for dc in 0..2 {
                            let idx = (r * 2 + dr) * 24 + (c * 2 + dc);
                            if conv_out[f][idx] > max_val {
                                max_val = conv_out[f][idx];
                            }
                        }
                    }
                    pool_out[f][r * 12 + c] = max_val;
                }
            }
        }
        // 4) Flatten: sử dụng 6 bộ lọc đầu → 6×12×12 = 864 phần tử.
        let mut flat = vec![0; 864];
        for f in 0..6 {
            for i in 0..144 {
                flat[f * 144 + i] = pool_out[f][i];
            }
        }
        // 5) FC: Tính logits từ vector flat (864 phần tử) cho 10 neuron.
        let mut logits = vec![0; 10];
        let mut fc_out = vec![0; 10];
        for i in 0..10 {
            let mut sum = 0;
            for j in 0..864 {
                sum += fixed_mul(self.fc1[i][j], flat[j]);
            }
            fc_out[i] = if sum > 0 { sum } else { 0 };
            logits[i] = fc_out[i];
        }
        (logits, vec![flat, fc_out])
    }

    /// backward: Tính gradient cho lớp FC sử dụng loss MSE (one-hot target).
    /// Ở đây target có giá trị FP_ONE tại vị trí nhãn.
    fn backward(&mut self, _image: &[u8], label: u8, logits: Vec<Fixed>, intermediates: Vec<Vec<Fixed>>) {
        let flat = &intermediates[0]; // flat có độ dài 864.
        let mut target = vec![0; 10];
        if (label as usize) < 10 {
            target[label as usize] = FP_ONE;
        }
        let error: Vec<Fixed> = logits.iter().zip(target.iter()).map(|(&l, &t)| l - t).collect();
        let lr: Fixed = 1; // learning rate.
        // Cập nhật trọng số của lớp FC (fc1): kích thước 10×864.
        for i in 0..10 {
            for j in 0..864 {
                let grad = fixed_mul(error[i], flat[j]);
                self.fc1[i][j] -= fixed_mul(lr, grad);
            }
        }
        // (Gradient cho lớp conv chưa được tính trong ví dụ này.)
    }

    fn train_image(&mut self, image: &[u8], label: u8) {
        let (logits, intermediates) = self.forward(image);
        self.backward(image, label, logits, intermediates);
    }

    /// train_batch: Huấn luyện trên 1 batch gồm nhiều ảnh.
    /// Ảnh được lưu liên tiếp (mỗi ảnh 784 byte), labels chứa nhãn tương ứng.
    fn train_batch(&mut self, images: &[u8], labels: &[u8]) {
        let n = labels.len();
        for i in 0..n {
            let start = i * 784;
            let end = start + 784;
            let img = &images[start..end];
            self.train_image(img, labels[i]);
        }
    }

    /// flatten_weights: Gom tất cả trọng số của mô hình thành Vec<i32>.
    /// Ở đây mô hình có trọng số = conv1 (250 phần tử) nối với fc1 (8640 phần tử) = 250 + 8640 = 8890.
    fn flatten_weights(&self) -> Vec<Fixed> {
        let mut w = Vec::with_capacity(250 + 8640);
        for f in 0..10 {
            for i in 0..5 {
                for j in 0..5 {
                    w.push(self.conv1[f][i][j]);
                }
            }
        }
        for i in 0..10 {
            for j in 0..864 {
                w.push(self.fc1[i][j]);
            }
        }
        w
    }
}


#[jolt::provable(stack_size = 10000000, memory_size = 900000000000)]
fn fib(
    images: &[u8],    // n ảnh, mỗi ảnh 784 byte (28×28)
    labels: &[u8],    // n nhãn
    weights: &[u8] // Mảng trọng số đầu vào, kích thước phải bằng TOTAL_WEIGHTS (8790)
) -> Vec<u8> {
    let input_weights: Vec<i32> = weights.chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let mut model = Model::new();
    model.load_weights(&input_weights);

    // Huấn luyện model trên toàn bộ ảnh của batch.
    model.train_batch(images, labels);

    // Lấy trọng số mới từ mô hình.
    let new_weights = model.flatten_weights();
    // Chuyển Vec<i32> thành Vec<u8> (mỗi i32 thành 4 byte little-endian).
    let new_weights_bytes: Vec<u8> = new_weights.iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    new_weights_bytes
}