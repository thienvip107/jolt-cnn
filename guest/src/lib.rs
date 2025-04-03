#![no_main]

extern crate alloc;

use alloc::vec::Vec;

type Fixed = i32;
const FP_SHIFT: u32 = 8;
const FP_ONE: Fixed = 1 << FP_SHIFT; // 256
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

const TOTAL_WEIGHTS: usize = 21750;

#[jolt::provable(stack_size = 10000000, memory_size = 900000000000)]
fn fib(
    images: &[u8],  // Dữ liệu: n ảnh, mỗi ảnh 784 byte (28×28)
    labels: &[u8]   // n nhãn
) -> Vec<u8> {
    // Kiểm tra dữ liệu vào
    // Khởi tạo trọng số của mô hình ở dạng mảng phẳng:
    // Conv: 6 bộ lọc, mỗi bộ lọc 5×5 = 150 phần tử.
    let mut conv_weights: [Fixed; 150] = [1; 150];
    // FC: 10 neuron, mỗi neuron 864 trọng số = 8640 phần tử.
    let mut fc_weights: [Fixed; 8640] = [1; 8640];

    // Lấy ảnh và nhãn đầu tiên.
    let img = &images[0..784];
    let _label = labels[0]; // không sử dụng trong dummy update

    // 1) Chuyển ảnh thành mảng 28×28 dạng Q8.8, lưu dưới dạng mảng phẳng 784 phần tử.
    let mut input: [Fixed; 784] = [0; 784];
    for r in 0..28 {
        for c in 0..28 {
            input[r * 28 + c] = fixed_from_pixel(img[r * 28 + c]);
        }
    }

    // 2) Convolution: Với mỗi trong 6 bộ lọc, thực hiện valid convolution với kernel 5×5.
    // Output mỗi bộ lọc: 24×24 = 576 phần tử, tổng conv_out có độ dài 6×576 = 3456.
    let mut conv_out: Vec<Fixed> = vec![0; 6 * 576];
    for f in 0..6 {
        for r in 0..24 {
            for c in 0..24 {
                let mut sum = 0;
                for kr in 0..5 {
                    for kc in 0..5 {
                        // Trọng số của bộ lọc f tại vị trí (kr,kc):
                        let w = conv_weights[f * 25 + kr * 5 + kc];
                        // Pixel tại vị trí (r+kr, c+kc)
                        let pixel = input[(r + kr) * 28 + (c + kc)];
                        sum += fixed_mul(w, pixel);
                    }
                }
                if sum < 0 { sum = 0; }
                conv_out[f * 576 + r * 24 + c] = sum;
            }
        }
    }

    // 3) Pooling: 2×2 max pooling trên conv_out cho mỗi bộ lọc, output mỗi bộ lọc có kích thước 12×12.
    // Tích lũy kết quả pooling của 6 bộ lọc thành vector pool_out có độ dài 6×144 = 864.
    let mut pool_out: Vec<Fixed> = vec![0; 6 * 144];
    for f in 0..6 {
        for r in 0..12 {
            for c in 0..12 {
                let mut max_val = 0;
                for dr in 0..2 {
                    for dc in 0..2 {
                        let idx = f * 576 + (r * 2 + dr) * 24 + (c * 2 + dc);
                        if conv_out[idx] > max_val {
                            max_val = conv_out[idx];
                        }
                    }
                }
                pool_out[f * 144 + r * 12 + c] = max_val;
            }
        }
    }
    // 4) Flatten: pool_out đã có độ dài 864.
    let flat = &pool_out; // slice có độ dài 864

    // 5) Fully Connected: tính logits cho 10 đầu ra, sử dụng fc_weights.
    let mut logits = [0; 10];
    for neuron in 0..10 {
        let mut sum = 0;
        for j in 0..864 {
            sum += fixed_mul(fc_weights[neuron * 864 + j], flat[j]);
        }
        logits[neuron] = sum;
    }
    
    // 6) Dummy update: tăng tất cả trọng số của conv và fc lên 1.
    for i in 0..conv_weights.len() {
        conv_weights[i] += 1;
    }
    for i in 0..fc_weights.len() {
        fc_weights[i] += 1;
    }

    let mut new_weights: Vec<Fixed> = Vec::with_capacity(TOTAL_WEIGHTS);
    new_weights.extend_from_slice(&conv_weights);
    new_weights.extend_from_slice(&fc_weights);
    let bytes: Vec<u8> = new_weights.iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    bytes
}
