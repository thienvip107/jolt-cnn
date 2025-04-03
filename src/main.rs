/// Kích thước ảnh đầu vào 5x5
const IN_SIZE: usize = 5;
/// Kích thước kernel 3x3
const KERNEL_SIZE: usize = 3;
/// Kết quả conv => (5-3+1)=3 => out 3×3=9
const OUT_CONV_SIZE: usize = IN_SIZE - KERNEL_SIZE + 1; 
/// Số lớp (class) ở FC output
const NUM_CLASSES: usize = 10;

use rand::prelude::*;
use mnist::MnistBuilder;
use ndarray::Array3;

fn load_and_partition_mnist() -> (Vec<(Vec<u8>, Vec<u8>)>, Vec<(Vec<u8>, Vec<u8>)>) {
    // Configure MNIST loading: we use 60k training and 10k test as usual
    let mnist = MnistBuilder::new()
        .label_format_digit() // labels as 0-9 digits
        .training_set_length(60_000)
        .test_set_length(10_000)
        .validation_set_length(0)
        .finalize();
    
    // The Mnist struct contains flat vectors for images and labels
    let trn_images = mnist.trn_img;
    let trn_labels = mnist.trn_lbl;
    let tst_images = mnist.tst_img;
    let tst_labels = mnist.tst_lbl;
    
    // Each image is 28x28 = 784 bytes. Partition training images among clients.
    let num_clients = 5;
    let images_per_client = 60_000 / num_clients;
    let mut client_partitions: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for i in 0..num_clients {
        let start = i * images_per_client * 28 * 28;
        let end = start + images_per_client * 28 * 28;
        let img_slice = trn_images[start..end].to_vec();
        let lbl_slice = trn_labels[i * images_per_client..(i+1)*images_per_client].to_vec();
        client_partitions.push((img_slice, lbl_slice));
    }
    // Put all remaining data as test (for simplicity, we use provided test set)
    let test_data = (tst_images, tst_labels);
    (client_partitions, vec![test_data])
}


pub fn main() {
    let (prove_fib, verify_fib) = guest::build_fib();
    let mut rng = rand::thread_rng();

//     // Sinh 6,400 ảnh, mỗi ảnh 16 byte
//     let n_images = 640;
//     let mut all_images = vec![0u8; n_images * 16];
//     for i in 0..(n_images * 16) {
//         all_images[i] = rng.gen_range(0..=255);
//     }
//     // Sinh 6,400 nhãn (ví dụ: nhãn 0 hoặc 1)
//     let mut all_labels = vec![0u8; n_images];
//     for i in 0..n_images {
//         all_labels[i] = rng.gen_range(0..=1);
//     }

//     // Khởi tạo trọng số ban đầu (không toàn 0)
//     let mut weights = [3; 22];

//     // Số batch = 6400 / 64 = 100
//     let batch_size = 64;
//     let n_batches = n_images / batch_size;

//     for b in 0..n_batches {
//         let start_img = b * batch_size * 16;
//         let end_img = start_img + batch_size * 16;
//         let batch_images = &all_images[start_img..end_img];

//         let start_lbl = b * batch_size;
//         let end_lbl = start_lbl + batch_size;
//         let batch_labels = &all_labels[start_lbl..end_lbl];

//         // Gọi hàm fib() cho batch 64 ảnh, truyền trọng số hiện tại để cập nhật
//         let (new_weights, proof) = prove_fib( &batch_images, &batch_labels, weights );
//            // Run one training step in the zkVM
//    println!("Updated weights: {:?}", new_weights);
//    weights = new_weights;
//    let valid = verify_fib(proof);
//    println!("Proof valid: {}", valid);

//     }

    let n_images = 1;
    let mut images = vec![0u8; n_images * 784];
    let mut labels = vec![0u8; n_images];
    for i in 0..(n_images * 784) {
        images[i] = rng.gen_range(0..=255);
    }
    for i in 0..n_images {
        labels[i] = rng.gen_range(0..=9);
    }
    let mut weights = vec![1; 21750];
    println!("images: {:?}", images);
    // Call prove_fib() with images, labels, and the mutable weights.
    // prove_fib() is expected to update weights in place and return a proof.
    let (nweights, proof) = prove_fib(&images, &labels);
       println!("Updated weights: {:?}", nweights);

    // 5) Thử forward xem logits cuối
    // let (logits, _, _) = forward_cnn(
    //     &input_image,
    //     &conv_weights,
    //     &conv_bias,
    //     &fc_weights,
    //     &fc_bias,
    // );
    // println!("Logits cuối: {:?}", logits);
}
