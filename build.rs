use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=MINNOW_CUDA_ARCH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    // Keep the portable Ampere PTX baseline unless explicitly targeting a newer GPU.
    // This affects only minnow's custom kernels; Candle and cuBLAS build separately.
    let arch = env::var("MINNOW_CUDA_ARCH").unwrap_or_else(|_| "compute_80".into());
    assert!(
        arch.starts_with("compute_"),
        "MINNOW_CUDA_ARCH must be a PTX target such as compute_121"
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .arg(format!("-arch={arch}"))
        .args(["-ptx", "-O3", "--fmad=false", "src/kernels.cu", "-o"])
        .arg(output.join("minnow.ptx"))
        .status()
        .expect("nvcc is required when building with --features cuda");
    assert!(status.success(), "compiling minnow CUDA kernels failed");
    // Native block-scaled FP4 is separate from portable BF16 PTX. SM120f
    // supports SM120/121; the runtime checks capability before loading weights.
    let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args([
            "-ptx",
            "-arch=compute_120f",
            "-O3",
            "--fmad=false",
            "src/cuda/nvfp4.cu",
            "-o",
        ])
        .arg(output.join("minnow-nvfp4.ptx"))
        .status()
        .expect("nvcc is required");
    assert!(status.success(), "compiling native NVFP4 kernels failed");
    println!("cargo:rerun-if-changed=vendor/flash-attention");
    let includes = cudaforge::DependencyManager::new()
        .with_cutlass(Some("7d49e6c7e2f8896c47f586706e67e1fb215529dc"))
        .fetch_all(&output)
        .expect("fetching pinned CUTLASS headers");
    let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args([
            "-ptx",
            "-arch=compute_80",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Ivendor/flash-attention",
        ])
        .args(includes)
        .args(["src/cuda/flash.cu", "-o"])
        .arg(output.join("minnow-flash.ptx"))
        .status()
        .expect("nvcc is required");
    assert!(status.success(), "compiling block FlashAttention failed");
}
