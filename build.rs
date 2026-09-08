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
}
