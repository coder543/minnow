use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda");
    println!("cargo:rerun-if-env-changed=NVCC");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args([
            "-ptx",
            "-arch=compute_80",
            "-O3",
            "--fmad=false",
            "src/kernels.cu",
            "-o",
        ])
        .arg(output.join("minnow.ptx"))
        .status()
        .expect("nvcc is required when building with --features cuda");
    assert!(status.success(), "compiling minnow CUDA kernels failed");
}
