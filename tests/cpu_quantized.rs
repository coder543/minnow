use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use minnow::{
    container::Encoding,
    quant::{self, Weights},
};

// Independent row-major decoding validates direct packed reads, offsets, scales,
// repeated expert segments, and all projection widths used by mini and flash.
#[test]
fn cpu_quantized_experts_match_row_major_oracle() -> anyhow::Result<()> {
    Device::Cpu.with_context(|| -> anyhow::Result<()> {
        for encoding in [
            Encoding::I8Sym,
            Encoding::I8Mma,
            Encoding::I4Sym,
            Encoding::I4Mma,
            Encoding::Nvfp4,
        ] {
            for input in [192, 512, 1024, 2048, 4096] {
                let (experts, out, rows) = (3, 24, 7);
                let mut codes = vec![];
                let mut scales = vec![];
                let mut globals = vec![];
                let mut decoded = vec![];
                for e in 0..experts {
                    let values: Vec<_> = (0..out * input)
                        .map(|i| ((i * 17 + e * 29) % 257) as f32 / 257. - 0.5)
                        .collect();
                    if encoding == Encoding::Nvfp4 {
                        let (c, s, g) = quant::nvfp4::encode(&values)?;
                        decoded.extend(quant::nvfp4::decode(&c, &s, g)?);
                        let (c, s) = quant::nvfp4::pack(&c, &s, input, false)?;
                        codes.extend(c);
                        scales.extend(s);
                        globals.push(g);
                    } else {
                        let (c, s) = quant::encode(&values, encoding.row_major(), 64)?;
                        decoded.extend(
                            quant::decode(&c, &s, encoding.row_major(), 64)?
                                .into_iter()
                                .map(|v| bf16::from_f32(v).to_f32()),
                        );
                        let (c, s) =
                            quant::repack(&c, &s, encoding.row_major(), encoding, 64, input)?;
                        codes.extend(c);
                        scales.extend(s);
                    }
                }
                // Nonzero storage offsets must still address the selected expert.
                codes.insert(0, 0);
                let code_len = codes.len() - 1;
                let scales = if encoding == Encoding::Nvfp4 {
                    let len = scales.len();
                    scales.insert(0, 0);
                    Tensor::from_vec(scales, len + 1, &Device::Cpu)?.narrow(0, 1, len)?
                } else {
                    let mut v = vec![f16::ZERO];
                    v.extend(
                        scales
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|v| f16::from_bits(u16::from_le_bytes(*v))),
                    );
                    let len = v.len() - 1;
                    Tensor::from_vec(v, len + 1, &Device::Cpu)?.narrow(0, 1, len)?
                };
                let weight = Weights {
                    int8_activations: false,
                    codes: Tensor::from_vec(codes, code_len + 1, &Device::Cpu)?
                        .narrow(0, 1, code_len)?,
                    scales,
                    global_scales: if globals.is_empty() {
                        None
                    } else {
                        Some(Tensor::from_vec(globals, experts, &Device::Cpu)?)
                    },
                    shape: (experts, out, input),
                    encoding,
                    group_size: if encoding == Encoding::Nvfp4 { 16 } else { 64 },
                };
                let x = Tensor::from_vec(
                    (0..rows * input)
                        .map(|i| ((i * 13 % 101) as f32 - 50.) / 59.)
                        .collect::<Vec<_>>(),
                    (rows, input),
                    &Device::Cpu,
                )?;
                let segments = [(2, 1), (0, 3), (2, 1), (1, 2)];
                let actual = weight.grouped(&x, &segments)?.to_vec2::<f32>()?;
                let selected = if encoding == Encoding::Nvfp4 {
                    quant::nvfp4::quantize_rows(&x)?
                } else {
                    x.clone()
                };
                let full = Tensor::from_vec(decoded, (experts, out, input), &Device::Cpu)?;
                let mut row = 0;
                for (e, n) in segments {
                    let expected = selected
                        .narrow(0, row, n)?
                        .matmul(&full.get(e)?.t()?)?
                        .to_vec2::<f32>()?;
                    assert_eq!(actual[row..row + n], expected, "{encoding:?}, K={input}");
                    row += n;
                }
                assert!(weight.grouped(&x, &[(3, 7)]).is_err());
                assert!(weight.grouped(&x, &[(0, 6)]).is_err());
            }
        }
        Ok(())
    })
}

#[test]
fn cpu_rejects_bf16_before_loading_weights() {
    let error = minnow::model::Model::load(
        std::path::Path::new("nonexistent"),
        DType::BF16,
        &Device::Cpu,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("CPU execution requires FP32"));
}
