#!/usr/bin/env python3
"""Task 1.3 preprocessing: denoise classroom audio with DeepFilterNet2.

"""

from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio


def _ensure_torchaudio_backend_compat() -> None:
    """Provide torchaudio.backend.common expected by deepfilternet 0.5.x."""
    try:
        __import__("torchaudio.backend.common")
        return
    except ModuleNotFoundError:
        pass

    common_mod = types.ModuleType("torchaudio.backend.common")
    common_mod.AudioMetaData = object

    backend_mod = types.ModuleType("torchaudio.backend")
    backend_mod.common = common_mod

    sys.modules.setdefault("torchaudio.backend", backend_mod)
    sys.modules["torchaudio.backend.common"] = common_mod


_ensure_torchaudio_backend_compat()

from df import enhance, init_df


TARGET_SR = 16_000
EPS = 1e-12


def resolve_input_path(input_path: str) -> Path:
    """Resolve input path and support a fallback to ./input/ for convenience."""
    path = Path(input_path)
    if path.exists():
        return path

    fallback = Path("input") / input_path
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find input file '{input_path}' or fallback '{fallback}'."
    )


def load_mono_16k_wav(path: Path) -> torch.Tensor:
    """Load a wav file and convert it to mono 16 kHz."""
    waveform_np, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if waveform_np.size == 0:
        raise ValueError(f"Input audio is empty: {path}")

    waveform = torch.from_numpy(waveform_np.T)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    return waveform.squeeze(0)


def denoise_with_deepfilternet2(waveform_16k: torch.Tensor) -> torch.Tensor:
    """Apply DeepFilterNet2 denoising and return audio at 16 kHz."""
    model, df_state, _ = init_df()
    model_sr = int(df_state.sr())

    model_input = waveform_16k.unsqueeze(0)
    if model_sr != TARGET_SR:
        model_input = torchaudio.functional.resample(model_input, TARGET_SR, model_sr)

    with torch.no_grad():
        denoised_model_sr = enhance(model, df_state, model_input)

    if denoised_model_sr.dim() == 1:
        denoised_model_sr = denoised_model_sr.unsqueeze(0)

    if model_sr != TARGET_SR:
        denoised_16k = torchaudio.functional.resample(
            denoised_model_sr, model_sr, TARGET_SR
        )
    else:
        denoised_16k = denoised_model_sr

    denoised_16k = denoised_16k.squeeze(0)

    target_len = waveform_16k.numel()
    cur_len = denoised_16k.numel()
    if cur_len > target_len:
        denoised_16k = denoised_16k[:target_len]
    elif cur_len < target_len:
        denoised_16k = torch.nn.functional.pad(denoised_16k, (0, target_len - cur_len))

    return denoised_16k


def frame_rms_and_power(
    waveform: torch.Tensor, frame_len: int = 400, hop_len: int = 160
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute frame-level RMS and average power (25 ms frames, 10 ms hop at 16 kHz)."""
    if waveform.numel() < frame_len:
        waveform = torch.nn.functional.pad(waveform, (0, frame_len - waveform.numel()))

    frames = waveform.unfold(0, frame_len, hop_len)
    power = torch.mean(frames * frames, dim=1)
    rms = torch.sqrt(power + EPS)
    return rms, power


def estimate_snr_db(
    waveform: torch.Tensor,
    rms_threshold: float = 0.01,
    frame_len: int = 400,
    hop_len: int = 160,
) -> float:
    """Estimate SNR using voiced/unvoiced frame power as signal/noise proxy."""
    rms, power = frame_rms_and_power(waveform, frame_len=frame_len, hop_len=hop_len)
    voiced = rms >= rms_threshold
    unvoiced = ~voiced

    if not torch.any(voiced):
        raise ValueError(
            "No voiced frames detected. Lower --rms-threshold or check input audio."
        )

    signal_power = power[voiced].mean().item()
    noise_power = power[unvoiced].mean().item() if torch.any(unvoiced) else EPS

    return 10.0 * math.log10((signal_power + EPS) / (noise_power + EPS))


def compute_snr_improvement_db(
    input_waveform: torch.Tensor,
    output_waveform: torch.Tensor,
    rms_threshold: float = 0.01,
) -> tuple[float, float, float]:
    """Return (snr_improvement_db, snr_in_db, snr_out_db)."""
    snr_in = estimate_snr_db(input_waveform, rms_threshold=rms_threshold)
    snr_out = estimate_snr_db(output_waveform, rms_threshold=rms_threshold)
    return snr_out - snr_in, snr_in, snr_out


def _plot_waveform(ax: plt.Axes, waveform: torch.Tensor, sr: int, title: str) -> None:
    time_axis = torch.arange(waveform.numel(), dtype=torch.float32) / float(sr)
    ax.plot(time_axis.numpy(), waveform.numpy(), linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)


def _plot_spectrogram(ax: plt.Axes, waveform: torch.Tensor, sr: int, title: str):
    n_fft = 1024
    win_length = 400
    hop_length = 160
    window = torch.hann_window(win_length)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    spec_db = 20.0 * torch.log10(spec.abs().clamp_min(1e-8))
    im = ax.imshow(
        spec_db.numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0.0, waveform.numel() / float(sr), 0.0, sr / 2.0],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return im


def plot_comparison(
    input_waveform: torch.Tensor,
    output_waveform: torch.Tensor,
    sr: int = TARGET_SR,
    output_path: str = "denoise_comparison.png",
) -> None:
    """Plot waveform + spectrogram before/after and save as PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    _plot_waveform(axes[0, 0], input_waveform, sr, "Before Denoising - Waveform")
    _plot_waveform(axes[0, 1], output_waveform, sr, "After Denoising - Waveform")

    im_before = _plot_spectrogram(
        axes[1, 0], input_waveform, sr, "Before Denoising - Spectrogram"
    )
    _plot_spectrogram(axes[1, 1], output_waveform, sr, "After Denoising - Spectrogram")

    cbar = fig.colorbar(im_before, ax=axes[1, :], orientation="horizontal", pad=0.12)
    cbar.set_label("Magnitude (dB)")

    fig.suptitle("DeepFilterNet2 Denoising Comparison", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 1.3: Denoise original_segment.wav using DeepFilterNet2."
    )
    parser.add_argument(
        "--input",
        default="original_segment.wav",
        help="Input mono wav file path (default: original_segment.wav).",
    )
    parser.add_argument(
        "--output",
        default="denoised_segment.wav",
        help="Output denoised wav path (default: denoised_segment.wav).",
    )
    parser.add_argument(
        "--plot",
        default="denoise_comparison.png",
        nargs="?",
        const="denoise_comparison.png",
        help="Output comparison plot path (default: denoise_comparison.png).",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=0.01,
        help="RMS threshold used to detect voiced frames (default: 0.01).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args.input)
    input_waveform = load_mono_16k_wav(input_path)
    denoised_waveform = denoise_with_deepfilternet2(input_waveform)

    sf.write(args.output, denoised_waveform.cpu().numpy(), TARGET_SR)

    snr_improvement, snr_in, snr_out = compute_snr_improvement_db(
        input_waveform, denoised_waveform, rms_threshold=args.rms_threshold
    )

    print(f"Input SNR (voiced/unvoiced proxy):  {snr_in:.2f} dB")
    print(f"Output SNR (voiced/unvoiced proxy): {snr_out:.2f} dB")
    print(f"SNR improvement:                    {snr_improvement:.2f} dB")

    plot_comparison(
        input_waveform,
        denoised_waveform,
        sr=TARGET_SR,
        output_path=args.plot,
    )

    print(f"Saved denoised audio to: {args.output}")
    print(f"Saved comparison figure to: {args.plot}")


if __name__ == "__main__":
    main()
