import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# Set random seed for reproducibility
np.random.seed(42)


def jitter(x, sigma=0.8):
    """Add random noise to the signal"""
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    """Apply random amplitude scaling to the signal"""
    # Generate scaling factors (mean=1.0, std=sigma) for normalized signals
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):  # For each channel
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    """Randomly permute segments of the time series"""
    x_np = x if isinstance(x, np.ndarray) else x.numpy()
    N, C, L = x_np.shape
    orig_steps = np.arange(L)

    ret = np.empty_like(x_np)
    for i in range(N):
        # Choose number of segments
        n_seg = np.random.randint(1, max_segments + 1)
        if n_seg > 1:
            # Split and permute indices
            if seg_mode == "random":
                split_points = np.random.choice(L - 2, n_seg - 1, replace=False) + 1
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, n_seg)
            order = np.random.permutation(len(splits))
            warp = np.concatenate([splits[o] for o in order])
        else:
            warp = orig_steps

        ret[i] = x_np[i][:, warp]
    return ret


def frequency_masking(x, mask_ratio=0.1):
    """Apply frequency masking by zeroing out random frequency components"""
    x_np = x if isinstance(x, np.ndarray) else x.numpy()
    N, C, L = x_np.shape
    ret = np.empty_like(x_np)
    
    for i in range(N):
        for c in range(C):
            # FFT to frequency domain
            fft_signal = np.fft.fft(x_np[i, c, :])
            
            # Calculate number of frequencies to mask
            n_freqs = len(fft_signal)
            n_mask = int(n_freqs * mask_ratio)
            
            # Randomly select frequency indices to mask (excluding DC component)
            freq_indices = np.random.choice(range(1, n_freqs), size=n_mask, replace=False)
            
            # Zero out selected frequency components
            fft_masked = fft_signal.copy()
            fft_masked[freq_indices] = 0
            
            # IFFT back to time domain
            ret[i, c, :] = np.real(np.fft.ifft(fft_masked))
    
    return ret


def spectral_temporal_augmentation(x, max_segments=8, mask_ratio=0.1):
    """Combined spectral perturbation and temporal permutation augmentation"""
    # First apply temporal permutation with maximum segments
    temporal_aug = permutation(x, max_segments=max_segments, seg_mode="random")
    
    # Then apply frequency masking to the temporally permuted signal
    spectral_temporal_aug = frequency_masking(temporal_aug, mask_ratio=mask_ratio)
    
    return spectral_temporal_aug


def tstcc_weak_augmentation(x, mask_ratio=0.1):
    """TSTCC-style weak augmentation: minimal frequency masking only"""
    return frequency_masking(x, mask_ratio=mask_ratio)


def tstcc_strong_augmentation(x, max_segments=20, mask_ratio=0.3):
    """TSTCC-style strong augmentation: aggressive temporal + spectral disruption"""
    # Maximum temporal permutation
    temporal_aug = permutation(x, max_segments=max_segments, seg_mode="random")
    # Heavy frequency masking
    strong_aug = frequency_masking(temporal_aug, mask_ratio=mask_ratio)
    return strong_aug


class Config:
    """Simple config class for augmentation parameters"""

    class Augmentation:
        jitter_scale_ratio = 0.001  # For scaling around 1.0 (need to increase this as well! -> basically was set to 2)
        jitter_ratio = 0.001      # Small noise for normalized signals (
        max_seg = 8
        max_segment_freq = 20
        freq_mask_ratio = 0.2     # Frequency masking ratio

    augmentation = Augmentation()


def DataTransform(sample, config):
    """Apply weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg),
                        config.augmentation.jitter_ratio)
    return weak_aug, strong_aug


# Generate synthetic ECG signal using neurokit
duration = 10  # seconds
sampling_rate = 1_000  # Hz
ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=70, noise=0.1)

# Normalize the ECG signal (z-score normalization)
ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

# Reshape to expected format: (batch_size=1, channels=1, length=100000)
ecg_reshaped = ecg_signal.reshape(1, 1, -1)

# Create config
config = Config()

# Apply augmentations
weak_aug, strong_aug = DataTransform(ecg_reshaped, config)
freq_aug = frequency_masking(ecg_reshaped, config.augmentation.freq_mask_ratio)
spectral_temporal_aug = spectral_temporal_augmentation(ecg_reshaped, 
                                                       max_segments=config.augmentation.max_segment_freq,
                                                       mask_ratio=config.augmentation.freq_mask_ratio)

# TSTCC-style augmentations
mask_ratio_strong = 0.1
mask_ratio_weak = 0.1
tstcc_weak = tstcc_weak_augmentation(ecg_reshaped, mask_ratio=mask_ratio_weak)
tstcc_strong = tstcc_strong_augmentation(ecg_reshaped,
                                         max_segments=config.augmentation.max_segment_freq,
                                         mask_ratio=mask_ratio_strong
                                         )

# Create time axis
time_axis = np.linspace(0, duration, len(ecg_signal))

# Create the plots in 3x3 grid format
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# Define zoom window for better visualization (first 4 seconds)
zoom_end = 5.0
zoom_mask = time_axis <= zoom_end

# Row 1: Original strategies
# Original signal
axes[0, 0].plot(time_axis[zoom_mask], ecg_signal[zoom_mask], 'b-', linewidth=1.5)
axes[0, 0].set_title('Original Reference ECG Signal\n(Normalized)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Amplitude', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(0, zoom_end)

# Weak augmentation (scaling)
axes[0, 1].plot(time_axis[zoom_mask], weak_aug[0, 0, zoom_mask], 'g-', linewidth=1.5)
axes[0, 1].set_title(f'Legacy Weak Augmentation\n(Scaling: σ={config.augmentation.jitter_scale_ratio})', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, zoom_end)

# Strong augmentation (permutation + jitter)
axes[0, 2].plot(time_axis[zoom_mask], strong_aug[0, 0, zoom_mask], 'r-', linewidth=1.5)
axes[0, 2].set_title(f'Legacy Strong Augmentation\n(Perm: {config.augmentation.max_seg} seg + Jitter: σ={config.augmentation.jitter_ratio})', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xlim(0, zoom_end)

# Row 2: Frequency-based strategies
# Original (repeated for comparison)
axes[1, 0].plot(time_axis[zoom_mask], ecg_signal[zoom_mask], 'b-', linewidth=1.5)
axes[1, 0].set_title('Original Reference ECG Signal\n(Normalized)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Amplitude', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, zoom_end)

# Frequency masking (moderate)
axes[1, 1].plot(time_axis[zoom_mask], freq_aug[0, 0, zoom_mask], 'm-', linewidth=1.5)
axes[1, 1].set_title(f'Frequency Masking\n(Mask: {config.augmentation.freq_mask_ratio*100:.0f}% spectral)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(0, zoom_end)

# Combined spectral-temporal (strong)
axes[1, 2].plot(time_axis[zoom_mask], spectral_temporal_aug[0, 0, zoom_mask], 'c-', linewidth=1.5)
axes[1, 2].set_title(f'Spectral-Temporal Combined\n({config.augmentation.max_segment_freq} seg + {config.augmentation.freq_mask_ratio*100:.0f}% freq)', fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xlim(0, zoom_end)

# Row 3: TSTCC strategies
# Original (repeated for comparison)
axes[2, 0].plot(time_axis[zoom_mask], ecg_signal[zoom_mask], 'b-', linewidth=1.5)
axes[2, 0].set_title('Original Reference ECG Signal\n(Normalized)', fontsize=12, fontweight='bold')
axes[2, 0].set_ylabel('Amplitude', fontsize=10)
axes[2, 0].set_xlabel('Time (seconds)', fontsize=10)
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_xlim(0, zoom_end)

# TSTCC Weak
axes[2, 1].plot(time_axis[zoom_mask], tstcc_weak[0, 0, zoom_mask], 'orange', linewidth=1.5)
axes[2, 1].set_title(f'TSTCC Weak Augmentation\n(Mask: {mask_ratio_weak*100}% freq only)', fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel('Time (seconds)', fontsize=10)
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_xlim(0, zoom_end)

# TSTCC Strong
axes[2, 2].plot(time_axis[zoom_mask], tstcc_strong[0, 0, zoom_mask], 'darkred', linewidth=1.5)
axes[2, 2].set_title(f'TSTCC Strong Augmentation\n({config.augmentation.max_segment_freq} seg + {mask_ratio_strong*100} freq)', fontsize=12, fontweight='bold')
axes[2, 2].set_xlabel('Time (seconds)', fontsize=10)
axes[2, 2].grid(True, alpha=0.3)
axes[2, 2].set_xlim(0, zoom_end)

# Save the figure with parameter information
params_str = f"scale{config.augmentation.jitter_scale_ratio}_jitter{config.augmentation.jitter_ratio}_seg{config.augmentation.max_seg}"
filename = f"create_figures/illustration_augmentation_{params_str}.png"
plt.tight_layout()
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Figure saved as: {filename}")
plt.show()

# Print some statistics to understand the transformations
print("ECG Augmentation Analysis:")
print("=" * 70)
print(f"Original signal           - Mean: {np.mean(ecg_signal):.3f}, Std: {np.std(ecg_signal):.3f}")
print(f"Weak augmentation         - Mean: {np.mean(weak_aug):.3f}, Std: {np.std(weak_aug):.3f}")
print(f"Strong augmentation       - Mean: {np.mean(strong_aug):.3f}, Std: {np.std(strong_aug):.3f}")
print(f"Frequency masking         - Mean: {np.mean(freq_aug):.3f}, Std: {np.std(freq_aug):.3f}")
print(f"Spectral-Temporal combo   - Mean: {np.mean(spectral_temporal_aug):.3f}, Std: {np.std(spectral_temporal_aug):.3f}")
print(f"TSTCC Weak (10% freq)     - Mean: {np.mean(tstcc_weak):.3f}, Std: {np.std(tstcc_weak):.3f}")
print(f"TSTCC Strong (20seg+30%)  - Mean: {np.mean(tstcc_strong):.3f}, Std: {np.std(tstcc_strong):.3f}")

print("\nAugmentation Effects:")
print("-" * 60)
print("• Scaling (weak): Changes amplitude while preserving temporal structure")
print("• Permutation + Jitter (strong): Alters temporal order and adds noise")
print("• Frequency masking: Removes specific frequency components via FFT")
print("• Spectral-Temporal combo: Maximum temporal disruption + spectral perturbation")
print("• TSTCC Weak: Minimal spectral disruption for weak contrastive pairs")
print("• TSTCC Strong: Aggressive temporal + spectral for strong contrastive pairs")
print("• TSTCC Strong creates maximum transformation for robust representation learning")