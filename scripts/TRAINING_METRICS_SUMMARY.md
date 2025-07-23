# BCR-Net Training and Evaluation Metrics Summary

## Training Configuration

### Loss Function
**We are training on MSE Loss (Mean Squared Error)**
- `criterion = nn.MSELoss()`
- This is the standard L2 loss: `MSE = mean((prediction - target)²)`
- PyTorch MSE loss automatically handles batching and averaging

### Why MSE Loss?
- MSE penalizes large errors more heavily (quadratic penalty)
- Provides smooth gradients for training
- Standard choice for regression problems like image reconstruction
- Matches the original Keras implementation which used `'mean_squared_error'`

## Evaluation Metrics

### 1. Mean Squared Error (MSE)
- **Formula**: `MSE = mean((pred - true)²)`
- **Units**: Same as data (squared)
- **Range**: [0, ∞), lower is better
- **Current Value**: ~0.130

### 2. Mean Absolute Error (MAE)  
- **Formula**: `MAE = mean(|pred - true|)`
- **Units**: Same as data
- **Range**: [0, ∞), lower is better
- **Current Value**: ~0.162

### 3. Peak Signal-to-Noise Ratio (PSNR)
- **Formula**: `PSNR = -10 * log10(MSE / pixel_max²)`
- **Units**: Decibels (dB)
- **Range**: (-∞, ∞), higher is better
- **Current Value**: ~8.86 dB
- **Note**: Training uses negative PSNR as error metric (lower negative = better)

### 4. Relative L2 Error (NEW)
- **Formula**: `||pred - true||₂ / ||true||₂`
- **Description**: L2 norm of error relative to L2 norm of ground truth
- **Units**: Dimensionless ratio
- **Range**: [0, ∞), lower is better
- **Current Value**: ~0.369 (36.95%)

### 5. Relative L1 Error (NEW)
- **Formula**: `||pred - true||₁ / ||true||₁`
- **Description**: L1 norm of error relative to L1 norm of ground truth  
- **Units**: Dimensionless ratio
- **Range**: [0, ∞), lower is better
- **Current Value**: ~0.168 (16.79%)

### 6. Relative Error (Percentage)
- **Formula**: `MAE / mean(|true|) * 100`
- **Description**: Average absolute error as percentage of signal magnitude
- **Units**: Percentage
- **Range**: [0, ∞), lower is better
- **Current Value**: ~16.79%

## Metric Interpretation

### What These Numbers Mean:
- **MSE = 0.130**: Average squared pixel error (in normalized [-1,1] range)
- **MAE = 0.162**: Average absolute pixel error 
- **PSNR = 8.86 dB**: Signal quality measure (8-10 dB is reasonable for this problem)
- **Rel L2 = 36.95%**: L2 error is ~37% of signal L2 norm
- **Rel L1 = 16.79%**: L1 error is ~17% of signal L1 norm

### Performance Context:
- All metrics are calculated on **unpadded data** (100×100 original size)
- Data is normalized to [-1, 1] range using min-max normalization
- Model trained for only 1 epoch (minimal training for demonstration)
- With more training epochs, these metrics should improve significantly

## Key Technical Details

### Data Preprocessing:
- Original images: 100×100 → Zero-padded to 128×128 for processing
- Min-max normalization: [original_range] → [-1, 1]
- Frequency domain processing for input scattering data

### Training Setup:
- Optimizer: AdamW with learning rate 0.001
- Gradient clipping: threshold = 1.0
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Batch size: 16 (for demo), typically 32
- Device: CUDA (GPU acceleration)

### Output Processing:
- Predictions are unpadded from 128×128 back to 100×100
- Both padded and unpadded results are saved for comparison
- All final metrics calculated on unpadded data for fair evaluation
