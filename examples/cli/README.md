# TRAS CLI Examples

This directory contains example configurations and usage instructions for TRAS command-line tools.

## Unified Command

TRAS provides a unified `tras_detect` command that automatically detects whether the input is a single image file or a directory:

- **Single image file**: Runs detection and saves JSON output
- **Directory**: Batch processes all JPEG/PNG images, generates JSON/CSV/PDF per image

## Quick Start

### Single Image Mode

```bash
# Basic detection
tras_detect image.jpg -o output.json

# With custom parameters
tras_detect image.jpg --ring-method deepcstrd --pith-method apd_dl
```

### Batch Processing Mode

#### 1. Basic Usage (CLI flags only)

Process all JPEG/PNG images in a folder with minimal configuration:

```bash
tras_detect /path/to/images \
  --scale-value 0.0213 \
  --scale-unit mm \
  --ring-method deepcstrd
```

#### 2. Using Configuration File (Recommended)

Create a configuration file based on `process_config.yml`:

```bash
# Copy example config
cp examples/cli/process_config.yml my_config.yml

# Edit my_config.yml to match your needs

# Run batch processing
tras_detect /path/to/images --config my_config.yml
```

#### 3. Override Config with CLI Flags

You can override specific settings from the config file using CLI flags:

```bash
tras_detect /path/to/images \
  --config my_config.yml \
  --scale-value 0.025 \
  --ring-method cstrd
```


## Configuration File Structure

The YAML configuration file supports:

- **Physical Scale** (required): `physical_scale.value` and `physical_scale.unit`
- **Preprocessing**: `preprocess.resize_scale`, `preprocess.remove_background`
- **Postprocessing**: `postprocess.sampling_nr`
- **Pith Detection**: `pith.auto`, `pith.method`
- **Ring Detection**: `rings.method` and method-specific parameters
  - **CS-TRD**: `rings.cstrd.*` parameters
  - **DeepCS-TRD**: `rings.deepcstrd.*` parameters (including `model` as ID or path)
  - **INBD**: `rings.inbd.*` parameters (including `model` as ID or path)

## Output Files

For each input image (e.g., `sample_001.jpg`), `tras_detect` generates:

- `sample_001.json` - Detection results (rings + pith coordinates)
- `sample_001.csv` - Ring measurements (area, perimeter, cumulative area)
- `sample_001.pdf` - Analysis report with plots

## Model Paths

Both DeepCS-TRD and INBD support model specification as:

1. **Model ID** (e.g., `generic`, `pinus_v2`, `INBD_EH`) - resolved from standard locations
2. **Filesystem path** (e.g., `/path/to/model.pth`) - used directly if file exists

Example:
```yaml
rings:
  deepcstrd:
    model: /custom/path/to/my_model.pth  # Custom model path
    # OR
    model: generic  # Standard model ID
```

## Examples

See `process_config.yml` for a complete configuration template with all available options.

