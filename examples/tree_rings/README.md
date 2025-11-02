# TRAS Tutorial: Tree Ring Analysis Workflow

This tutorial demonstrates the complete workflow for analyzing tree rings using TRAS (Tree Ring Analyzer Suite).

## 📁 Sample Data

- **sample.jpg**: Wood cross-section image (Pinus taeda sample)

## 🚀 Quick Start

### Launch TRAS

```bash
tras
# Or open directly with sample image
tras examples/tree_rings/sample.jpg
```

## 📖 Step-by-Step Workflow

### Step 1: Load Image

1. **File > Open** or drag and drop `sample.jpg`
2. The image appears in the main canvas
3. You can zoom (mouse wheel) and pan (middle-click drag)

### Step 2: Set Scale (Optional but Recommended)

1. **Tools > Set Image Scale**
2. Two options:
   - **Draw line**: Click two points on a known-length reference (e.g., ruler)
   - **Direct input**: Enter known scale factor
3. Enter physical length (e.g., "10 cm" or "1000 μm")
4. Scale is stored in metadata for physical unit measurements

**Why set scale?**
- Ring properties will show real dimensions (cm², mm)
- Export measurements in physical units
- Required for accurate radial width in cm

### Step 3: Add Sample Metadata (Optional)

1. **Tools > Sample Metadata**
2. Fill in:
   - **Harvested Year**: When the tree was cut (e.g., 2024)
   - **Sample Code**: Your sample identifier (e.g., "P001")
   - **Observations**: Notes about the sample
3. Click **OK** to save

**Why add metadata?**
- Automatically labels rings with years
- Organizes your data
- Included in exports

### Step 4: Preprocess Image (Optional)

1. **Tools > Preprocess Image**
2. Available operations:
   - **Crop**: Remove unwanted regions (⚠️ keep >100px margin!)
   - **Resize**: Scale down large images (10-100%)
   - **Background Removal**: Use U2Net to remove background (10-30s)
3. Click **Preview** to see changes
4. Click **Apply** when satisfied

**When to preprocess?**
- Large images (>3000px): Resize to 50-70% for faster detection
- Noisy background: Remove with U2Net
- Multiple samples in one image: Crop to focus on one

**⚠️ Important**: Leave adequate margin (>100px) around the wood section. Detection methods may fail on tightly cropped images.

### Step 5: Detect Tree Rings

1. **Tools > Tree Ring Detection**
2. The dialog opens with two steps:

#### Step 5a: Set Pith Location

Choose one method:
- **Click on image**: Click button, then click pith center on image
- **Auto-detect (APD)**: Automatic pith detection (~1 second)
  - Expand "APD Advanced Parameters" to customize
  - Choose method: `apd` (default) or `apd_pcl` (for unclear rings)

#### Step 5b: Detect Tree Rings

Choose one method:

**CS-TRD (Classical Method)**
- Edge-based detection
- Fast (~73s on CPU)
- Good for clear, regular rings
- **Linux/macOS only** (disabled on Windows)
- Expand "CS-TRD Advanced Parameters" to customize:
  - Gaussian Sigma (smoothing)
  - Thresholds (edge sensitivity)
  - Alpha (angular sampling)
  - Radial Samples (resolution)

**DeepCS-TRD (Deep Learning)**
- U-Net based detection
- Slower (~101s with GPU)
- Better for irregular/challenging rings
- **Works on all platforms**
- Expand "DeepCS-TRD Model & Parameters":
  - **Model**: Choose species-specific or generic
    - `generic` - Works for most species (recommended)
    - `pinus_v1/v2` - Optimized for Pinus taeda
    - `gleditsia` - For Gleditsia triacanthos
    - `salix` - For Salix humboldtiana
    - **📁 Upload** - Add your own trained model
  - **Processing Mode**: Full image (faster) vs Tiled (less memory)
  - **Rotations (TTA)**: More = accurate but slower
  - **Prediction Threshold**: Adjust sensitivity

3. Click **Detect with CS-TRD** or **Detect with DeepCS-TRD**
4. Wait for processing (watch cursor change to ⏳)
5. Rings appear as green polygons on the image
6. Click **OK** to close dialog

**Which method to use?**
- **Try APD first** for pith detection (very fast)
- **Try CS-TRD first** if on Linux/macOS (faster than DeepCS-TRD)
- **Use DeepCS-TRD** for challenging samples or if CS-TRD fails
- **Experiment with parameters** if results are poor

### Step 6: Manual Editing (Optional)

After detection, you can manually adjust rings:

1. **Select Edit Mode**:
   - Click **Edit Polygons** button or press `Ctrl+J`
2. **Edit Points**:
   - Click and drag points to adjust ring boundaries
   - Right-click on a point to delete it
   - Click on edge between points to add new point
3. **Add Missing Rings**:
   - Click **Create Polygons** button
   - Draw new ring by clicking points around it
   - Right-click or press `Enter` to close polygon
   - Type ring label in dialog (e.g., "ring_15")
4. **Delete Rings**:
   - Select ring from label list
   - Press `Delete` key or right-click > Delete

**Tips for manual editing:**
- Zoom in (mouse wheel) for precise adjustments
- Use arrow keys to fine-tune point positions
- Auto-relabeling happens when you view Ring Properties

### Step 7: Measure Radial Width (Optional)

1. **Tools > Measure Ring Width**
2. Dialog opens showing current pith location
3. Choose action:
   - **Set Direction**: Click to define radial line from pith
   - **Clear Previous**: Remove existing measurement line
   - **Export to .POS**: Save measurements (after setting direction)
   - **Close**: Exit without changes

**Setting direction:**
1. Click **Set Direction**
2. Dialog closes
3. Click once on image to define radial line direction
4. A red line appears from pith through your click point
5. Ring widths are measured along this line

**Exporting to .POS format:**
1. After setting direction, reopen dialog
2. Click **Export to .POS**
3. Save file (compatible with CooRecorder software)
4. File contains: metadata, scale, ring measurements

**When to use radial width?**
- Required for dendrochronology dating
- Standard format for dendro software
- More accurate than area-based measurements

### Step 8: View Ring Properties

1. **Tools > Ring Properties**
2. Auto-relabeling occurs (rings numbered/dated)
3. Table displays for each ring:
   - **Label**: Ring identifier (or year if metadata provided)
   - **Area**: Ring area (px² or cm² if scale set)
   - **Perimeter**: Ring boundary length (px or cm)
   - **Radial Width**: If measured (px or cm)
4. Sort by clicking column headers
5. Click **Export to CSV** to save

**Understanding the data:**
- **Without scale**: All values in pixels
- **With scale**: Physical units (cm², cm)
- **With metadata + harvested year**: Labels show actual years
- **Radial width**: Only shown if measured in Step 7

### Step 9: Save Your Work

1. **File > Save** or `Ctrl+S`
2. Saves as `.json` file containing:
   - All ring polygons (coordinates)
   - Ring labels
   - Metadata (if added)
   - Preprocessing info
   - Scale information
   - Pith location
3. Reopen later with **File > Open**

**What's saved?**
- Ring boundaries (as polygon points)
- Ring labels/years
- All metadata
- BUT NOT the measurements themselves - regenerate via Ring Properties

### Step 10: Export Results

Choose export format based on your needs:

**Option A: CSV Export (from Ring Properties)**
- Contains: Label, Area, Perimeter, Radial Width
- Good for: Spreadsheet analysis, statistics
- Units: Pixels or physical (if scale set)

**Option B: .POS Export (from Measure Ring Width)**
- Contains: Ring borders as coordinate pairs
- Good for: CooRecorder, dendrochronology software
- Standard format in tree ring science

**Option C: JSON (from Save)**
- Contains: Complete annotation with all info
- Good for: Reopening in TRAS, archiving, sharing

## 🎯 Common Workflows

### Workflow 1: Quick Visual Analysis
```
Load Image → Detect Rings → View Properties
```
Skip metadata, scale, and radial measurements for fast exploration.

### Workflow 2: Full Dendrochronology Analysis
```
Load Image → Metadata → Scale → Preprocess → Detect Rings 
→ Measure Radial Width → Export .POS
```
Complete workflow for dating, climate reconstruction, etc.

### Workflow 3: Ring Counting
```
Load Image → Detect Rings → Manual Edit → View Properties → Save
```
Focus on accurate ring identification and counting.

### Workflow 4: Batch Processing (Multiple Samples)
```
For each image:
  Load → Preprocess → Detect → Save JSON
Later: Load JSON → Measure → Export CSV
```
Separate detection from measurement for efficiency.

## 🔧 Troubleshooting

### "No rings detected"
- **Try different method**: CS-TRD vs DeepCS-TRD
- **Adjust parameters**: Expand advanced settings
- **Check image quality**: Rescan at higher resolution
- **Preprocess**: Try background removal or contrast adjustment

### "Detection missed some rings"
- **Manual editing**: Add missing rings manually
- **Try DeepCS-TRD**: Better for challenging samples
- **Lower threshold**: In DeepCS-TRD parameters (e.g., 0.3)

### "Too many false positives"
- **Increase threshold**: In DeepCS-TRD parameters (e.g., 0.6)
- **Manual editing**: Delete false rings
- **Try CS-TRD**: Sometimes more conservative

### "Image is rotated/inclined after preprocessing"
- This has been fixed in recent versions
- Make sure you have the latest TRAS version
- Preprocessing uses PIL for robust image handling

### "CS-TRD failed / AssertionError"
- **Leave more margin**: Crop with >100px around wood section
- **Don't crop too tight**: Use resize instead
- **Try DeepCS-TRD**: More robust to tight crops

## 📚 Tips & Best Practices

1. **Image Quality Matters**
   - Use high-resolution scans (300+ DPI)
   - Ensure good lighting and contrast
   - Clean samples before scanning

2. **Set Scale Early**
   - Do it right after loading image
   - Use a calibrated ruler in the scan
   - Double-check scale accuracy

3. **Name Samples Consistently**
   - Use systematic sample codes
   - Include location, species, sample number
   - Example: "Site_A_Pine_001"

4. **Save Frequently**
   - Save JSON after detection
   - Save again after manual edits
   - Keep backups of important data

5. **Experiment with Methods**
   - Try both CS-TRD and DeepCS-TRD
   - Test different models for DeepCS-TRD
   - Adjust parameters based on results

6. **Manual QC Always**
   - Always visually inspect detected rings
   - Fix obvious errors manually
   - Document any issues in metadata

## 🎓 Advanced Topics

### Custom DeepCS-TRD Models

If you have trained your own model:

1. Open **Tools > Tree Ring Detection**
2. Expand "DeepCS-TRD Model & Parameters"
3. Click **📁 Upload** button
4. Select your `.pth` model file
5. Enter model name (e.g., "eucalyptus")
6. Choose tile size (0 or 256)
7. Model appears in dropdown as "eucalyptus (custom)"

Your model is saved to:
```
tras/tree_ring_methods/deepcstrd/models/deep_cstrd/0_eucalyptus_1504.pth
```

### Batch Processing with Python

For processing multiple images programmatically, use the TRAS Python API:

```python
from tras.utils.cstrd_helper import detect_rings_cstrd
from tras.utils.deepcstrd_helper import detect_rings_deepcstrd
from tras.utils.apd_helper import detect_pith_apd
import cv2

# Load image
image = cv2.imread("sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect pith
cx, cy = detect_pith_apd(image)

# Detect rings with DeepCS-TRD
rings = detect_rings_deepcstrd(
    image, 
    center_xy=(cx, cy),
    model_id="generic",
    tile_size=0
)

print(f"Detected {len(rings)} rings")
```

## 📞 Support

- **Issues**: https://github.com/hmarichal93/tras/issues
- **Email**: hmarichal93@gmail.com
- **Documentation**: https://hmarichal93.github.io/tras/

## 🌲 Happy Tree Ring Analyzing!

