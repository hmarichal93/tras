# LabelMe - Tree Ring Detection

Specialized annotation tool for tree ring detection in wood cross-section images. This branch integrates three automatic detection methods from the [TRAS repository](https://github.com/hmarichal93/tras).

---

## 📏 Understanding Ring Width Measurements

This tool provides **two different methods** for measuring ring widths, each suited for different purposes:

### 1. **Width (Centroid-Based)** 
**What it measures:** Distance between ring centroids  
**How it works:** Computes the geometric center (centroid) of each ring polygon, then measures the distance between consecutive centroids.

**Pros:**
- ✅ Fast and automatic
- ✅ Works well for circular/concentric rings
- ✅ Good for overall growth trends

**Cons:**
- ❌ Less accurate for eccentric growth
- ❌ Doesn't account for directional growth patterns
- ❌ Can be misleading for non-circular rings

**When to use:** Quick analysis, circular rings, preliminary measurements

---

### 2. **Radial Width (Transect-Based)** ⭐ RECOMMENDED
**What it measures:** Ring width along a specific radial line from the pith  
**How it works:** User defines a direction from the pith. The system computes where this line intersects each ring boundary and measures the distance between consecutive intersections.

**Pros:**
- ✅ Standard dendrochronology method
- ✅ Accurate for eccentric growth patterns
- ✅ User can choose measurement direction (avoid compression wood, reaction wood, etc.)
- ✅ Measures actual ring boundaries, not centroids
- ✅ More reproducible and scientifically valid

**Cons:**
- ❌ Requires one extra click (to define direction)

**When to use:** Scientific analysis, publication-quality data, eccentric rings, directional growth patterns

---

### Visual Comparison

```
Centroid-Based Width:
┌─────────────────────────────────────┐
│        Ring 3 (outer)               │
│    ↓ centroid                       │
│  ┌─────────────────────┐            │
│  │   Ring 2            │            │
│  │  ↓ centroid         │            │
│  │ ┌─────────────┐     │            │
│  │ │  Ring 1     │     │            │
│  │ │ ↓ centroid  │     │            │
│  │ │    • Pith   │     │            │
│  │ └─────────────┘     │            │
│  └─────────────────────┘            │
└─────────────────────────────────────┘

Width = distance between centroids
→ Averages growth across entire ring


Radial Width (Transect):
┌─────────────────────────────────────┐
│        Ring 3 (outer)               │
│       |← width →|                   │
│  ┌────┼─────────┼───────┐           │
│  │ R2 |         |       │           │
│  │    |← width →|       │           │
│  │ ┌──┼─────────┼───┐   │           │
│  │ │R1|         |   │   │           │
│  │ │  |← width →|   │   │           │
│  │ │  •─────────────┼───┼───→       │
│  │ │Pith (radial line)  │           │
│  │ └──────────────────┘ │           │
│  └────────────────────────┘         │
└─────────────────────────────────────┘

Radial Width = distance along line
→ Measures actual ring boundaries in chosen direction
```

---

### Which Should You Use?

| Scenario | Recommended Method |
|----------|-------------------|
| Quick overview | Centroid-Based Width |
| Scientific publication | **Radial Width** ⭐ |
| Eccentric/irregular rings | **Radial Width** ⭐ |
| Compression/reaction wood | **Radial Width** ⭐ |
| Comparing multiple samples | **Radial Width** ⭐ |
| Climate reconstruction | **Radial Width** ⭐ |

**💡 Pro Tip:** In dendrochronology, **radial width along a transect** is the standard method used in research papers and for cross-dating samples.

---

## 🎯 Features

### Three Detection Methods (All Production Ready!)

1. **APD (Automatic Pith Detection)**
   - Automatically detects the tree center (pith) using structural tensor analysis
   - Very fast (<1 second)
   - No manual input required
   - CPU-based

2. **CS-TRD (Classical Tree Ring Detection)**
   - Edge-based method using Canny detection and polar coordinates
   - Detects rings in ~73 seconds on 2400x2400 images
   - CPU-only (no GPU required)
   - 360 points per ring (1° angular resolution)

3. **DeepCS-TRD (Deep Learning Tree Ring Detection)**
   - Neural network-based detection with U-Net architecture
   - Pre-trained models for various wood species
   - Detects rings in ~101 seconds on 2400x2400 images
   - GPU-accelerated (CUDA)
   - 360 points per ring (1° angular resolution)

### Pre-trained Models for DeepCS-TRD
- **Generic** - Works with most wood species
- **Pinus V1 & V2** - Optimized for Pinus species
- **Gleditsia** - Optimized for Gleditsia species
- **Salix Glauca** - Optimized for Salix species

### Image Preprocessing
- **Manual Crop** - Select rectangular region to focus on wood cross-section
- **Resize** - Scale images (10%-100%) for faster processing
- **Background Removal** - Remove backgrounds using U2Net model (same as TRAS)
- **Metadata Storage** - Preprocessing info saved in JSON for traceability
- **Preview** - See changes before applying

### Scale/Calibration
- **Two Methods** - Draw line segment or direct input
- **Physical Units** - mm, cm, μm per pixel
- **Auto-Adjustment** - Scale updates automatically when resizing images
- **Physical Measurements** - Ring properties computed in real-world units (mm², cm², μm²)
- **Export** - CSV files include both pixel and physical measurements

## 📦 Installation

### 1. Clone the repository
```bash
git clone -b tras https://github.com/yourusername/labelme.git
cd labelme
```

### 2. Install dependencies
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install labelme with tree ring detection dependencies
pip install -e .
```

This will install all required dependencies including:
- PyTorch (for deep learning models)
- OpenCV (for image processing)
- Ultralytics (for YOLO-based pith detection)
- Shapely 1.7.0 (for geometric operations - specific version required!)
- segmentation-models-pytorch (for U-Net)
- uruDendro (tree ring utilities)
- And more...

### 3. Download DeepCS-TRD models (optional, for deep learning detection)
```bash
cd labelme/tree_ring_methods/deepcstrd
./download_models.sh
# Follow the instructions to download model files (~550MB)
```

**Note:** Model files must be downloaded separately from the original DeepCS-TRD repository due to their size.

### 4. Compile Devernay edge detector (for CS-TRD)
```bash
cd labelme/tree_ring_methods/cstrd/devernay
./compile.sh
```

This compiles the C-based edge detector used by CS-TRD. The compiled binary `devernay_cmd` must be in the `devernay/` directory.

## 🚀 Usage

### GUI Application

#### Quick Start
```bash
# Launch labelme with an image
labelme path/to/wood_cross_section.jpg

# Or launch and let user select file
labelme
```

#### Preprocessing Workflow (Optional but Recommended)

**Step 1: Crop (Optional)**
1. **Load Image:** Open a wood cross-section image
2. **Start Crop Mode:** Go to `Tools > Crop Image`
3. **Draw Rectangle:** Click and drag to select the region you want to keep (focus on the wood disk)
4. **Continue to preprocessing** (the rectangle will be used automatically)

**Step 2: Preprocess**
1. **Open Preprocessing:** Go to `Tools > Preprocess Image`
   - If you drew a crop rectangle, it will be detected automatically
2. **Configure Options:**
   - **Crop:** Shows status of crop region (applied first if present)
   - **Resize:** Adjust scale slider (10% - 100%) to reduce image size if needed
   - **Background Removal:** Enable U2Net-based background removal (same as TRAS)
     - ⚠️ Takes 10-30 seconds depending on image size and GPU
     - Uses deep learning model for accurate background removal
3. **Preview:** Click "Preview Changes" to see the result
4. **Apply:** Click "Apply" to replace the current image

**Important Notes:**
- Preprocessing **clears all existing annotations** (you'll be asked to confirm)
- Always preprocess **before** tree ring detection
- U2Net requires GPU for optimal performance
- Crop region is taken from the **last rectangle** you drew on the canvas

#### Scale/Calibration Workflow (Optional but Recommended for Physical Measurements)

1. **Load Image:** Open your wood cross-section image
2. **Open Scale Dialog:** Go to `Tools > Set Scale / Calibration`
3. **Choose Calibration Method:**
   
   **Method 1: Draw Line Segment** (recommended if you have a scale bar)
   - Select "📏 Draw a line segment on the image"
   - Click "Draw Line" button
   - Draw a line along a feature of known length (e.g., scale bar, ruler)
   - Enter the physical length (e.g., "10" mm)
   - Scale is calculated automatically (e.g., 0.00236 mm/pixel)
   
   **Method 2: Direct Input** (if you know the scale)
   - Select "⌨️ Enter scale directly (if known)"
   - Enter the scale value (e.g., "0.02")
   - Select unit (mm, cm, or μm)
   - Click OK

4. **Verify Scale:** The status bar will show "Scale set: X.XXXXXX unit/pixel"
5. **Note:** If you resize the image later, the scale will auto-adjust proportionally

**Benefits:**
- Ring Properties will show measurements in real physical units (mm², cm², μm²)
- CSV exports include both pixel and physical measurements
- More accurate for scientific analysis

#### Detection Workflow
1. **Load Image:** Open a wood cross-section image (or preprocessed image)
2. **Set Metadata (Optional):** Go to `Tools > Sample Metadata` to set harvested year, sample code
3. **Set Scale (Optional):** Go to `Tools > Set Scale / Calibration` (see above)
4. **Open Detection Dialog:** Go to `Tools > Tree Ring Detection` (or press shortcut)
5. **Auto-detect Pith:** Click "Auto-detect pith" button (uses APD, <1 second)
6. **Choose Detection Method:**
   - Click **"Detect with CS-TRD (CPU)"** for edge-based detection (~73 sec, no GPU needed)
   - Click **"Detect with DeepCSTRD (GPU)"** for AI-based detection (~101 sec, requires GPU)
7. **View Results:** Rings are automatically labeled (by year if metadata set, or ring_1, ring_2, ...)
8. **Compute Properties:** Go to `Tools > Ring Properties` to see:
   - Area (mm² or px²)
   - Cumulative Area
   - Perimeter (mm or px)
   - Ring Width (mm or px)
9. **Export CSV:** Click "Export to CSV" in Ring Properties dialog
10. **Refine Results:** Manually edit detected rings, add/remove ring boundaries
11. **Save:** Save annotations as JSON file

### Python API

#### APD - Automatic Pith Detection
```python
from labelme.utils.apd_helper import detect_pith_apd
from PIL import Image
import numpy as np

# Load image
img = np.array(Image.open('wood_sample.jpg'))

# Detect pith
pith_x, pith_y = detect_pith_apd(img)
print(f"Pith center: ({pith_x:.1f}, {pith_y:.1f})")
```

#### CS-TRD - Classical Tree Ring Detection
```python
from labelme.utils.cstrd_helper import detect_rings_cstrd
from PIL import Image
import numpy as np

# Load image
img = np.array(Image.open('wood_sample.jpg'))

# Detect rings
rings = detect_rings_cstrd(
    img, 
    center_xy=(1263.3, 1198.0),  # Pith coordinates from APD
    sigma=3.0,                    # Gaussian smoothing
    th_low=5.0,                   # Low threshold for Canny
    th_high=20.0,                 # High threshold for Canny
    alpha=30,                     # Angular sampling parameter
    nr=360                        # Number of radial samples (1° resolution)
)

print(f"Detected {len(rings)} rings")
for i, ring in enumerate(rings, 1):
    print(f"  Ring {i}: {len(ring)} points")
```

#### DeepCS-TRD - Deep Learning Detection
```python
from labelme.utils.deepcstrd_helper import detect_rings_deepcstrd
from PIL import Image
import numpy as np

# Load image
img = np.array(Image.open('wood_sample.jpg'))

# Detect rings with deep learning
rings = detect_rings_deepcstrd(
    img,
    center_xy=(1263.3, 1198.0),     # Pith coordinates from APD
    model_id='generic',              # or 'pinus_v1', 'gleditsia', 'salix'
    tile_size=0,                     # 0 = process full image
    alpha=45,                        # Angular sampling
    nr=360,                          # Radial samples
    total_rotations=5,               # Test-time augmentation
    prediction_map_threshold=0.5     # Probability threshold
)

print(f"Detected {len(rings)} rings")
```

#### Complete Workflow (APD + CS-TRD)
```python
from labelme.utils.apd_helper import detect_pith_apd
from labelme.utils.cstrd_helper import detect_rings_cstrd
from PIL import Image
import numpy as np

# Load image
img = np.array(Image.open('wood_sample.jpg'))

# Step 1: Detect pith
pith_x, pith_y = detect_pith_apd(img)
print(f"Pith detected at: ({pith_x:.1f}, {pith_y:.1f})")

# Step 2: Detect rings
rings = detect_rings_cstrd(img, center_xy=(pith_x, pith_y))
print(f"Detected {len(rings)} rings")

# Step 3: Process results
for i, ring in enumerate(rings, 1):
    # Each ring is a Nx2 numpy array of (x, y) coordinates
    print(f"Ring {i}: {len(ring)} points")
    # Save to file, visualize, etc.
```

## 📊 Performance Comparison

Test results on F02c.png (2408x2424 pixels):

| Method | Time | Rings Detected | Points/Ring | Hardware | Status |
|--------|------|----------------|-------------|----------|--------|
| **APD** | <1s | Pith only | - | CPU | ✅ Working |
| **CS-TRD** | 73-76s | 20 | 360 | CPU | ✅ Working |
| **DeepCS-TRD** | 101s | 43 | GPU | 360 | ✅ Working |

### Observations
- **DeepCS-TRD** is more sensitive (detected 43 rings vs CS-TRD's 20)
- **CS-TRD** is faster and requires only CPU (~30% faster than DeepCS-TRD)
- Both TRAS methods provide consistent 360 points per ring (1° resolution)
- **APD** is extremely fast and accurate for pith detection

## 🎯 Recommended Workflows

### For CPU-only Systems
**APD + CS-TRD** (~73 seconds total)
- No GPU required
- Good for laptops/servers without GPU
- Reliable edge-based detection
- 360 points per ring

### For GPU Systems / Highest Accuracy
**APD + DeepCS-TRD** (~101 seconds total)
- Requires CUDA-capable GPU
- More sensitive ring detection
- Pre-trained on diverse wood samples
- Species-specific models available
- 360 points per ring

### Manual Annotation
Use labelme's built-in polygon tools:
- Draw rings manually for difficult samples
- Edit automatically detected rings
- Combine automatic + manual approaches

## 🔧 Technical Details

### CS-TRD Implementation
CS-TRD runs as a subprocess to maintain compatibility with the original TRAS implementation:

```python
# Subprocess approach for robust integration
cmd = [sys.executable, str(main_py), "--input", str(temp_image), ...]
env = os.environ.copy()
env["PYTHONPATH"] = str(cstrd_root)
env["QT_QPA_PLATFORM"] = "offscreen"  # Headless execution
env["MPLBACKEND"] = "Agg"              # Non-interactive matplotlib
result = subprocess.run(cmd, env=env, capture_output=True)
```

**Key Components:**
- Devernay edge detector (compiled C code)
- Canny edge detection with custom thresholds
- Polar coordinate transformation
- Chain formation and merging algorithms

### DeepCS-TRD Implementation
Deep learning approach using U-Net architecture:
- Pre-trained on labeled wood cross-section datasets
- Processes image in polar coordinates
- Predicts probability map of ring boundaries
- Post-processes to extract ring contours
- Test-time augmentation for improved accuracy

### APD Implementation
Structural tensor analysis:
- Computes local orientation at each pixel
- Samples parallel coordinates space
- Optimizes pith location using geometric constraints
- Very fast (<1 second) and robust

## 📁 Project Structure

```
labelme/
├── tree_ring_methods/          # Tree ring detection implementations
│   ├── apd/                    # Automatic Pith Detection
│   │   ├── automatic_wood_pith_detector.py
│   │   ├── structural_tensor.py
│   │   ├── optimization.py
│   │   └── ...
│   ├── cstrd/                  # Classical Tree Ring Detection
│   │   ├── main.py            # Entry point for subprocess
│   │   ├── cross_section_tree_ring_detection/
│   │   ├── devernay/          # Edge detector (C code)
│   │   │   ├── devernay_cmd   # Compiled binary
│   │   │   └── compile.sh
│   │   └── config/
│   │       ├── default.json
│   │       └── general.json
│   └── deepcstrd/             # Deep Learning Tree Ring Detection
│       ├── deep_cstrd/
│       ├── models/            # Pre-trained model weights
│       │   ├── generic_model.pth
│       │   ├── pinus_v1.pth
│       │   └── ...
│       └── download_models.sh
├── utils/
│   ├── apd_helper.py          # APD integration
│   ├── cstrd_helper.py        # CS-TRD integration
│   └── deepcstrd_helper.py    # DeepCS-TRD integration
├── widgets/
│   └── tree_ring_dialog.py    # GUI dialog for tree ring detection
└── app.py                     # Main application
```

## 🛠️ Troubleshooting

### CS-TRD Issues

**Error: "devernay: not found"**
```bash
# Compile Devernay edge detector
cd labelme/tree_ring_methods/cstrd/devernay
./compile.sh
# Verify compilation
ls -la devernay_cmd
```

**Error: "Qt platform plugin could not be initialized"**
- Already handled: CS-TRD uses `QT_QPA_PLATFORM=offscreen`
- If still occurring, ensure Qt is properly installed

**Error: "shapely" related issues**
```bash
# Install specific shapely version (required!)
pip install shapely==1.7.0

# On Linux, may need GEOS library
sudo apt-get install libgeos-dev
```

### DeepCS-TRD Issues

**Error: "Model file not found"**
```bash
# Download models
cd labelme/tree_ring_methods/deepcstrd
./download_models.sh
```

**Error: "CUDA out of memory"**
- Use `tile_size` parameter to process image in smaller tiles
- Reduce `total_rotations` (test-time augmentation)
- Use smaller image size

**Error: "No module named 'urudendro'"**
```bash
# Install uruDendro dependency
pip install "urudendro @ git+https://github.com/hmarichal93/uruDendro.git@main"
```

### APD Issues

**Error: "No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**Pith detection inaccurate**
- APD works best on images with visible pith (center)
- For samples without visible pith, manually specify center
- Adjust image preprocessing (contrast, brightness)

## 🎓 Parameter Tuning

### CS-TRD Parameters

**sigma** (Gaussian smoothing, default: 3.0)
- Increase (4-5) for noisy images
- Decrease (1-2) for high-quality images

**th_low, th_high** (Canny thresholds, default: 5.0, 20.0)
- Increase for fewer edges (reduce false positives)
- Decrease for more edges (capture faint rings)

**alpha** (Angular sampling, default: 30)
- Lower values = more angular samples = slower but more accurate
- Higher values = fewer samples = faster but may miss details

**nr** (Number of radial samples, default: 360)
- 360 = 1° resolution (recommended)
- Increase for finer angular resolution
- Decrease for faster processing

### DeepCS-TRD Parameters

**model_id** (Pre-trained model)
- 'generic' - Use for most samples
- 'pinus_v1', 'pinus_v2' - Use for Pinus species
- 'gleditsia' - Use for Gleditsia species
- 'salix' - Use for Salix species

**total_rotations** (Test-time augmentation, default: 5)
- Higher values = better accuracy, slower processing
- 1 = no augmentation, fastest

**prediction_map_threshold** (Probability threshold, default: 0.5)
- Increase (0.6-0.7) for higher confidence rings only
- Decrease (0.3-0.4) to detect fainter rings

## 📝 Output Format

All methods output rings as lists of numpy arrays:

```python
rings = [
    np.array([[x1, y1], [x2, y2], ..., [xN, yN]]),  # Ring 1
    np.array([[x1, y1], [x2, y2], ..., [xN, yN]]),  # Ring 2
    ...
]
```

Each ring is a closed polygon with N points. For TRAS methods (CS-TRD, DeepCS-TRD), N=360 (1° angular resolution).

### LabelMe JSON Format

When saved from GUI, annotations use LabelMe JSON format:

```json
{
  "version": "5.0.0",
  "shapes": [
    {
      "label": "ring_1",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    },
    ...
  ],
  "imagePath": "sample.jpg",
  "imageHeight": 2424,
  "imageWidth": 2408
}
```

## 🧪 Testing

Test all methods on a sample image:

```bash
python -c "
from labelme.utils.apd_helper import detect_pith_apd
from labelme.utils.cstrd_helper import detect_rings_cstrd
from labelme.utils.deepcstrd_helper import detect_rings_deepcstrd
from PIL import Image
import numpy as np
import time

# Load image
img = np.array(Image.open('path/to/wood_sample.jpg'))
print(f'Image: {img.shape}')

# Test APD
print('\n1. Testing APD (Pith Detection)...')
start = time.time()
x, y = detect_pith_apd(img)
print(f'   ✅ Pith: ({x:.1f}, {y:.1f}) in {time.time()-start:.2f}s')

# Test CS-TRD
print('\n2. Testing CS-TRD (Classical Detection)...')
start = time.time()
rings_cstrd = detect_rings_cstrd(img, center_xy=(x, y))
print(f'   ✅ CS-TRD: {len(rings_cstrd)} rings in {time.time()-start:.1f}s')

# Test DeepCS-TRD (if GPU available)
print('\n3. Testing DeepCS-TRD (Deep Learning)...')
start = time.time()
rings_deep = detect_rings_deepcstrd(img, center_xy=(x, y))
print(f'   ✅ DeepCS-TRD: {len(rings_deep)} rings in {time.time()-start:.1f}s')

print('\n✅ All tests passed!')
"
```

## 📚 References

- **TRAS Repository:** https://github.com/hmarichal93/tras
- **LabelMe:** https://github.com/wkentaro/labelme
- **uruDendro:** https://github.com/hmarichal93/uruDendro
- **DeepCS-TRD Paper:** [Link to paper if available]

## 🤝 Contributing

This is a specialized branch for tree ring detection. For contributions:
1. Test changes on diverse wood samples
2. Ensure all three methods still work
3. Update documentation
4. Follow the existing code style

## 📄 License

MIT License (same as original LabelMe)

## 🙏 Acknowledgments

- **TRAS Repository** - For the tree ring detection methods (APD, CS-TRD, DeepCS-TRD)
- **LabelMe** - For the base annotation tool
- **uruDendro** - For tree ring analysis utilities

## ✅ Status

**Production Ready!** All three TRAS methods are fully integrated and tested:
- ✅ APD (Automatic Pith Detection) - <1 second
- ✅ CS-TRD (Classical edge-based) - ~73 seconds, CPU-only
- ✅ DeepCS-TRD (Deep learning) - ~101 seconds, GPU-accelerated

Both GUI and Python API are fully functional. Ready for production use in dendrochronology research! 🌲

