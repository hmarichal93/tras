# TRAS Assets Directory

This directory contains images and media files for the README.

## Required Images

To complete the README, add the following images:

### 1. `tras-logo.png`
- **Description**: TRAS logo (tree ring themed)
- **Dimensions**: 200x200px recommended
- **Format**: PNG with transparent background
- **Content**: Stylized tree ring or dendrochronology symbol

### 2. `screenshot-main.png`
- **Description**: Main application interface screenshot
- **Dimensions**: 800-1200px width
- **Format**: PNG or JPG
- **Content**: TRAS GUI showing a wood cross-section with detected rings

### 3. `detection-methods.png`
- **Description**: Comparison of three detection methods
- **Dimensions**: ~400px width
- **Format**: PNG
- **Content**: Side-by-side or grid showing APD, CS-TRD, and DeepCS-TRD results

### 4. `preprocessing.png`
- **Description**: Before/after preprocessing example
- **Dimensions**: ~400px width
- **Format**: PNG
- **Content**: Original image vs cropped/processed version

### 5. `scale-calibration.png`
- **Description**: Scale calibration tool in action
- **Dimensions**: ~400px width
- **Format**: PNG
- **Content**: Image with calibration line drawn

### 6. `analysis-export.png`
- **Description**: Analysis and export features
- **Dimensions**: ~400px width
- **Format**: PNG
- **Content**: Ring properties dialog or export options

### 7. `workflow-diagram.png`
- **Description**: Visual workflow diagram
- **Dimensions**: ~700px width
- **Format**: PNG
- **Content**: Flowchart showing TRAS workflow steps

## Quick Image Generation Tips

### Using Screenshots
1. Launch TRAS: `tras`
2. Open a sample wood cross-section image
3. Use screenshot tool to capture different stages
4. Crop and resize as needed

### Using Gimp/Photoshop
- Create composite images showing before/after
- Add arrows and annotations for clarity
- Use consistent styling

### Using Online Tools
- **Excalidraw**: For workflow diagrams
- **Canva**: For logo design
- **Figma**: For professional layouts

## Alternative: Use Placeholders

If you don't have images yet, the README will show broken image icons. You can:

1. Remove the image tags temporarily
2. Use online placeholder services:
   ```markdown
   ![Logo](https://via.placeholder.com/200x200/8B5A2B/FFFFFF?text=TRAS)
   ```
3. Create simple SVG graphics inline

## SVG Logo Example

You can create a simple inline SVG logo:

```svg
<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="100" r="90" fill="none" stroke="#8B5A2B" stroke-width="3"/>
  <circle cx="100" cy="100" r="70" fill="none" stroke="#8B5A2B" stroke-width="3"/>
  <circle cx="100" cy="100" r="50" fill="none" stroke="#8B5A2B" stroke-width="3"/>
  <circle cx="100" cy="100" r="30" fill="none" stroke="#8B5A2B" stroke-width="3"/>
  <circle cx="100" cy="100" r="10" fill="#8B5A2B"/>
  <text x="100" y="185" text-anchor="middle" font-size="16" font-family="Arial" fill="#8B5A2B">TRAS</text>
</svg>
```

