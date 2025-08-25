# CSF Algorithm Animation with Manim and PDAL

## Overview

This project creates an educational animation explaining the Cloth Simulation Filter (CSF) algorithm used for ground point classification in LiDAR point clouds. The animation integrates with PDAL to demonstrate real-world usage and pipeline configuration.

## Features

- **Step-by-step algorithm explanation** with visual demonstrations
- **Physics simulation visualization** showing cloth behavior
- **PDAL integration** for real-world pipeline examples
- **Parameter tuning guidance** with visual effects
- **Synthetic and real data examples**

## Algorithm Steps Animated

1. **Problem Introduction** - Distinguishing ground vs non-ground points
2. **Cloth Initialization** - Virtual cloth setup above point cloud
3. **Physics Simulation** - Gravity and collision detection
4. **Ground Classification** - Distance-based point labeling
5. **Parameter Effects** - Tuning for different environments

## Installation

```bash
# Core requirements
pip install manim
pip install numpy

# Optional but recommended for real data
pip install pdal

# For development
pip install matplotlib  # Additional visualizations
```

## Usage

### Basic Animation
```bash
# Low quality for testing
manim -pql csf_animation.py CSFAlgorithmAnimation

# High quality for presentation
manim -pqh csf_animation.py CSFAlgorithmAnimation

# Parameter exploration scene
manim -pql csf_animation.py CSFParameterExploration
```

### Custom Configuration
```python
# Modify parameters in the scene
class CSFAlgorithmAnimation(Scene):
    def __init__(self):
        self.cloth_resolution = 0.5  # Cloth grid resolution
        self.gravity = 0.1          # Physics simulation speed
        self.threshold = 0.3        # Classification threshold
        super().__init__()
```

## PDAL Integration

The animation demonstrates real PDAL pipeline usage:

```json
{
  "pipeline": [
    {
      "type": "readers.las",
      "filename": "input.las"
    },
    {
      "type": "filters.csf",
      "resolution": 0.5,
      "threshold": 0.25,
      "rigidness": 2,
      "iterations": 500
    },
    {
      "type": "writers.las",
      "filename": "output_ground.las",
      "where": "Classification == 2"
    }
  ]
}
```

## Key CSF Parameters Explained

| Parameter | Range | Description | Animation Shows |
|-----------|-------|-------------|-----------------|
| **resolution** | 0.1-2.0m | Cloth grid cell size | Grid density effect |
| **threshold** | 0.1-1.0m | Distance threshold for ground classification | Classification sensitivity |
| **rigidness** | 1-3 | Cloth stiffness | Deformation behavior |
| **iterations** | 100-1000 | Simulation steps | Convergence quality |

## File Structure

```
csf_animation/
├── csf_animation.py          # Main animation script
├── README.md                 # This file
├── data/                     # Sample point cloud data
│   ├── sample.las           # Example LiDAR file
│   └── synthetic_data.py    # Generate test data
├── media/                   # Output animations
│   ├── videos/             # Rendered animations
│   └── images/             # Frame exports
└── utils/                  # Helper functions
    ├── pdal_utils.py       # PDAL pipeline helpers
    └── visualization.py    # Additional plotting
```

## Educational Content

### CSF Algorithm Theory

The Cloth Simulation Filter algorithm works by:

1. **Initialization**: Place a virtual cloth above the point cloud
2. **Physics**: Simulate cloth falling under gravity
3. **Collision**: Detect collisions with point cloud features
4. **Settling**: Allow cloth to settle on surfaces
5. **Classification**: Points close to cloth = ground, distant = non-ground

### Visual Learning Objectives

- Understand physics-based filtering approach
- See parameter effects on classification results
- Learn PDAL pipeline configuration
- Recognize appropriate use cases and limitations

## Advanced Features

### Real Data Integration
```python
def load_real_data(self, filename):
    """Load actual LiDAR data for animation"""
    if PDAL_AVAILABLE:
        pipeline = pdal.Pipeline([
            {"type": "readers.las", "filename": filename},
            {"type": "filters.sample", "radius": 0.1}  # Subsample for animation
        ])
        pipeline.execute()
        return pipeline.arrays[0]
```

### Interactive Parameter Tuning
```python
class InteractiveCSF(Scene):
    def construct(self):
        # Allow real-time parameter adjustment during animation
        self.interactive_sliders()
        self.live_classification_update()
```

### Performance Optimization
- Subsampling for large point clouds
- LOD (Level of Detail) for distant points
- Efficient collision detection algorithms

## Output Examples

The animation produces several visualization types:

1. **Algorithm Overview** - Conceptual workflow
2. **Step-by-Step Demonstration** - Detailed process
3. **Parameter Comparison** - Side-by-side effects
4. **Real Data Example** - Practical application

## Troubleshooting

### Common Issues

1. **PDAL not found**: Install with `conda install -c conda-forge pdal`
2. **Memory issues**: Reduce point cloud size in `generate_synthetic_lidar()`
3. **Slow rendering**: Use `-ql` flag for quick previews

### Performance Tips

- Use synthetic data for initial development
- Reduce cloth resolution for faster simulation
- Limit animation iterations for testing

## Contributing

To extend the animation:

1. Add new algorithm variants (PMF, SMRF)
2. Include more real-world examples
3. Add interactive parameter exploration
4. Improve visual effects and transitions

## References

- Zhang, W., et al. (2016). "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation"
- PDAL Documentation: https://pdal.io/
- Manim Documentation: https://docs.manim.community/

## License

Educational use - appropriate for academic and training purposes.
