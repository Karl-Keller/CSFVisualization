"""
PDAL Utilities for CSF Algorithm Animation
==========================================

Helper functions for integrating PDAL processing with Manim animations.
Provides data loading, processing, and CSF parameter management.
"""

import json
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    import pdal
    PDAL_AVAILABLE = True
except ImportError:
    PDAL_AVAILABLE = False
    print("Warning: PDAL not available. Some features will be limited.")


class CSFPipelineManager:
    """Manages PDAL pipelines for CSF algorithm demonstration"""
    
    def __init__(self, default_params: Optional[Dict] = None):
        """
        Initialize CSF pipeline manager
        
        Args:
            default_params: Default CSF parameters
        """
        self.default_params = default_params or {
            "resolution": 0.5,
            "threshold": 0.25,
            "rigidness": 2,
            "iterations": 500,
            "time_step": 0.65
        }
        
    def create_csf_pipeline(self, 
                           input_file: str, 
                           output_file: str,
                           csf_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create PDAL pipeline for CSF filtering
        
        Args:
            input_file: Path to input LAS file
            output_file: Path to output LAS file
            csf_params: CSF parameters override
            
        Returns:
            PDAL pipeline dictionary
        """
        params = {**self.default_params, **(csf_params or {})}
        
        pipeline = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": input_file
                },
                {
                    "type": "filters.csf",
                    **params
                },
                {
                    "type": "writers.las",
                    "filename": output_file
                }
            ]
        }
        
        return pipeline
        
    def create_comparative_pipeline(self, 
                                  input_file: str,
                                  param_sets: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create multiple pipelines for parameter comparison
        
        Args:
            input_file: Input LAS file
            param_sets: List of parameter dictionaries
            
        Returns:
            List of pipeline dictionaries
        """
        pipelines = []
        
        for i, params in enumerate(param_sets):
            output_file = f"output_csf_{i}.las"
            pipeline = self.create_csf_pipeline(input_file, output_file, params)
            pipelines.append(pipeline)
            
        return pipelines
        
    def execute_pipeline(self, pipeline_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Execute PDAL pipeline and return point array
        
        Args:
            pipeline_dict: PDAL pipeline configuration
            
        Returns:
            Numpy array of processed points or None if PDAL unavailable
        """
        if not PDAL_AVAILABLE:
            print("PDAL not available - returning synthetic data")
            return self.generate_synthetic_result()
            
        try:
            pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
            pipeline.execute()
            return pipeline.arrays[0]
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            return None
            
    def generate_synthetic_result(self) -> np.ndarray:
        """Generate synthetic point cloud for demonstration"""
        np.random.seed(42)
        
        # Create synthetic point cloud with ground and non-ground points
        n_points = 1000
        
        # Ground points
        ground_x = np.random.uniform(-10, 10, n_points // 2)
        ground_y = np.random.uniform(-10, 10, n_points // 2)
        ground_z = np.random.normal(0, 0.1, n_points // 2)
        ground_classification = np.full(n_points // 2, 2)  # Ground class
        
        # Non-ground points (vegetation, buildings)
        nonground_x = np.random.uniform(-10, 10, n_points // 2)
        nonground_y = np.random.uniform(-10, 10, n_points // 2)
        nonground_z = np.random.uniform(0.5, 5.0, n_points // 2)
        nonground_classification = np.full(n_points // 2, 1)  # Unclassified
        
        # Combine arrays
        x = np.concatenate([ground_x, nonground_x])
        y = np.concatenate([ground_y, nonground_y])
        z = np.concatenate([ground_z, nonground_z])
        classification = np.concatenate([ground_classification, nonground_classification])
        
        # Create structured array similar to PDAL output
        dtype = [('X', 'f8'), ('Y', 'f8'), ('Z', 'f8'), ('Classification', 'u1')]
        points = np.zeros(n_points, dtype=dtype)
        points['X'] = x
        points['Y'] = y
        points['Z'] = z
        points['Classification'] = classification
        
        return points


class AnimationDataProcessor:
    """Process point cloud data for Manim animation"""
    
    @staticmethod
    def subsample_points(points: np.ndarray, 
                        max_points: int = 500,
                        method: str = 'random') -> np.ndarray:
        """
        Subsample points for animation performance
        
        Args:
            points: Input point array
            max_points: Maximum number of points to keep
            method: Subsampling method ('random', 'grid', 'fps')
            
        Returns:
            Subsampled point array
        """
        if len(points) <= max_points:
            return points
            
        if method == 'random':
            indices = np.random.choice(len(points), max_points, replace=False)
            return points[indices]
        elif method == 'grid':
            return AnimationDataProcessor._grid_subsample(points, max_points)
        elif method == 'fps':
            return AnimationDataProcessor._farthest_point_sample(points, max_points)
        else:
            raise ValueError(f"Unknown subsampling method: {method}")
    
    @staticmethod
    def _grid_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
        """Grid-based subsampling"""
        x, y, z = points['X'], points['Y'], points['Z']
        
        # Calculate grid size
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        
        grid_size = np.sqrt((x_range * y_range) / max_points)
        
        # Create grid indices
        x_grid = ((x - x.min()) / grid_size).astype(int)
        y_grid = ((y - y.min()) / grid_size).astype(int)
        
        # Select one point per grid cell
        grid_dict = {}
        for i, (gx, gy) in enumerate(zip(x_grid, y_grid)):
            key = (gx, gy)
            if key not in grid_dict:
                grid_dict[key] = i
                
        indices = list(grid_dict.values())
        return points[indices[:max_points]]
    
    @staticmethod
    def _farthest_point_sample(points: np.ndarray, max_points: int) -> np.ndarray:
        """Farthest point sampling for better coverage"""
        xyz = np.column_stack([points['X'], points['Y'], points['Z']])
        
        # Start with random point
        selected = [np.random.randint(len(points))]
        
        for _ in range(max_points - 1):
            # Calculate distances to all selected points
            distances = np.min([
                np.linalg.norm(xyz - xyz[idx], axis=1) 
                for idx in selected
            ], axis=0)
            
            # Select farthest point
            farthest_idx = np.argmax(distances)
            selected.append(farthest_idx)
            
        return points[selected]
    
    @staticmethod
    def normalize_coordinates(points: np.ndarray, 
                            target_range: Tuple[float, float] = (-5, 5)) -> np.ndarray:
        """
        Normalize point coordinates for animation display
        
        Args:
            points: Input point array
            target_range: Target coordinate range
            
        Returns:
            Points with normalized coordinates
        """
        normalized_points = points.copy()
        
        for coord in ['X', 'Y', 'Z']:
            values = points[coord]
            min_val, max_val = values.min(), values.max()
            
            if max_val > min_val:
                # Normalize to [0, 1] then scale to target range
                normalized = (values - min_val) / (max_val - min_val)
                range_size = target_range[1] - target_range[0]
                normalized_points[coord] = normalized * range_size + target_range[0]
            else:
                normalized_points[coord] = target_range[0]
                
        return normalized_points
    
    @staticmethod
    def separate_by_classification(points: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Separate points by classification codes
        
        Args:
            points: Point array with Classification field
            
        Returns:
            Dictionary mapping classification codes to point arrays
        """
        classifications = {}
        unique_classes = np.unique(points['Classification'])
        
        for class_code in unique_classes:
            mask = points['Classification'] == class_code
            classifications[class_code] = points[mask]
            
        return classifications


class CSFParameterExplorer:
    """Explore CSF parameter effects for educational purposes"""
    
    PARAMETER_RANGES = {
        'resolution': (0.1, 2.0),
        'threshold': (0.05, 1.0),
        'rigidness': (1, 3),
        'iterations': (100, 1000),
        'time_step': (0.1, 1.0)
    }
    
    PARAMETER_DESCRIPTIONS = {
        'resolution': 'Cloth grid cell size (smaller = finer detail)',
        'threshold': 'Distance threshold for ground classification',
        'rigidness': 'Cloth stiffness (higher = less deformable)',
        'iterations': 'Number of simulation steps',
        'time_step': 'Simulation time step size'
    }
    
    @classmethod
    def generate_parameter_sets(cls, 
                              base_params: Dict[str, float],
                              vary_param: str,
                              num_steps: int = 5) -> List[Dict[str, float]]:
        """
        Generate parameter sets varying one parameter
        
        Args:
            base_params: Base parameter configuration
            vary_param: Parameter to vary
            num_steps: Number of parameter values to generate
            
        Returns:
            List of parameter dictionaries
        """
        if vary_param not in cls.PARAMETER_RANGES:
            raise ValueError(f"Unknown parameter: {vary_param}")
            
        param_range = cls.PARAMETER_RANGES[vary_param]
        values = np.linspace(param_range[0], param_range[1], num_steps)
        
        parameter_sets = []
        for value in values:
            param_set = base_params.copy()
            param_set[vary_param] = value
            parameter_sets.append(param_set)
            
        return parameter_sets
    
    @classmethod
    def get_extreme_comparisons(cls) -> List[Tuple[str, Dict[str, float]]]:
        """
        Get extreme parameter configurations for comparison
        
        Returns:
            List of (description, parameters) tuples
        """
        comparisons = [
            ("Conservative (Fine resolution, low threshold)", {
                'resolution': 0.1,
                'threshold': 0.1,
                'rigidness': 3,
                'iterations': 1000,
                'time_step': 0.65
            }),
            ("Aggressive (Coarse resolution, high threshold)", {
                'resolution': 1.0,
                'threshold': 0.8,
                'rigidness': 1,
                'iterations': 200,
                'time_step': 0.65
            }),
            ("Balanced (Default settings)", {
                'resolution': 0.5,
                'threshold': 0.25,
                'rigidness': 2,
                'iterations': 500,
                'time_step': 0.65
            })
        ]
        
        return comparisons


# Example usage and testing functions
def create_sample_data(output_file: str = "sample_data.las"):
    """Create sample LAS file for testing"""
    if not PDAL_AVAILABLE:
        print("PDAL not available - cannot create sample file")
        return None
        
    # Generate synthetic data
    processor = AnimationDataProcessor()
    manager = CSFPipelineManager()
    points = manager.generate_synthetic_result()
    
    # Create pipeline to write LAS file
    pipeline = {
        "pipeline": [
            {
                "type": "writers.las",
                "filename": output_file,
                "scale_x": 0.01,
                "scale_y": 0.01,
                "scale_z": 0.01
            }
        ]
    }
    
    try:
        pdal_pipeline = pdal.Pipeline(json.dumps(pipeline))
        pdal_pipeline.execute()
        print(f"Sample data created: {output_file}")
        return output_file
    except Exception as e:
        print(f"Failed to create sample data: {e}")
        return None


def demonstrate_parameter_effects():
    """Demonstrate how different parameters affect CSF results"""
    manager = CSFPipelineManager()
    explorer = CSFParameterExplorer()
    
    # Get extreme parameter comparisons
    comparisons = explorer.get_extreme_comparisons()
    
    print("CSF Parameter Effects Demonstration")
    print("=" * 40)
    
    for description, params in comparisons:
        print(f"\n{description}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
            
    # Show parameter variation effects
    base_params = {'resolution': 0.5, 'threshold': 0.25, 'rigidness': 2, 
                   'iterations': 500, 'time_step': 0.65}
    
    print(f"\nVarying threshold from {explorer.PARAMETER_RANGES['threshold']}:")
    threshold_sets = explorer.generate_parameter_sets(base_params, 'threshold', 3)
    for i, params in enumerate(threshold_sets):
        print(f"  Set {i+1}: threshold = {params['threshold']:.2f}")


if __name__ == "__main__":
    # Demonstration and testing
    print("CSF Animation PDAL Utilities")
    print("=" * 30)
    
    # Test basic functionality
    manager = CSFPipelineManager()
    
    # Create sample pipeline
    pipeline = manager.create_csf_pipeline("input.las", "output.las")
    print("Sample Pipeline:")
    print(json.dumps(pipeline, indent=2))
    
    # Demonstrate parameter effects
    demonstrate_parameter_effects()
    
    # Test data processing
    processor = AnimationDataProcessor()
    points = manager.generate_synthetic_result()
    
    print(f"\nGenerated {len(points)} synthetic points")
    print(f"Ground points: {np.sum(points['Classification'] == 2)}")
    print(f"Non-ground points: {np.sum(points['Classification'] == 1)}")
    
    # Test subsampling
    subsampled = processor.subsample_points(points, max_points=100)
    print(f"Subsampled to {len(subsampled)} points")
    
    # Test normalization
    normalized = processor.normalize_coordinates(subsampled)
    print(f"Coordinate ranges after normalization:")
    print(f"  X: [{normalized['X'].min():.2f}, {normalized['X'].max():.2f}]")
    print(f"  Y: [{normalized['Y'].min():.2f}, {normalized['Y'].max():.2f}]")
    print(f"  Z: [{normalized['Z'].min():.2f}, {normalized['Z'].max():.2f}]")
