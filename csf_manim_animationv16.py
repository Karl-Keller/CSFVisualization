"""
CSF (Cloth Simulation Filter) Algorithm Animation using Manim and PDAL
Author: Generated for nonComp PPE project
Description: Educational animation explaining ground point filtering using cloth simulation

This comprehensive animation demonstrates:
- CSF algorithm theory and implementation
- Step-by-step visual explanation
- PDAL integration for real-world usage
- Parameter tuning effects
- Interactive exploration capabilities
"""

from manim import *
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Define custom colors
BROWN = "#8B4513"
LIGHT_GRAY = "#D3D3D3"

# Try to import PDAL - graceful fallback if not available
try:
    import pdal
    PDAL_AVAILABLE = True
except ImportError:
    PDAL_AVAILABLE = False
    print("Warning: PDAL not available. Using synthetic data.")

# Import our utility classes
try:
    from pdal_utils import CSFPipelineManager, AnimationDataProcessor, CSFParameterExplorer
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: pdal_utils not found. Using embedded implementations.")

class CSFAlgorithmAnimation(Scene):
    def construct(self):
        # Title sequence
        self.play_title_sequence()
        
        # Step 1: Introduce the problem
        self.introduce_problem()
        
        # Step 2: Show algorithm overview
        self.algorithm_overview()
        
        # Step 3: Demonstrate cloth initialization
        self.cloth_initialization()
        
        # Step 4: Show physics simulation
        self.physics_simulation()
        
        # Step 5: Ground point classification
        self.ground_classification()
        
        # Step 6: Real data example with PDAL
        self.real_data_example()
        
        # Step 7: Parameters and tuning
        self.parameter_tuning()
        
        # Step 8: Conclusion
        self.conclusion()

    def play_title_sequence(self):
        """Opening title sequence"""
        title = Text("CSF Algorithm", font_size=72, color=BLUE)
        subtitle = Text("Cloth Simulation Filter for Ground Point Detection", 
                       font_size=36, color=WHITE)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        # Add 3D effect
        title_3d = title.copy().set_color(BLUE_E).shift(0.1*LEFT + 0.1*DOWN)
        
        self.play(
            Write(title_3d),
            Write(title),
            FadeIn(subtitle, shift=UP)
        )
        self.wait(2)
        self.play(FadeOut(title_3d, title, subtitle))

    def introduce_problem(self):
        """Introduce the ground filtering problem"""
        problem_title = Text("The Problem: Ground vs Non-Ground Points", 
                           font_size=48, color=YELLOW)
        problem_title.to_edge(UP)
        
        # Create 3D-like point cloud visualization
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 4, 1],
            x_length=8,
            y_length=6,
            tips=False
        )
        
        # Generate synthetic LiDAR points
        ground_points, vegetation_points, building_points = self.generate_synthetic_lidar()
        
        # Create point groups
        ground_dots = VGroup(*[
            Dot(axes.coords_to_point(x, z), color=BROWN, radius=0.03) 
            for x, y, z in ground_points
        ])
        
        vegetation_dots = VGroup(*[
            Dot(axes.coords_to_point(x, z), color=GREEN, radius=0.03) 
            for x, y, z in vegetation_points
        ])
        
        building_dots = VGroup(*[
            Dot(axes.coords_to_point(x, z), color=GRAY, radius=0.03) 
            for x, y, z in building_points
        ])
        
        self.play(Write(problem_title))
        self.play(Create(axes))
        
        # Animate point appearance
        self.play(FadeIn(ground_dots, lag_ratio=0.1))
        self.wait(0.5)
        self.play(FadeIn(vegetation_dots, lag_ratio=0.1))
        self.wait(0.5)
        self.play(FadeIn(building_dots, lag_ratio=0.1))
        
        # Add labels
        ground_label = Text("Ground Points", color=BROWN, font_size=24)
        vegetation_label = Text("Vegetation", color=GREEN, font_size=24)
        building_label = Text("Buildings", color=GRAY, font_size=24)
        
        labels = VGroup(ground_label, vegetation_label, building_label)
        labels.arrange(RIGHT, buff=1).to_edge(DOWN)
        
        self.play(Write(labels))
        self.wait(2)
        
        # Store for later use
        self.ground_points = ground_points
        self.vegetation_points = vegetation_points
        self.building_points = building_points
        self.axes = axes
        
        self.play(FadeOut(problem_title, labels))

    def algorithm_overview(self):
        """Show CSF algorithm overview"""
        title = Text("CSF Algorithm Overview", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        # Algorithm steps
        steps = [
            "1. Initialize virtual cloth above point cloud",
            "2. Apply gravity and collision detection",
            "3. Simulate cloth settling on surface",
            "4. Classify points based on distance to cloth",
            "5. Refine classification iteratively"
        ]
        
        step_objects = VGroup()
        for i, step in enumerate(steps):
            step_text = Text(step, font_size=28, color=WHITE)
            if i == 0:
                step_text.next_to(title, DOWN, buff=1)
            else:
                step_text.next_to(step_objects[-1], DOWN, buff=0.5, aligned_edge=LEFT)
            step_objects.add(step_text)
        
        self.play(Write(title))
        
        for step in step_objects:
            self.play(Write(step))
            self.wait(0.8)
        
        self.wait(2)
        self.play(FadeOut(title, step_objects))

    def cloth_initialization(self):
        """Demonstrate cloth initialization"""
        title = Text("Step 1: Cloth Initialization", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        # Show point cloud again
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        point_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=WHITE, radius=0.02) 
            for x, y, z in all_points
        ])
        
        self.play(Write(title))
        self.play(Create(self.axes), FadeIn(point_dots))
        
        # Find bounding box
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_z, max_z = all_points[:, 2].min(), all_points[:, 2].max()
        
        # Create cloth grid above points
        cloth_resolution = 0.5
        cloth_height = max_z + 2
        
        x_coords = np.arange(min_x - 1, max_x + 1, cloth_resolution)
        y_coords = np.arange(min_z - 1, max_z + 1, cloth_resolution)
        
        # Create cloth grid visualization
        cloth_points = []
        cloth_lines = VGroup()
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                cloth_points.append([x, 0, cloth_height])
                # Add grid lines
                if i < len(x_coords) - 1:
                    line = Line(
                        self.axes.coords_to_point(x, cloth_height),
                        self.axes.coords_to_point(x + cloth_resolution, cloth_height),
                        color=BLUE, stroke_width=1
                    )
                    cloth_lines.add(line)
                if j < len(y_coords) - 1:
                    line = Line(
                        self.axes.coords_to_point(x, cloth_height),
                        self.axes.coords_to_point(x, cloth_height + cloth_resolution),
                        color=BLUE, stroke_width=1
                    )
                    cloth_lines.add(line)
        
        # Add cloth particles
        cloth_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=BLUE, radius=0.02) 
            for x, y, z in cloth_points
        ])
        
        # Animate cloth appearance
        explanation = Text("Virtual cloth initialized above highest points", 
                         font_size=24, color=YELLOW)
        explanation.to_edge(DOWN)
        
        self.play(Write(explanation))
        self.play(Create(cloth_lines), FadeIn(cloth_dots, lag_ratio=0.1))
        
        self.wait(2)
        
        # Store cloth for animation
        self.cloth_points = np.array(cloth_points)
        self.cloth_dots = cloth_dots
        self.cloth_lines = cloth_lines
        
        self.play(FadeOut(title, explanation))

    def physics_simulation(self):
        """Animate physics simulation"""
        title = Text("Step 2: Physics Simulation", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Parameters
        gravity = 0.1
        iterations = 20
        
        explanation = Text("Cloth falls under gravity with collision detection", 
                         font_size=24, color=YELLOW)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        
        # Animate cloth falling and settling
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        for iteration in range(iterations):
            new_cloth_positions = []
            
            for i, (x, y, z) in enumerate(self.cloth_points):
                # Apply gravity
                new_z = z - gravity
                
                # Collision detection with point cloud
                nearby_points = all_points[
                    (np.abs(all_points[:, 0] - x) < 0.3) & 
                    (np.abs(all_points[:, 1] - y) < 0.3)
                ]
                
                if len(nearby_points) > 0:
                    max_nearby_z = nearby_points[:, 2].max()
                    if new_z < max_nearby_z + 0.1:
                        new_z = max_nearby_z + 0.1
                
                new_cloth_positions.append([x, y, new_z])
            
            self.cloth_points = np.array(new_cloth_positions)
            
            # Update visualization
            new_cloth_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=BLUE, radius=0.02) 
                for x, y, z in self.cloth_points
            ])
            
            if iteration % 3 == 0:  # Update every few iterations for smooth animation
                self.play(
                    Transform(self.cloth_dots, new_cloth_dots),
                    run_time=0.3
                )
        
        settled_explanation = Text("Cloth settles on surface features", 
                                 font_size=24, color=GREEN)
        settled_explanation.to_edge(DOWN)
        
        self.play(Transform(explanation, settled_explanation))
        self.wait(2)
        self.play(FadeOut(title, explanation))

    def ground_classification(self):
        """Show ground point classification"""
        title = Text("Step 3: Ground Classification", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Calculate distances from points to cloth
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        threshold = 0.3
        
        ground_classified = []
        non_ground_classified = []
        
        for point in all_points:
            x, y, z = point
            
            # Find nearest cloth point (simplified)
            distances = np.sqrt(
                (self.cloth_points[:, 0] - x)**2 + 
                (self.cloth_points[:, 2] - z)**2
            )
            nearest_cloth_z = self.cloth_points[np.argmin(distances), 2]
            
            distance_to_cloth = abs(z - nearest_cloth_z)
            
            if distance_to_cloth < threshold:
                ground_classified.append(point)
            else:
                non_ground_classified.append(point)
        
        # Create classification visualization
        ground_classified_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.03) 
            for x, y, z in ground_classified
        ])
        
        non_ground_classified_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=RED, radius=0.03) 
            for x, y, z in non_ground_classified
        ])
        
        explanation = Text(f"Points within {threshold}m of cloth = Ground (Brown)", 
                         font_size=24, color=YELLOW)
        explanation.to_edge(DOWN)
        
        self.play(Write(explanation))
        self.play(FadeIn(ground_classified_dots, lag_ratio=0.1))
        self.wait(1)
        
        explanation2 = Text("Points above threshold = Non-Ground (Red)", 
                          font_size=24, color=YELLOW)
        explanation2.to_edge(DOWN)
        
        self.play(Transform(explanation, explanation2))
        self.play(FadeIn(non_ground_classified_dots, lag_ratio=0.1))
        
        self.wait(2)
        self.play(FadeOut(title, explanation, ground_classified_dots, non_ground_classified_dots))

    def real_data_example(self):
        """Show example with real/realistic data using PDAL"""
        title = Text("Real Data Example with PDAL", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Show PDAL pipeline configuration
        pipeline_title = Text("PDAL Pipeline Configuration", font_size=32, color=YELLOW)
        pipeline_title.next_to(title, DOWN, buff=0.8)
        
        if PDAL_AVAILABLE and UTILS_AVAILABLE:
            # Create actual PDAL pipeline
            manager = CSFPipelineManager()
            pipeline_dict = manager.create_csf_pipeline("input.las", "output.las")
            pipeline_text = self.format_pipeline_json(pipeline_dict)
        else:
            # Show example pipeline
            pipeline_text = self.get_pdal_pipeline_code()
        
        code_display = Text(
            pipeline_text,
            font_size=14,
            color=WHITE,
            font="monospace",
            line_spacing=1.2
        )
        code_display.next_to(pipeline_title, DOWN, buff=0.5)
        
        # Add execution demonstration
        execution_demo = VGroup()
        
        # Input visualization
        input_box = Rectangle(width=2, height=1, color=BLUE)
        input_label = Text("input.las", font_size=16, color=WHITE)
        input_label.move_to(input_box)
        input_group = VGroup(input_box, input_label)
        
        # CSF filter
        filter_box = Rectangle(width=3, height=1.5, color=GREEN)
        filter_label = Text("filters.csf", font_size=16, color=WHITE)
        filter_params = Text("resolution: 0.5\nthreshold: 0.25", font_size=12, color=LIGHT_GRAY)
        filter_params.next_to(filter_label, DOWN, buff=0.1)
        filter_content = VGroup(filter_label, filter_params)
        filter_content.move_to(filter_box)
        filter_group = VGroup(filter_box, filter_content)
        
        # Output visualization
        output_box = Rectangle(width=2, height=1, color=RED)
        output_label = Text("output.las", font_size=16, color=WHITE)
        output_label.move_to(output_box)
        output_group = VGroup(output_box, output_label)
        
        # Arrange pipeline flow
        execution_demo.add(input_group, filter_group, output_group)
        execution_demo.arrange(RIGHT, buff=1)
        execution_demo.to_edge(DOWN, buff=1)
        
        # Add arrows
        arrow1 = Arrow(input_group.get_right(), filter_group.get_left(), color=WHITE)
        arrow2 = Arrow(filter_group.get_right(), output_group.get_left(), color=WHITE)
        
        self.play(Write(pipeline_title))
        self.play(Write(code_display))
        self.wait(2)
        
        # Animate pipeline execution
        self.play(FadeIn(execution_demo))
        self.play(Create(arrow1))
        self.wait(0.5)
        self.play(Create(arrow2))
        
        # Show processing animation
        processing_text = Text("Processing...", font_size=24, color=YELLOW)
        processing_text.next_to(filter_group, UP, buff=0.5)
        
        self.play(Write(processing_text))
        
        # Simulate processing with pulsing effect
        for _ in range(3):
            self.play(filter_group.animate.scale(1.1), run_time=0.3)
            self.play(filter_group.animate.scale(1/1.1), run_time=0.3)
        
        # Show completion
        complete_text = Text("Classification Complete!", font_size=24, color=GREEN)
        complete_text.move_to(processing_text)
        
        self.play(Transform(processing_text, complete_text))
        self.wait(2)
        
        # Show results visualization if we have real data
        if PDAL_AVAILABLE and UTILS_AVAILABLE:
            self.show_real_data_results()
        
        self.play(FadeOut(title, pipeline_title, code_display, execution_demo, 
                         arrow1, arrow2, processing_text))

    def show_real_data_results(self):
        """Show actual CSF results on real data"""
        try:
            manager = CSFPipelineManager()
            processor = AnimationDataProcessor()
            
            # Generate synthetic data that mimics real LiDAR
            points = manager.generate_synthetic_result()
            
            # Process for animation
            subsampled = processor.subsample_points(points, max_points=300)
            normalized = processor.normalize_coordinates(subsampled, (-4, 4))
            
            # Separate by classification
            classifications = processor.separate_by_classification(normalized)
            
            # Create visualization
            results_title = Text("Classification Results", font_size=32, color=YELLOW)
            results_title.to_edge(UP, buff=1)
            
            self.play(Write(results_title))
            
            # Show before/after comparison
            before_after = VGroup()
            
            # Before (all points same color)
            before_axes = Axes(
                x_range=[-4, 4, 2], y_range=[-2, 4, 2],
                x_length=4, y_length=3, tips=False
            )
            before_title = Text("Before CSF", font_size=20, color=WHITE)
            before_title.next_to(before_axes, UP)
            
            before_dots = VGroup(*[
                Dot(before_axes.coords_to_point(p['X'], p['Z']), 
                    color=WHITE, radius=0.02) 
                for p in normalized
            ])
            
            before_group = VGroup(before_axes, before_title, before_dots)
            
            # After (colored by classification)
            after_axes = Axes(
                x_range=[-4, 4, 2], y_range=[-2, 4, 2],
                x_length=4, y_length=3, tips=False
            )
            after_title = Text("After CSF", font_size=20, color=WHITE)
            after_title.next_to(after_axes, UP)
            
            after_dots = VGroup()
            for class_code, class_points in classifications.items():
                color = BROWN if class_code == 2 else RED  # Ground vs non-ground
                dots = VGroup(*[
                    Dot(after_axes.coords_to_point(p['X'], p['Z']), 
                        color=color, radius=0.02) 
                    for p in class_points
                ])
                after_dots.add(dots)
            
            after_group = VGroup(after_axes, after_title, after_dots)
            
            # Arrange side by side
            before_after.add(before_group, after_group)
            before_after.arrange(RIGHT, buff=2)
            before_after.center()
            
            self.play(Create(before_group))
            self.wait(1)
            self.play(Create(after_group))
            
            # Add legend
            legend = VGroup()
            ground_legend = VGroup(
                Dot(color=BROWN, radius=0.05),
                Text("Ground Points", font_size=16, color=BROWN)
            )
            ground_legend.arrange(RIGHT, buff=0.3)
            
            nonground_legend = VGroup(
                Dot(color=RED, radius=0.05),
                Text("Non-Ground Points", font_size=16, color=RED)
            )
            nonground_legend.arrange(RIGHT, buff=0.3)
            
            legend.add(ground_legend, nonground_legend)
            legend.arrange(DOWN, buff=0.3)
            legend.to_edge(DOWN)
            
            self.play(FadeIn(legend))
            self.wait(3)
            
            self.play(FadeOut(results_title, before_after, legend))
            
        except Exception as e:
            print(f"Error showing real data results: {e}")

    def format_pipeline_json(self, pipeline_dict: Dict) -> str:
        """Format pipeline dictionary for display"""
        # Simplify for display
        simplified = {
            "pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {
                    "type": "filters.csf",
                    "resolution": pipeline_dict["pipeline"][1].get("resolution", 0.5),
                    "threshold": pipeline_dict["pipeline"][1].get("threshold", 0.25),
                    "rigidness": pipeline_dict["pipeline"][1].get("rigidness", 2)
                },
                {"type": "writers.las", "filename": "output.las"}
            ]
        }
        
        return json.dumps(simplified, indent=2)
    
    # _code()
            
    #         code_text = Text(pipeline_code, font_size=16, color=WHITE, font="monospace")
    #         code_text.next_to(no_pdal_text, DOWN, buff=0.5)
            
    #         self.play(Write(no_pdal_text))
    #         self.play(Write(code_text))
    #         self.wait(3)
    #         self.play(FadeOut(no_pdal_text, code_text))
        
    #     self.play(FadeOut(title))

    def parameter_tuning(self):
        """Show parameter effects with interactive demonstrations"""
        title = Text("Parameter Tuning Effects", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Show parameter comparison
        if UTILS_AVAILABLE:
            explorer = CSFParameterExplorer()
            comparisons = explorer.get_extreme_comparisons()
            
            self.show_parameter_comparisons(comparisons)
        else:
            self.show_basic_parameter_effects()
        
        # Show parameter ranges and effects
        param_explanation = Text("Key Parameters and Their Effects", 
                               font_size=32, color=YELLOW)
        param_explanation.to_edge(UP, buff=2)
        
        self.play(Transform(title, param_explanation))
        
        # Create parameter visualization
        param_demos = VGroup()
        
        # Resolution effect
        res_demo = self.create_parameter_demo(
            "Resolution", 
            ["Coarse (1.0m)", "Medium (0.5m)", "Fine (0.1m)"],
            [BLUE, GREEN, RED],
            "Grid density affects detail level"
        )
        
        # Threshold effect  
        thresh_demo = self.create_parameter_demo(
            "Threshold",
            ["Conservative (0.1m)", "Balanced (0.25m)", "Aggressive (0.5m)"],
            [GREEN, YELLOW, RED],
            "Distance tolerance for ground classification"
        )
        
        param_demos.add(res_demo, thresh_demo)
        param_demos.arrange(DOWN, buff=1)
        param_demos.center()
        
        for demo in param_demos:
            self.play(FadeIn(demo))
            self.wait(2)
        
        # Interactive parameter slider visualization
        self.show_parameter_slider_demo()
        
        self.wait(2)
        self.play(FadeOut(title, param_demos))

    def create_parameter_demo(self, param_name: str, 
                            variations: List[str], 
                            colors: List[str],
                            description: str) -> VGroup:
        """Create a visual demonstration of parameter effects"""
        demo_group = VGroup()
        
        # Parameter name
        param_title = Text(f"{param_name} Effect", font_size=24, color=WHITE)
        demo_group.add(param_title)
        
        # Visual comparison
        comparison_group = VGroup()
        
        for i, (variation, color) in enumerate(zip(variations, colors)):
            # Create mini visualization
            demo_rect = Rectangle(width=1.5, height=1, color=color, fill_opacity=0.3)
            demo_label = Text(variation, font_size=14, color=WHITE)
            demo_label.next_to(demo_rect, DOWN, buff=0.1)
            
            # Add sample points with different density/classification
            if "Resolution" in param_name:
                # Show different grid densities
                grid_size = 0.3 - i * 0.1  # Finer grid for higher resolution
                points = self.create_grid_points(demo_rect, grid_size, color)
            else:
                # Show different classification results
                points = self.create_classification_points(demo_rect, i, color)
            
            variation_group = VGroup(demo_rect, demo_label, points)
            comparison_group.add(variation_group)
        
        comparison_group.arrange(RIGHT, buff=0.5)
        comparison_group.next_to(param_title, DOWN, buff=0.5)
        demo_group.add(comparison_group)
        
        # Description
        desc_text = Text(description, font_size=16, color=GRAY)
        desc_text.next_to(comparison_group, DOWN, buff=0.3)
        demo_group.add(desc_text)
        
        return demo_group

    def create_grid_points(self, container: Rectangle, grid_size: float, color: str) -> VGroup:
        """Create grid points to show resolution effects"""
        points = VGroup()
        
        # Create grid within container bounds
        width = container.width
        height = container.height
        center = container.get_center()
        
        x_steps = int(width / grid_size)
        y_steps = int(height / grid_size)
        
        for i in range(x_steps):
            for j in range(y_steps):
                x = center[0] - width/2 + i * grid_size
                y = center[1] - height/2 + j * grid_size
                point = Dot([x, y, 0], radius=0.02, color=color)
                points.add(point)
        
        return points

    def create_classification_points(self, container: Rectangle, 
                                   aggressiveness: int, color: str) -> VGroup:
        """Create points showing classification aggressiveness"""
        points = VGroup()
        
        # Generate random points
        np.random.seed(42 + aggressiveness)
        n_points = 20
        
        width = container.width * 0.8
        height = container.height * 0.8
        center = container.get_center()
        
        for _ in range(n_points):
            x = center[0] + np.random.uniform(-width/2, width/2)
            y = center[1] + np.random.uniform(-height/2, height/2)
            
            # Different classification based on aggressiveness
            if aggressiveness == 0:  # Conservative - fewer ground points
                point_color = BROWN if np.random.random() < 0.3 else RED
            elif aggressiveness == 1:  # Balanced
                point_color = BROWN if np.random.random() < 0.6 else RED
            else:  # Aggressive - more ground points
                point_color = BROWN if np.random.random() < 0.8 else RED
            
            point = Dot([x, y, 0], radius=0.02, color=point_color)
            points.add(point)
        
        return points

    def show_parameter_slider_demo(self):
        """Show interactive parameter adjustment visualization"""
        slider_title = Text("Interactive Parameter Adjustment", 
                           font_size=28, color=YELLOW)
        slider_title.to_edge(UP, buff=3)
        
        self.play(Write(slider_title))
        
        # Create slider visualization
        slider_bg = Rectangle(width=6, height=0.3, color=GRAY)
        slider_handle = Circle(radius=0.2, color=BLUE, fill_opacity=1)
        slider_handle.move_to(slider_bg.get_left())
        
        slider_label = Text("Threshold", font_size=20, color=WHITE)
        slider_label.next_to(slider_bg, LEFT, buff=0.5)
        
        value_display = Text("0.1", font_size=20, color=WHITE)
        value_display.next_to(slider_bg, RIGHT, buff=0.5)
        
        slider_group = VGroup(slider_label, slider_bg, slider_handle, value_display)
        slider_group.center()
        
        # Result visualization area
        result_area = Rectangle(width=8, height=4, color=WHITE, fill_opacity=0.1)
        result_area.next_to(slider_group, DOWN, buff=1)
        
        # Create sample points for demonstration
        sample_points = self.create_sample_scene_for_slider(result_area)
        
        self.play(Create(slider_group), Create(result_area))
        self.play(FadeIn(sample_points))
        
        # Animate slider movement and show effects
        threshold_values = [0.1, 0.25, 0.4, 0.25, 0.1]
        
        for threshold in threshold_values:
            # Move slider
            new_x = slider_bg.get_left()[0] + (threshold - 0.1) / 0.4 * slider_bg.width
            target_pos = [new_x, slider_handle.get_center()[1], 0]
            
            # Update value display
            new_value = Text(f"{threshold:.2f}", font_size=20, color=WHITE)
            new_value.move_to(value_display)
            
            # Update point colors based on threshold
            new_points = self.update_points_for_threshold(sample_points, threshold, result_area)
            
            self.play(
                slider_handle.animate.move_to(target_pos),
                Transform(value_display, new_value),
                Transform(sample_points, new_points),
                run_time=1.5
            )
            self.wait(0.5)
        
        self.play(FadeOut(slider_title, slider_group, result_area, sample_points))

    def create_sample_scene_for_slider(self, container: Rectangle) -> VGroup:
        """Create sample scene for slider demonstration"""
        points = VGroup()
        
        # Generate synthetic terrain-like points
        np.random.seed(42)
        n_points = 50
        
        width = container.width * 0.9
        height = container.height * 0.9
        center = container.get_center()
        
        for i in range(n_points):
            x = center[0] + np.random.uniform(-width/2, width/2)
            
            # Create terrain-like Z distribution
            terrain_z = 0.1 * np.sin(3 * (x - center[0])) + 0.05 * np.random.normal()
            
            # Add some vegetation/building points
            if np.random.random() < 0.3:  # 30% chance of elevated points
                z = terrain_z + np.random.uniform(0.2, 1.0)
            else:
                z = terrain_z
            
            y = center[1] + z * height/2  # Map z to y for 2D display
            
            # Initial coloring (will be updated by slider)
            point = Dot([x, y, 0], radius=0.03, color=WHITE)
            point.z_value = z  # Store actual z value for threshold comparison
            point.terrain_z = terrain_z  # Store terrain height
            
            points.add(point)
        
        return points

    def update_points_for_threshold(self, points: VGroup, 
                                  threshold: float, 
                                  container: Rectangle) -> VGroup:
        """Update point colors based on threshold"""
        new_points = VGroup()
        
        for point in points:
            # Calculate distance to "cloth" (simplified as terrain)
            distance = abs(point.z_value - point.terrain_z)
            
            # Classify based on threshold
            if distance < threshold:
                color = BROWN  # Ground
            else:
                color = RED    # Non-ground
            
            new_point = Dot(point.get_center(), radius=0.03, color=color)
            new_point.z_value = point.z_value
            new_point.terrain_z = point.terrain_z
            
            new_points.add(new_point)
        
        return new_points

    def show_parameter_comparisons(self, comparisons: List[Tuple[str, Dict]]):
        """Show side-by-side parameter comparisons"""
        comparison_title = Text("Parameter Set Comparisons", 
                              font_size=32, color=YELLOW)
        comparison_title.to_edge(UP, buff=1.5)
        
        self.play(Write(comparison_title))
        
        # Create comparison grid
        comparison_group = VGroup()
        
        for i, (description, params) in enumerate(comparisons):
            # Create comparison box
            comp_box = Rectangle(width=4, height=3, color=WHITE, fill_opacity=0.1)
            
            # Title
            comp_title = Text(description.split('(')[0], font_size=16, color=WHITE)
            comp_title.next_to(comp_box, UP, buff=0.1)
            
            # Parameters display
            param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
            param_display = Text(param_text, font_size=12, color=GRAY, font="monospace")
            param_display.move_to(comp_box.get_top() + DOWN * 0.5)
            
            # Simulated result visualization
            result_viz = self.create_result_visualization(params, comp_box)
            
            comp_item = VGroup(comp_box, comp_title, param_display, result_viz)
            comparison_group.add(comp_item)
        
        comparison_group.arrange(RIGHT, buff=0.5)
        comparison_group.next_to(comparison_title, DOWN, buff=1)
        
        for comp in comparison_group:
            self.play(FadeIn(comp))
            self.wait(1)
        
        self.wait(2)
        self.play(FadeOut(comparison_title, comparison_group))

    def create_result_visualization(self, params: Dict, container: Rectangle) -> VGroup:
        """Create visualization showing parameter effects"""
        viz_group = VGroup()
        
        # Extract key parameters
        resolution = params.get('resolution', 0.5)
        threshold = params.get('threshold', 0.25)
        
        # Create visualization based on parameters
        width = container.width * 0.8
        height = container.height * 0.4
        center = container.get_center() + DOWN * 0.5
        
        # Grid density based on resolution
        grid_size = max(0.1, resolution * 0.2)
        x_steps = int(width / grid_size)
        y_steps = int(height / grid_size)
        
        for i in range(min(x_steps, 15)):  # Limit for visualization
            for j in range(min(y_steps, 8)):
                x = center[0] - width/2 + i * grid_size
                y = center[1] - height/2 + j * grid_size
                
                # Classification based on threshold (simplified)
                if threshold < 0.2:  # Conservative
                    color = BROWN if np.random.random() < 0.4 else RED
                elif threshold > 0.4:  # Aggressive
                    color = BROWN if np.random.random() < 0.8 else RED
                else:  # Balanced
                    color = BROWN if np.random.random() < 0.6 else RED
                
                point = Dot([x, y, 0], radius=0.02, color=color)
                viz_group.add(point)
        
        return viz_group

    def conclusion(self):
        """Conclusion and summary"""
        title = Text("CSF Algorithm Summary", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        summary_points = [
            "✓ Physics-based approach using cloth simulation",
            "✓ Robust for complex terrain and vegetation",
            "✓ Tunable parameters for different environments",
            "✓ Widely implemented in PDAL and other tools",
            "✓ Effective for LiDAR ground filtering"
        ]
        
        self.play(Write(title))
        
        summary_group = VGroup()
        for point in summary_points:
            text = Text(point, font_size=28, color=WHITE)
            summary_group.add(text)
        
        summary_group.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        summary_group.center()
        
        for point in summary_group:
            self.play(Write(point))
            self.wait(0.8)
        
        # Final animation
        thank_you = Text("Thank you!", font_size=48, color=YELLOW)
        thank_you.to_edge(DOWN)
        
        self.play(Write(thank_you))
        self.wait(2)
        
        # Fade out everything
        self.play(FadeOut(*self.mobjects))

    def generate_synthetic_lidar(self):
        """Generate synthetic LiDAR-like data"""
        np.random.seed(42)
        
        # Ground points (terrain)
        x_ground = np.linspace(-4, 4, 50)
        y_ground = np.zeros(50)
        z_ground = 0.2 * np.sin(x_ground) + 0.1 * np.random.normal(size=50)
        ground_points = np.column_stack([x_ground, y_ground, z_ground])
        
        # Vegetation points
        veg_x = np.random.uniform(-3, 3, 30)
        veg_y = np.zeros(30)
        veg_z = np.random.uniform(0.5, 3.5, 30)
        vegetation_points = np.column_stack([veg_x, veg_y, veg_z])
        
        # Building points
        building_x = np.random.uniform(1, 3, 20)
        building_y = np.zeros(20)
        building_z = np.random.uniform(2, 4, 20)
        building_points = np.column_stack([building_x, building_y, building_z])
        
        return ground_points, vegetation_points, building_points

    def create_pdal_pipeline_visualization(self):
        """Create visual representation of PDAL pipeline"""
        pipeline_json = {
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
        
        pipeline_text = Text(
            json.dumps(pipeline_json, indent=2),
            font_size=14,
            color=WHITE,
            font="monospace"
        )
        
        return pipeline_text

    def get_pdal_pipeline_code(self):
        """Get PDAL pipeline code as string"""
        return '''
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
        '''

# Additional scene for interactive parameter exploration
class CSFParameterExploration(Scene):
    def construct(self):
        """Interactive exploration of CSF parameters"""
        title = Text("CSF Parameter Effects", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Show side-by-side comparison with different parameters
        left_title = Text("Conservative (threshold=0.1)", font_size=24, color=GREEN)
        right_title = Text("Aggressive (threshold=0.5)", font_size=24, color=RED)
        
        left_title.to_corner(UL, buff=1)
        right_title.to_corner(UR, buff=1)
        
        self.play(Write(left_title), Write(right_title))
        
        # This would show actual filtering results with different parameters
        # Implementation would use real PDAL processing
        
        self.wait(3)

    def show_basic_parameter_effects(self):
        """Show basic parameter effects when utils not available"""
        parameters = [
            ("Cloth Resolution", "0.1 - 2.0m", "Finer resolution = better detail, slower processing"),
            ("Class Threshold", "0.1 - 1.0m", "Lower threshold = more conservative ground classification"),
            ("Rigidness", "1 - 3", "Higher rigidness = cloth less deformable"),
            ("Iterations", "100 - 1000", "More iterations = better convergence")
        ]
        
        param_table = VGroup()
        headers = ["Parameter", "Range", "Effect"]
        header_row = VGroup(*[Text(h, font_size=24, color=YELLOW) for h in headers])
        header_row.arrange(RIGHT, buff=2)
        param_table.add(header_row)
        
        for param, range_val, effect in parameters:
            row = VGroup(
                Text(param, font_size=20, color=WHITE),
                Text(range_val, font_size=20, color=BLUE),
                Text(effect, font_size=18, color=GRAY)
            )
            row.arrange(RIGHT, buff=2, aligned_edge=LEFT)
            if len(param_table) > 0:
                row[0].align_to(header_row[0], LEFT)
                row[1].align_to(header_row[1], LEFT)
                row[2].align_to(header_row[2], LEFT)
            param_table.add(row)
        
        param_table.arrange(DOWN, buff=0.3)
        param_table.center()
        
        for row in param_table:
            self.play(Write(row))
            self.wait(0.5)
        
        self.wait(2)
        self.play(FadeOut(param_table))

    def conclusion(self):
        """Enhanced conclusion with implementation guidance"""
        title = Text("CSF Algorithm: Complete Implementation Guide", 
                    font_size=40, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Algorithm summary
        summary_title = Text("Algorithm Summary", font_size=32, color=YELLOW)
        summary_title.next_to(title, DOWN, buff=1)
        
        summary_points = [
            "✓ Physics-based approach using cloth simulation",
            "✓ Robust for complex terrain and vegetation",
            "✓ Tunable parameters for different environments",
            "✓ Widely implemented in PDAL and other tools",
            "✓ Effective for LiDAR ground filtering"
        ]
        
        self.play(Write(summary_title))
        
        summary_group = VGroup()
        for point in summary_points:
            text = Text(point, font_size=24, color=WHITE)
            summary_group.add(text)
        
        summary_group.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        summary_group.next_to(summary_title, DOWN, buff=0.5)
        
        for point in summary_group:
            self.play(Write(point))
            self.wait(0.6)
        
        # Implementation guidance
        impl_title = Text("Implementation Tips", font_size=32, color=GREEN)
        impl_title.next_to(summary_group, DOWN, buff=1)
        
        self.play(Write(impl_title))
        
        impl_tips = [
            "• Start with default parameters and adjust based on terrain",
            "• Use finer resolution for detailed urban environments",
            "• Increase rigidness for steep terrain",
            "• Monitor processing time vs. accuracy trade-offs"
        ]
        
        impl_group = VGroup()
        for tip in impl_tips:
            text = Text(tip, font_size=20, color=LIGHT_GRAY)
            impl_group.add(text)
        
        impl_group.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        impl_group.next_to(impl_title, DOWN, buff=0.5)
        
        for tip in impl_group:
            self.play(Write(tip))
            self.wait(0.5)
        
        # PDAL command example
        if PDAL_AVAILABLE:
            pdal_title = Text("Ready to Use with PDAL:", font_size=28, color=BLUE)
            pdal_command = Text("pdal pipeline csf_pipeline.json", 
                             font_size=20, color=WHITE, font="monospace")
            pdal_command.next_to(pdal_title, DOWN, buff=0.3)
            
            pdal_group = VGroup(pdal_title, pdal_command)
            pdal_group.next_to(impl_group, DOWN, buff=1)
            
            self.play(Write(pdal_group))
            self.wait(2)
        
        # Final thank you with animation
        thank_you = Text("Thank you for Learning with Us!", 
                        font_size=36, color=YELLOW)
        thank_you.center()
        
        # Create sparkle effect
        sparkles = VGroup()
        for _ in range(20):
            sparkle = Star(n=4, outer_radius=0.1, color=YELLOW, fill_opacity=0.8)
            sparkle.move_to([
                np.random.uniform(-7, 7),
                np.random.uniform(-4, 4),
                0
            ])
            sparkles.add(sparkle)
        
        self.play(
            FadeOut(*[obj for obj in self.mobjects if obj != thank_you]),
            Write(thank_you)
        )
        
        # Animate sparkles
        self.play(LaggedStart(*[
            FadeIn(sparkle, scale=0.5) for sparkle in sparkles
        ], lag_ratio=0.1))
        
        # Rotate sparkles
        self.play(
            *[Rotate(sparkle, 2*PI) for sparkle in sparkles],
            run_time=3
        )
        
        self.wait(2)
        
        # Final fade out
        self.play(FadeOut(*self.mobjects))


# Additional specialized scenes for different use cases
class CSFParameterExploration(Scene):
    """Interactive exploration of CSF parameters"""
    
    def construct(self):
        title = Text("CSF Parameter Interactive Exploration", 
                    font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Create parameter control panel
        self.create_parameter_panel()
        
        # Create result visualization area
        self.create_result_area()
        
        # Demonstrate parameter effects
        self.demonstrate_parameter_effects()
        
        self.wait(3)

    def create_parameter_panel(self):
        """Create visual parameter control panel"""
        panel_title = Text("Parameter Controls", font_size=32, color=YELLOW)
        panel_title.to_edge(LEFT, buff=1).shift(UP * 2)
        
        self.play(Write(panel_title))
        
        # Create sliders for each parameter
        self.sliders = {}
        params = [
            ("resolution", 0.1, 2.0, 0.5),
            ("threshold", 0.05, 1.0, 0.25),
            ("rigidness", 1, 3, 2),
            ("iterations", 100, 1000, 500)
        ]
        
        slider_group = VGroup()
        
        for i, (name, min_val, max_val, default) in enumerate(params):
            slider = self.create_slider(name, min_val, max_val, default)
            slider_group.add(slider)
        
        slider_group.arrange(DOWN, buff=0.8)
        slider_group.next_to(panel_title, DOWN, buff=1)
        
        self.play(Create(slider_group))
        self.slider_group = slider_group

    def create_slider(self, name: str, min_val: float, 
                     max_val: float, default: float) -> VGroup:
        """Create individual parameter slider"""
        # Label
        label = Text(name.title(), font_size=20, color=WHITE)
        
        # Slider track
        track = Rectangle(width=3, height=0.1, color=GRAY)
        
        # Slider handle
        handle_pos = (default - min_val) / (max_val - min_val)
        handle_x = track.get_left()[0] + handle_pos * track.width
        handle = Circle(radius=0.1, color=BLUE, fill_opacity=1)
        handle.move_to([handle_x, track.get_center()[1], 0])
        
        # Value display
        value_text = Text(f"{default:.2f}", font_size=16, color=WHITE)
        value_text.next_to(track, RIGHT, buff=0.3)
        
        # Range labels
        min_label = Text(f"{min_val}", font_size=12, color=GRAY)
        max_label = Text(f"{max_val}", font_size=12, color=GRAY)
        min_label.next_to(track, LEFT, buff=0.2)
        max_label.next_to(track, RIGHT, buff=0.2)
        
        slider = VGroup(label, track, handle, value_text, min_label, max_label)
        slider.arrange_submobjects()
        
        return slider

    def create_result_area(self):
        """Create area for showing CSF results"""
        result_title = Text("Classification Results", font_size=32, color=YELLOW)
        result_title.to_edge(RIGHT, buff=2).shift(UP * 2)
        
        self.play(Write(result_title))
        
        # Result visualization area
        result_area = Rectangle(width=6, height=4, color=WHITE, fill_opacity=0.1)
        result_area.next_to(result_title, DOWN, buff=1)
        
        self.play(Create(result_area))
        self.result_area = result_area

    def demonstrate_parameter_effects(self):
        """Demonstrate how parameters affect results"""
        # Show different parameter combinations
        param_sets = [
            ("Conservative", {"resolution": 0.2, "threshold": 0.1}),
            ("Balanced", {"resolution": 0.5, "threshold": 0.25}),
            ("Aggressive", {"resolution": 1.0, "threshold": 0.5})
        ]
        
        for name, params in param_sets:
            # Update slider positions
            self.update_sliders(params)
            
            # Update result visualization
            self.update_results(params, name)
            
            self.wait(2)

    def update_sliders(self, params: Dict[str, float]):
        """Animate slider updates"""
        # This would animate the slider handles to new positions
        # Implementation would move handles and update value displays
        pass

    def update_results(self, params: Dict[str, float], config_name: str):
        """Update result visualization based on parameters"""
        # Clear previous results
        config_label = Text(f"Configuration: {config_name}", 
                          font_size=20, color=YELLOW)
        config_label.next_to(self.result_area, UP, buff=0.5)
        
        self.play(Write(config_label))
        
        # Generate sample results based on parameters
        points = self.generate_classification_result(params)
        
        self.play(FadeIn(points))
        self.wait(1)
        self.play(FadeOut(points, config_label))

    def generate_classification_result(self, params: Dict[str, float]) -> VGroup:
        """Generate sample classification results"""
        points = VGroup()
        
        # Generate points based on parameters
        threshold = params.get("threshold", 0.25)
        resolution = params.get("resolution", 0.5)
        
        # Simulate classification results
        for _ in range(50):
            x = np.random.uniform(-2.5, 2.5)
            y = np.random.uniform(-1.5, 1.5)
            
            # Classification probability based on threshold
            if threshold < 0.2:
                is_ground = np.random.random() < 0.4
            elif threshold > 0.4:
                is_ground = np.random.random() < 0.8
            else:
                is_ground = np.random.random() < 0.6
            
            color = BROWN if is_ground else RED
            point = Dot([x, y, 0], radius=0.03, color=color)
            points.add(point)
        
        return points


if __name__ == "__main__":
    # Example usage and execution instructions
    print("""
    CSF Algorithm Animation Setup
    =============================
    
    Prerequisites:
    1. Install Manim: pip install manim
    2. Install PDAL (optional): pip install pdal
    3. Install utilities: Place pdal_utils.py in same directory
    
    Basic Execution:
    1. Main animation (comprehensive):
       manim -pql csf_animation.py CSFAlgorithmAnimation
       
    2. Parameter exploration:
       manim -pql csf_animation.py CSFParameterExploration
    
    3. High quality output:
       manim -pqh csf_animation.py CSFAlgorithmAnimation
    
    Advanced Usage:
    - Use real LAS files by placing them in ./data/ directory
    - Modify parameters in the CSFPipelineManager class
    - Customize visualization by editing the Scene classes
    
    Output:
    - MP4 animations will be created in ./media/videos/
    - Use these for presentations, training, or documentation
    
    Educational Applications:
    - GIS and remote sensing courses
    - LiDAR processing workshops
    - PDAL training sessions
    - Algorithm explanation presentations
    """)
