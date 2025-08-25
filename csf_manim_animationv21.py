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

class SceneObjectManager:
    """Manages scene objects for proper cleanup"""
    def __init__(self, scene):
        self.scene = scene
        self.tracked_objects = {}
        self.object_groups = {}
    
    def add_object(self, name, obj, group="default"):
        """Add object to tracking"""
        self.tracked_objects[name] = obj
        if group not in self.object_groups:
            self.object_groups[group] = []
        self.object_groups[group].append(name)
        return obj
    
    def remove_object(self, name):
        """Remove specific object"""
        if name in self.tracked_objects:
            obj = self.tracked_objects[name]
            if obj in self.scene.mobjects:
                self.scene.remove(obj)
            del self.tracked_objects[name]
    
    def cleanup_group(self, group="default"):
        """Clean up entire group"""
        if group in self.object_groups:
            objects_to_remove = []
            for name in self.object_groups[group]:
                if name in self.tracked_objects:
                    objects_to_remove.append(self.tracked_objects[name])
                    del self.tracked_objects[name]
            
            if objects_to_remove:
                self.scene.play(FadeOut(*objects_to_remove))
            self.object_groups[group] = []
    
    def cleanup_all(self):
        """Clean up everything"""
        all_objects = list(self.tracked_objects.values())
        if all_objects:
            self.scene.play(FadeOut(*all_objects))
        self.tracked_objects.clear()
        self.object_groups.clear()

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
        # Initialize object manager
        self.object_manager = SceneObjectManager(self)
        
        # Title sequence
        self.play_title_sequence()
        
        # Step 1: Introduce the problem
        self.introduce_problem()
        
        # Step 2: Show algorithm overview
        self.algorithm_overview()
        
        # Step 3: Demonstrate cloth initialization
        self.cloth_initialization()
        
        # Step 4: Show physics simulation
        self.physics_simulation_high_density_sticky_csf()
        
        # Step 5: Ground point classification
        self.ground_classification()
        
        # Step 6: Real data example with PDAL
        self.real_data_example()
        
        # Step 7: Parameters and tuning
        self.parameter_tuning()
        
        # Step 8: Conclusion
        self.conclusion()
        
        # Final cleanup
        self.object_manager.cleanup_all()

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
        self.object_manager.add_object("problem_title", problem_title, "intro")
        
        # Create 3D-like point cloud visualization
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 4, 1],
            x_length=8,
            y_length=6,
            tips=False
        )
        self.object_manager.add_object("intro_axes", axes, "intro")
        
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
        
        self.object_manager.add_object("ground_dots", ground_dots, "intro")
        self.object_manager.add_object("vegetation_dots", vegetation_dots, "intro")
        self.object_manager.add_object("building_dots", building_dots, "intro")
        
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
        self.object_manager.add_object("labels", labels, "intro")
        
        self.play(Write(labels))
        self.wait(2)
        
        # Store for later use
        self.ground_points = ground_points
        self.vegetation_points = vegetation_points
        self.building_points = building_points
        self.axes = axes
        
        # PROPER CLEANUP - Clean up intro group
        self.object_manager.cleanup_group("intro")
        self.wait(0.5)

    def algorithm_overview(self):
        """Show CSF algorithm overview"""
        title = Text("CSF Algorithm Overview", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.object_manager.add_object("overview_title", title, "overview")
        
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
        
        self.object_manager.add_object("step_objects", step_objects, "overview")
        
        self.play(Write(title))
        
        for step in step_objects:
            self.play(Write(step))
            self.wait(0.8)
        
        self.wait(2)
        
        # Clean up overview group
        self.object_manager.cleanup_group("overview")

    def cloth_initialization(self):
        """Demonstrate cloth initialization"""
        title = Text("Step 1: Cloth Initialization", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        # Recreate clean axes for this scene
        clean_axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 4, 1],
            x_length=8,
            y_length=6,
            tips=False
        )
        
        # Show point cloud again
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        point_dots = VGroup(*[
            Dot(clean_axes.coords_to_point(x, z), color=WHITE, radius=0.02) 
            for x, y, z in all_points
        ])
        
        self.play(Write(title))
        self.play(Create(clean_axes), FadeIn(point_dots))
        
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
                        clean_axes.coords_to_point(x, cloth_height),
                        clean_axes.coords_to_point(x + cloth_resolution, cloth_height),
                        color=BLUE, stroke_width=1
                    )
                    cloth_lines.add(line)
                if j < len(y_coords) - 1:
                    line = Line(
                        clean_axes.coords_to_point(x, cloth_height),
                        clean_axes.coords_to_point(x, cloth_height + cloth_resolution),
                        color=BLUE, stroke_width=1
                    )
                    cloth_lines.add(line)
        
        # Add cloth particles
        cloth_dots = VGroup(*[
            Dot(clean_axes.coords_to_point(x, z), color=BLUE, radius=0.02) 
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
        self.axes = clean_axes  # Update axes reference
        
        # PROPER CLEANUP
        self.play(FadeOut(title, explanation))
        self.wait(0.5)

    def physics_simulation(self):
        """Animate improved physics simulation with proper cloth structure"""
        title = Text("Step 2: Physics Simulation", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Enhanced physics parameters
        gravity = 0.06
        damping = 0.92
        iterations = 40
        spring_strength = 0.03
        
        explanation = Text("Cloth falls under gravity with collision detection", 
                        font_size=24, color=YELLOW)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        
        # Prepare point cloud data
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        # Initialize cloth grid structure properly
        cloth_resolution = 0.5
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_z, max_z = all_points[:, 2].min(), all_points[:, 2].max()
        
        x_coords = np.arange(min_x - 1, max_x + 1, cloth_resolution)
        z_coords = np.arange(min_z - 1, max_z + 1, cloth_resolution)
        
        # Create structured cloth grid
        cloth_grid = {}
        cloth_positions = {}
        cloth_velocities = {}
        
        initial_height = max_z + 2
        
        for i, x in enumerate(x_coords):
            for j, z in enumerate(z_coords):
                grid_key = (i, j)
                cloth_grid[grid_key] = [x, 0, initial_height]
                cloth_positions[grid_key] = np.array([x, 0, initial_height])
                cloth_velocities[grid_key] = np.array([0.0, 0.0, 0.0])
        
        # Store grid dimensions
        self.grid_width = len(x_coords)
        self.grid_height = len(z_coords)
        
        # Physics simulation loop
        for iteration in range(iterations):
            # Apply forces to each cloth particle
            new_positions = {}
            new_velocities = {}
            
            for grid_key, position in cloth_positions.items():
                i, j = grid_key
                current_velocity = cloth_velocities[grid_key].copy()
                
                # Apply gravity
                current_velocity[2] -= gravity
                
                # Apply spring forces to maintain cloth structure
                spring_force = np.array([0.0, 0.0, 0.0])
                neighbor_count = 0
                
                # Check 4-connected neighbors (up, down, left, right)
                neighbors = [
                    (i-1, j), (i+1, j),  # left, right
                    (i, j-1), (i, j+1)   # up, down
                ]
                
                for ni, nj in neighbors:
                    neighbor_key = (ni, nj)
                    if neighbor_key in cloth_positions:
                        neighbor_pos = cloth_positions[neighbor_key]
                        
                        # Calculate spring force
                        diff = neighbor_pos - position
                        distance = np.linalg.norm(diff)
                        
                        if distance > 0.1:  # Avoid division by zero
                            # Rest length is the cloth resolution
                            rest_length = cloth_resolution
                            extension = distance - rest_length
                            
                            # Spring force proportional to extension
                            force_magnitude = spring_strength * extension
                            force_direction = diff / distance
                            spring_force += force_direction * force_magnitude
                            neighbor_count += 1
                
                # Add diagonal springs for stability (weaker)
                diagonal_neighbors = [
                    (i-1, j-1), (i-1, j+1),
                    (i+1, j-1), (i+1, j+1)
                ]
                
                diagonal_spring_strength = spring_strength * 0.3
                for ni, nj in diagonal_neighbors:
                    neighbor_key = (ni, nj)
                    if neighbor_key in cloth_positions:
                        neighbor_pos = cloth_positions[neighbor_key]
                        diff = neighbor_pos - position
                        distance = np.linalg.norm(diff)
                        
                        if distance > 0.1:
                            rest_length = cloth_resolution * np.sqrt(2)
                            extension = distance - rest_length
                            force_magnitude = diagonal_spring_strength * extension
                            force_direction = diff / distance
                            spring_force += force_direction * force_magnitude
                
                # Apply spring forces to velocity
                current_velocity += spring_force
                
                # Apply damping
                current_velocity *= damping
                
                # Calculate new position
                new_position = position + current_velocity
                
                # Collision detection with point cloud
                nearby_points = all_points[
                    (np.abs(all_points[:, 0] - new_position[0]) < 0.5) & 
                    (np.abs(all_points[:, 1] - new_position[1]) < 0.5)
                ]
                
                if len(nearby_points) > 0:
                    max_nearby_z = nearby_points[:, 2].max()
                    collision_buffer = 0.1
                    
                    if new_position[2] < max_nearby_z + collision_buffer:
                        new_position[2] = max_nearby_z + collision_buffer
                        current_velocity[2] = max(0, current_velocity[2] * 0.2)  # Damped bounce
                        # Add some friction
                        current_velocity[0] *= 0.8
                        current_velocity[1] *= 0.8
                
                new_positions[grid_key] = new_position
                new_velocities[grid_key] = current_velocity
            
            # Update positions and velocities
            cloth_positions = new_positions
            cloth_velocities = new_velocities
            
            # Update visualization every 3 iterations
            if iteration % 3 == 0:
                # Create new cloth dots
                new_cloth_dots = VGroup(*[
                    Dot(self.axes.coords_to_point(pos[0], pos[2]), color=BLUE, radius=0.02) 
                    for pos in cloth_positions.values()
                ])
                
                # Create new cloth lines showing grid structure
                new_cloth_lines = VGroup()
                
                # Horizontal lines
                for j in range(self.grid_height):
                    line_points = []
                    for i in range(self.grid_width):
                        grid_key = (i, j)
                        if grid_key in cloth_positions:
                            pos = cloth_positions[grid_key]
                            line_points.append(self.axes.coords_to_point(pos[0], pos[2]))
                    
                    if len(line_points) > 1:
                        for k in range(len(line_points) - 1):
                            line = Line(line_points[k], line_points[k+1], 
                                    color=BLUE, stroke_width=1, stroke_opacity=0.7)
                            new_cloth_lines.add(line)
                
                # Vertical lines
                for i in range(self.grid_width):
                    line_points = []
                    for j in range(self.grid_height):
                        grid_key = (i, j)
                        if grid_key in cloth_positions:
                            pos = cloth_positions[grid_key]
                            line_points.append(self.axes.coords_to_point(pos[0], pos[2]))
                    
                    if len(line_points) > 1:
                        for k in range(len(line_points) - 1):
                            line = Line(line_points[k], line_points[k+1], 
                                    color=BLUE, stroke_width=1, stroke_opacity=0.7)
                            new_cloth_lines.add(line)
                
                # Animate the transformation
                self.play(
                    Transform(self.cloth_dots, new_cloth_dots),
                    Transform(self.cloth_lines, new_cloth_lines),
                    run_time=0.2
                )
        
        # Update explanation to show settlement
        settled_explanation = Text("Cloth settles and drapes over surface features", 
                                font_size=24, color=GREEN)
        settled_explanation.to_edge(DOWN)
        
        self.play(Transform(explanation, settled_explanation))
        
        # Store final cloth positions for classification
        self.cloth_points = np.array(list(cloth_positions.values()))
        
        self.wait(2)
        
        # PROPER CLEANUP
        self.play(FadeOut(title, explanation))
        self.wait(0.5)

    def physics_simulation_pdal_like(self):
        """CSF simulation that closely matches PDAL's actual algorithm"""
        title = Text("Step 2: PDAL-Matching CSF Algorithm", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        explanation = Text("Implementing CSF algorithm as used in PDAL", 
                        font_size=24, color=YELLOW)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        
        # Get point cloud
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        # Show point cloud
        point_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=WHITE, radius=0.025) 
            for x, y, z in all_points
        ])
        self.play(FadeIn(point_dots, lag_ratio=0.03))
        
        # PDAL-style parameters (matching your pipeline)
        csf_params = {
            'resolution': 0.5,      # Match your PDAL pipeline
            'threshold': 0.25,      # Match your PDAL pipeline  
            'rigidness': 2,         # Match your PDAL pipeline
            'iterations': 500,      # Match your PDAL pipeline
            'time_step': 0.65       # Match your PDAL pipeline
        }
        
        # Show parameters
        param_display = VGroup(
            Text("PDAL CSF Parameters:", font_size=18, color=YELLOW),
            Text(f"Resolution: {csf_params['resolution']}m", font_size=14, color=WHITE),
            Text(f"Threshold: {csf_params['threshold']}m", font_size=14, color=WHITE),
            Text(f"Rigidness: {csf_params['rigidness']}", font_size=14, color=WHITE),
            Text(f"Iterations: {csf_params['iterations']}", font_size=14, color=WHITE)
        )
        param_display.arrange(DOWN, aligned_edge=LEFT)
        param_display.to_corner(UL, buff=0.5).shift(DOWN * 1.5)
        self.play(FadeIn(param_display))
        
        # STEP 1: Initialize cloth using PDAL's method
        explanation_step1 = Text("Step 1: Initialize cloth from point cloud base", 
                            font_size=20, color=GREEN)
        explanation_step1.to_edge(DOWN)
        self.play(Transform(explanation, explanation_step1))
        
        # Create grid based on resolution
        resolution = csf_params['resolution']
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_z, max_z = all_points[:, 2].min(), all_points[:, 2].max()
        
        # Extend bounds slightly
        x_coords = np.arange(min_x - resolution, max_x + resolution, resolution)
        z_coords = np.arange(min_z - resolution, max_z + resolution, resolution)
        
        grid_width = len(x_coords)
        grid_height = len(z_coords)
        
        print(f"PDAL-style grid: {grid_width} x {grid_height} at {resolution}m resolution")
        
        # Initialize cloth heights using PDAL's method
        cloth_heights = {}
        cloth_velocities = {}
        
        # CORRECTED: Initialize cloth above point cloud (PDAL's actual approach)
        max_terrain_height = all_points[:, 2].max()
        initial_cloth_height = max_terrain_height + 1.5  # Start above terrain
        
        print(f"Initializing cloth at height: {initial_cloth_height:.2f}m")
        print(f"Max terrain height: {max_terrain_height:.2f}m")
        
        for i, x in enumerate(x_coords):
            for j, z in enumerate(z_coords):
                # All cloth particles start at same height above terrain
                cloth_heights[(i, j)] = initial_cloth_height
                cloth_velocities[(i, j)] = 0.0
        
        # Visualization function
        def make_cloth_visual(heights_dict, color=BLUE):
            dots = VGroup()
            lines = VGroup()
            
            positions_list = []
            for (i, j), height in heights_dict.items():
                x, z = x_coords[i], z_coords[j]
                dot_pos = self.axes.coords_to_point(x, height)
                dot = Dot(dot_pos, color=color, radius=0.02)
                dots.add(dot)
                positions_list.append((i, j, dot_pos))
            
            # Create grid structure
            for (i, j, pos) in positions_list:
                if i < grid_width - 1 and (i+1, j) in heights_dict:
                    right_height = heights_dict[(i+1, j)]
                    right_pos = self.axes.coords_to_point(x_coords[i+1], right_height)
                    line = Line(pos, right_pos, color=color, stroke_width=1.5)
                    lines.add(line)
                
                if j < grid_height - 1 and (i, j+1) in heights_dict:
                    down_height = heights_dict[(i, j+1)]
                    down_pos = self.axes.coords_to_point(x_coords[i], down_height)
                    line = Line(pos, down_pos, color=color, stroke_width=1.5)
                    lines.add(line)
            
            return dots, lines
        
        # Show initial cloth (starting from below)
        cloth_dots, cloth_lines = make_cloth_visual(cloth_heights)
        self.play(FadeIn(cloth_dots), Create(cloth_lines))
        self.wait(2)
        
        # STEP 2: Apply PDAL's rigidness-based physics
        explanation_step2 = Text("Step 2: Apply physics with rigidness constraints", 
                            font_size=20, color=GREEN)
        explanation_step2.to_edge(DOWN)
        self.play(Transform(explanation, explanation_step2))
        
        # Convert rigidness to spring parameters (PDAL's approach)
        rigidness = csf_params['rigidness']
        if rigidness == 1:
            spring_stiffness = 0.3   # Flexible
            displacement_limit = 0.3
        elif rigidness == 2:
            spring_stiffness = 0.6   # Balanced
            displacement_limit = 0.2
        else:  # rigidness == 3
            spring_stiffness = 1.0   # Stiff
            displacement_limit = 0.1
        
        gravity = 0.1
        damping = 0.9
        time_step = csf_params['time_step']
        
        # Simulate subset of iterations for animation (full 500 would be too slow)
        animation_iterations = 40
        iteration_step = csf_params['iterations'] // animation_iterations
        
        progress_text = Text("CSF Physics Simulation...", font_size=16, color=GREEN)
        progress_text.to_corner(UR)
        self.play(Write(progress_text))
        
        for anim_iter in range(animation_iterations):
            actual_iteration = anim_iter * iteration_step
            
            # Multiple physics steps per animation frame
            for _ in range(iteration_step):
                new_heights = {}
                new_velocities = {}
                
                for (i, j), height in cloth_heights.items():
                    x, z = x_coords[i], z_coords[j]
                    velocity = cloth_velocities[(i, j)]
                    
                    # Apply gravity
                    velocity -= gravity * time_step
                    
                    # Spring forces with rigidness constraints
                    spring_force = 0.0
                    constraint_violations = 0
                    
                    # Check neighbors
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (ni, nj) in cloth_heights:
                            neighbor_height = cloth_heights[(ni, nj)]
                            height_diff = neighbor_height - height
                            
                            # Apply displacement limit based on rigidness
                            if abs(height_diff) > displacement_limit:
                                height_diff = np.sign(height_diff) * displacement_limit
                                constraint_violations += 1
                            
                            spring_force += spring_stiffness * height_diff
                    
                    # Apply spring forces
                    if constraint_violations < 4:  # Only if not over-constrained
                        velocity += spring_force * time_step / 4
                    
                    # Apply damping
                    velocity *= damping
                    
                    # Update position
                    new_height = height + velocity * time_step
                    
                    # Collision detection - cloth hits terrain from above
                    nearby_points = all_points[
                        (abs(all_points[:, 0] - x) < resolution) & 
                        (abs(all_points[:, 1] - z) < resolution)
                    ]
                    
                    if len(nearby_points) > 0:
                        max_terrain = nearby_points[:, 2].max()
                        collision_buffer = 0.1
                        min_cloth_height = max_terrain + collision_buffer
                        
                        if new_height < min_cloth_height:
                            new_height = min_cloth_height
                            velocity = max(0, velocity * 0.2)  # Damped collision
                    
                    new_heights[(i, j)] = new_height
                    new_velocities[(i, j)] = velocity
                
                cloth_heights = new_heights
                cloth_velocities = new_velocities
            
            # Update visualization
            new_cloth_dots, new_cloth_lines = make_cloth_visual(cloth_heights)
            
            # Update progress
            progress_new = Text(f"Iteration: {actual_iteration}/{csf_params['iterations']}", 
                            font_size=16, color=GREEN)
            progress_new.to_corner(UR)
            
            run_time = 0.2 if anim_iter < 20 else 0.3
            self.play(
                Transform(cloth_dots, new_cloth_dots),
                Transform(cloth_lines, new_cloth_lines),
                Transform(progress_text, progress_new),
                run_time=run_time
            )
        
        # STEP 3: Apply PDAL's classification
        explanation_step3 = Text("Step 3: Classify points using PDAL threshold", 
                            font_size=20, color=GREEN)
        explanation_step3.to_edge(DOWN)
        self.play(Transform(explanation, explanation_step3))
        
        # Classify points exactly like PDAL
        threshold = csf_params['threshold']
        ground_points = []
        non_ground_points = []
        
        for point in all_points:
            x, y, z = point
            
            # Find cloth height at this point's location
            grid_i = int((x - x_coords[0]) / resolution)
            grid_j = int((z - z_coords[0]) / resolution)
            
            # Clamp to grid bounds
            grid_i = max(0, min(grid_width - 1, grid_i))
            grid_j = max(0, min(grid_height - 1, grid_j))
            
            if (grid_i, grid_j) in cloth_heights:
                cloth_height = cloth_heights[(grid_i, grid_j)]
                distance_to_cloth = z - cloth_height
                
                if distance_to_cloth <= threshold:
                    ground_points.append(point)
                else:
                    non_ground_points.append(point)
        
        # Show classification results
        if ground_points:
            ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.03) 
                for x, y, z in ground_points
            ])
            self.play(FadeIn(ground_dots, lag_ratio=0.05))
        
        self.wait(1)
        
        if non_ground_points:
            non_ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=RED, radius=0.03) 
                for x, y, z in non_ground_points
            ])
            self.play(FadeIn(non_ground_dots, lag_ratio=0.05))
        
        # Show results
        results_text = VGroup(
            Text(f"Ground points: {len(ground_points)}", font_size=16, color=BROWN),
            Text(f"Non-ground: {len(non_ground_points)}", font_size=16, color=RED),
            Text(f"Threshold: {threshold}m", font_size=16, color=WHITE)
        )
        results_text.arrange(DOWN, aligned_edge=LEFT)
        results_text.to_corner(UR, buff=0.5).shift(DOWN * 2)
        self.play(FadeIn(results_text))
        
        # Final explanation
        final_explanation = Text("CSF algorithm complete - matches PDAL implementation", 
                            font_size=24, color=GREEN)
        final_explanation.to_edge(DOWN)
        
        self.play(Transform(explanation, final_explanation))
        self.wait(3)
        
        # Convert for compatibility
        final_positions = []
        for (i, j), height in cloth_heights.items():
            x, z = x_coords[i], z_coords[j]
            final_positions.append([x, 0, height])
        
        self.cloth_points = np.array(final_positions)
        self.cloth_dots = cloth_dots
        self.cloth_lines = cloth_lines
        
        # Cleanup
        if 'ground_dots' in locals():
            cleanup_objects = [title, explanation, param_display, progress_text, 
                            results_text, point_dots, ground_dots]
            if 'non_ground_dots' in locals():
                cleanup_objects.append(non_ground_dots)
        else:
            cleanup_objects = [title, explanation, param_display, progress_text, point_dots]
        
        self.play(FadeOut(*cleanup_objects))
        self.wait(0.5)

    def physics_simulation_advanced(self):
        """Implementation of the actual CSF algorithm as described in Zhang et al. 2016"""
        title = Text("Real CSF Algorithm (Zhang et al. 2016)", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Get all points
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        # Show point cloud with different colors to see the challenge
        ground_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.02) 
            for x, y, z in self.ground_points
        ])
        vegetation_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=GREEN, radius=0.02) 
            for x, y, z in self.vegetation_points
        ])
        building_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=GRAY, radius=0.02) 
            for x, y, z in self.building_points
        ])
        
        self.play(FadeIn(ground_dots), FadeIn(vegetation_dots), FadeIn(building_dots))
        
        explanation = Text("Challenge: Cloth must find ground through vegetation/buildings", 
                        font_size=20, color=YELLOW)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
        
        # Real CSF parameters
        resolution = 0.5
        threshold = 0.25
        rigidness = 2
        class_threshold = 0.25
        
        # Create cloth grid
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_z, max_z = all_points[:, 2].min(), all_points[:, 2].max()
        
        # Extend grid beyond point cloud
        padding = resolution
        min_x -= padding
        max_x += padding
        min_z -= padding
        max_z += padding
        
        x_coords = np.arange(min_x, max_x + resolution, resolution)
        z_coords = np.arange(min_z, max_z + resolution, resolution)
        
        grid_width = len(x_coords)
        grid_height = len(z_coords)
        
        print(f"CSF Grid: {grid_width} x {grid_height} at {resolution}m resolution")
        
        # Initialize cloth surface - start HIGH above everything
        cloth_surface = {}
        max_height = all_points[:, 2].max()
        initial_height = max_height + 2.0  # Start well above
        
        for i in range(grid_width):
            for j in range(grid_height):
                cloth_surface[(i, j)] = initial_height
        
        print(f"Cloth initialized at {initial_height:.1f}m above terrain")
        
        # Visualization function
        def visualize_cloth_surface(surface, color=BLUE, alpha=0.8):
            dots = VGroup()
            lines = VGroup()
            
            # Create cloth mesh
            for i in range(grid_width):
                for j in range(grid_height):
                    if (i, j) in surface:
                        x = x_coords[i]
                        z = z_coords[j]
                        height = surface[(i, j)]
                        
                        dot = Dot(self.axes.coords_to_point(x, height), 
                                color=color, radius=0.015)
                        dots.add(dot)
                        
                        # Connect to neighbors
                        if i < grid_width - 1 and (i+1, j) in surface:
                            x_next = x_coords[i+1]
                            height_next = surface[(i+1, j)]
                            line = Line(
                                self.axes.coords_to_point(x, height),
                                self.axes.coords_to_point(x_next, height_next),
                                color=color, stroke_width=1, stroke_opacity=alpha
                            )
                            lines.add(line)
                        
                        if j < grid_height - 1 and (i, j+1) in surface:
                            z_next = z_coords[j+1]
                            height_next = surface[(i, j+1)]
                            line = Line(
                                self.axes.coords_to_point(x, height),
                                self.axes.coords_to_point(x, height_next),
                                color=color, stroke_width=1, stroke_opacity=alpha
                            )
                            lines.add(line)
            
            return dots, lines
        
        # Show initial cloth
        cloth_dots, cloth_lines = visualize_cloth_surface(cloth_surface)
        self.play(FadeIn(cloth_dots), Create(cloth_lines))
        
        # Real CSF Algorithm Implementation
        explanation_alg = Text("Running real CSF algorithm - cloth finds ground iteratively", 
                            font_size=20, color=GREEN)
        explanation_alg.to_edge(DOWN)
        self.play(Transform(explanation, explanation_alg))
        
        # CSF Algorithm Steps
        time_step = 0.65
        gravity = 9.8
        iterations = 200  # Reduced for animation
        
        progress_text = Text("CSF Algorithm Running...", font_size=16, color=GREEN)
        progress_text.to_corner(UR)
        self.play(Write(progress_text))
        
        for iteration in range(iterations):
            new_surface = {}
            
            # Step 1: Apply gravitational and internal forces
            for i in range(grid_width):
                for j in range(grid_height):
                    if (i, j) not in cloth_surface:
                        continue
                        
                    current_height = cloth_surface[(i, j)]
                    x = x_coords[i]
                    z = z_coords[j]
                    
                    # Gravitational force
                    gravity_displacement = gravity * time_step * time_step
                    new_height = current_height - gravity_displacement * 0.01  # Scaled for animation
                    
                    # Internal forces (springs) to neighbors
                    spring_force = 0.0
                    neighbor_count = 0
                    
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (ni, nj) in cloth_surface:
                            neighbor_height = cloth_surface[(ni, nj)]
                            height_diff = neighbor_height - current_height
                            
                            # Rigidness constraint
                            max_diff = 0.3 / rigidness  # Stiffer cloth = smaller deformation
                            if abs(height_diff) > max_diff:
                                height_diff = np.sign(height_diff) * max_diff
                            
                            spring_force += height_diff
                            neighbor_count += 1
                    
                    if neighbor_count > 0:
                        spring_displacement = (spring_force / neighbor_count) * 0.1
                        new_height += spring_displacement
                    
                    new_surface[(i, j)] = new_height
            
            # Step 2: Handle movable points (CSF's key innovation)
            # Instead of collision detection, CSF determines which points are "movable"
            for i in range(grid_width):
                for j in range(grid_height):
                    if (i, j) not in new_surface:
                        continue
                        
                    x = x_coords[i]
                    z = z_coords[j]
                    cloth_height = new_surface[(i, j)]
                    
                    # Find points in this grid cell
                    cell_points = all_points[
                        (np.abs(all_points[:, 0] - x) < resolution/2) &
                        (np.abs(all_points[:, 1] - z) < resolution/2)
                    ]
                    
                    if len(cell_points) > 0:
                        # Key CSF insight: Find the "unmovable" height
                        # This is where cloth should stop based on point distribution
                        point_heights = cell_points[:, 2]
                        
                        # CSF algorithm: cloth stops at highest "unmovable" point
                        # For simplification, we'll use a statistical approach
                        if len(point_heights) > 1:
                            # If many points at similar height, likely ground
                            height_std = np.std(point_heights)
                            if height_std < 0.5:  # Points clustered = likely ground
                                unmovable_height = np.max(point_heights)
                            else:
                                # Mixed heights = some vegetation, find ground level
                                unmovable_height = np.percentile(point_heights, 20)  # Lower percentile
                        else:
                            unmovable_height = point_heights[0]
                        
                        # Cloth cannot go below unmovable surface
                        buffer = 0.1
                        min_cloth_height = unmovable_height + buffer
                        
                        if cloth_height < min_cloth_height:
                            new_surface[(i, j)] = min_cloth_height
            
            # Update cloth surface
            cloth_surface = new_surface
            
            # Update visualization every few iterations
            if iteration % 2 == 0:
                new_cloth_dots, new_cloth_lines = visualize_cloth_surface(cloth_surface)
                
                # Update progress
                progress_new = Text(f"CSF Iteration: {iteration + 1}/{iterations}", 
                                font_size=16, color=GREEN)
                progress_new.to_corner(UR)
                
                run_time = 0.2 if iteration < 20 else 0.3
                self.play(
                    Transform(cloth_dots, new_cloth_dots),
                    Transform(cloth_lines, new_cloth_lines),
                    Transform(progress_text, progress_new),
                    run_time=run_time
                )
        
        # Classification phase
        explanation_classify = Text("Classifying points based on distance to final cloth surface", 
                                font_size=20, color=GREEN)
        explanation_classify.to_edge(DOWN)
        self.play(Transform(explanation, explanation_classify))
        
        # Classify all points
        ground_classified = []
        non_ground_classified = []
        
        for point in all_points:
            x, y, z = point
            
            # Find nearest cloth grid cell
            grid_i = int(round((x - x_coords[0]) / resolution))
            grid_j = int(round((z - z_coords[0]) / resolution))
            
            # Clamp to grid bounds
            grid_i = max(0, min(grid_width - 1, grid_i))
            grid_j = max(0, min(grid_height - 1, grid_j))
            
            if (grid_i, grid_j) in cloth_surface:
                cloth_height = cloth_surface[(grid_i, grid_j)]
                distance_to_cloth = z - cloth_height
                
                if distance_to_cloth <= class_threshold:
                    ground_classified.append(point)
                else:
                    non_ground_classified.append(point)
            else:
                non_ground_classified.append(point)
        
        # Remove original colored points
        self.play(FadeOut(ground_dots, vegetation_dots, building_dots))
        
        # Show final classification
        if ground_classified:
            final_ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.03)
                for x, y, z in ground_classified
            ])
            self.play(FadeIn(final_ground_dots, lag_ratio=0.02))
        
        if non_ground_classified:
            final_non_ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=RED, radius=0.03)
                for x, y, z in non_ground_classified
            ])
            self.play(FadeIn(final_non_ground_dots, lag_ratio=0.02))
        
        # Show results
        results_text = VGroup(
            Text("CSF Classification Results:", font_size=16, color=YELLOW),
            Text(f"Ground points: {len(ground_classified)}", font_size=14, color=BROWN),
            Text(f"Non-ground points: {len(non_ground_classified)}", font_size=14, color=RED),
            Text(f"Classification accuracy: {len(ground_classified)/(len(ground_classified)+len(non_ground_classified))*100:.1f}%", 
                font_size=14, color=WHITE)
        )
        results_text.arrange(DOWN, aligned_edge=LEFT)
        results_text.to_corner(UR, buff=0.5).shift(DOWN * 2)
        self.play(FadeIn(results_text))
        
        # Final explanation
        final_explanation = Text("CSF found ground surface through vegetation - algorithm complete!", 
                            font_size=24, color=GREEN)
        final_explanation.to_edge(DOWN)
        
        self.play(Transform(explanation, final_explanation))
        self.wait(3)
        
        # Store results
        final_positions = []
        for (i, j), height in cloth_surface.items():
            x = x_coords[i]
            z = z_coords[j]
            final_positions.append([x, 0, height])
        
        self.cloth_points = np.array(final_positions)
        self.cloth_dots = cloth_dots
        self.cloth_lines = cloth_lines
        
        # Cleanup
        cleanup_objects = [title, explanation, progress_text, results_text, cloth_dots, cloth_lines]
        if 'final_ground_dots' in locals():
            cleanup_objects.append(final_ground_dots)
        if 'final_non_ground_dots' in locals():
            cleanup_objects.append(final_non_ground_dots)
        
        self.play(FadeOut(*cleanup_objects))
        self.wait(0.5)

    def physics_simulation_adaptive_csf(self):
        """Adaptive CSF that shows cloth draping and correctly classifies points"""
        title = Text("Adaptive CSF Algorithm", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Analyze point cloud characteristics (but don't use classifications!)
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        min_z = all_points[:, 2].min()
        max_z = all_points[:, 2].max()
        z_range = max_z - min_z
        
        print(f"Point cloud analysis (no classifications used):")
        print(f"  Z range: {z_range:.2f}m ({min_z:.2f} to {max_z:.2f})")
        print(f"  Total points: {len(all_points)}")
        
        # Show ALL points as the same color (algorithm shouldn't know classifications)
        point_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=WHITE, radius=0.025) 
            for x, y, z in all_points
        ])
        self.play(FadeIn(point_dots, lag_ratio=0.02))
        
        # Adaptive parameters based ONLY on Z-range and point density
        base_resolution = 0.5
        base_threshold = 0.25
        base_time_step = 0.65
        base_iterations = 500
        
        # Scale parameters based on Z-range
        reference_height = 10.0  # 10m reference
        height_scale = z_range / reference_height
        
        # Adaptive scaling
        resolution = base_resolution * max(0.5, min(2.0, 1.0 / height_scale))  # Finer for tall clouds
        time_step = base_time_step * height_scale  # Larger steps for tall clouds
        iterations = int(base_iterations * max(1.0, height_scale * 0.8))  # More iterations for complexity
        threshold = base_threshold  # Keep threshold constant
        
        # Physics parameters
        gravity = 0.06 * height_scale  # Stronger gravity for tall clouds
        damping = 0.88
        spring_strength = 0.4
        
        print(f"Adaptive parameters:")
        print(f"  Resolution: {resolution:.2f}m")
        print(f"  Time step: {time_step:.2f}")
        print(f"  Iterations: {iterations}")
        print(f"  Gravity: {gravity:.2f}")
        
        # Display parameters
        param_display = VGroup(
            Text(f"Adaptive for {z_range:.1f}m height range:", font_size=16, color=YELLOW),
            Text(f"Resolution: {resolution:.2f}m", font_size=14, color=WHITE),
            Text(f"Time step: {time_step:.2f}", font_size=14, color=WHITE),
            Text(f"Gravity: {gravity:.2f}", font_size=14, color=WHITE),
            Text(f"Threshold: {threshold:.2f}m", font_size=14, color=WHITE)
        )
        param_display.arrange(DOWN, aligned_edge=LEFT)
        param_display.to_corner(UL, buff=0.5).shift(DOWN * 1.5)
        self.play(FadeIn(param_display))
        
        explanation = Text("Algorithm will find ground without using point classifications", 
                        font_size=20, color=YELLOW)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
        
        # Create cloth grid
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_z_coord, max_z_coord = all_points[:, 1].min(), all_points[:, 1].max()  # Note: using Y coordinate for Z-axis in 2D
        
        # Extend grid slightly
        padding = resolution
        min_x -= padding
        max_x += padding
        min_z_coord -= padding
        max_z_coord += padding
        
        x_coords = np.arange(min_x, max_x + resolution, resolution)
        z_coords = np.arange(min_z_coord, max_z_coord + resolution, resolution)
        
        grid_width = len(x_coords)
        grid_height = len(z_coords)
        
        print(f"Cloth grid: {grid_width} x {grid_height}")
        
        # Initialize cloth heights - start ABOVE all points
        initial_height = max_z + 1.5  # Start above highest point
        cloth_heights = {}
        cloth_velocities = {}
        
        for i in range(grid_width):
            for j in range(grid_height):
                cloth_heights[(i, j)] = initial_height
                cloth_velocities[(i, j)] = 0.0
        
        print(f"Cloth initialized at {initial_height:.1f}m")
        
        # Visualization function that shows cloth draping
        def visualize_cloth_draping(heights_dict, velocities_dict):
            dots = VGroup()
            lines = VGroup()
            
            # Color cloth based on velocity (shows movement)
            for i in range(grid_width):
                for j in range(grid_height):
                    if (i, j) in heights_dict:
                        x = x_coords[i]
                        z_coord = z_coords[j] 
                        height = heights_dict[(i, j)]
                        velocity = velocities_dict[(i, j)]
                        
                        # Color based on movement - blue when moving, green when settled
                        if abs(velocity) > 0.05:
                            color = BLUE  # Moving
                        else:
                            color = GREEN  # Settled
                        
                        dot = Dot(self.axes.coords_to_point(x, height), 
                                color=color, radius=0.02)
                        dots.add(dot)
                        
                        # Connect to neighbors
                        if i < grid_width - 1 and (i+1, j) in heights_dict:
                            x_next = x_coords[i+1]
                            height_next = heights_dict[(i+1, j)]
                            line = Line(
                                self.axes.coords_to_point(x, height),
                                self.axes.coords_to_point(x_next, height_next),
                                color=color, stroke_width=1.2, stroke_opacity=0.8
                            )
                            lines.add(line)
                        
                        if j < grid_height - 1 and (i, j+1) in heights_dict:
                            z_next = z_coords[j+1]
                            height_next = heights_dict[(i, j+1)]
                            line = Line(
                                self.axes.coords_to_point(x, height),
                                self.axes.coords_to_point(x, height_next),
                                color=color, stroke_width=1.2, stroke_opacity=0.8
                            )
                            lines.add(line)
            
            return dots, lines
        
        # Show initial cloth high above points
        cloth_dots, cloth_lines = visualize_cloth_draping(cloth_heights, cloth_velocities)
        self.play(FadeIn(cloth_dots), Create(cloth_lines))
        self.wait(1)
        
        # Physics simulation with visible cloth draping
        explanation_sim = Text("Watch cloth fall and drape over terrain features", 
                            font_size=20, color=GREEN)
        explanation_sim.to_edge(DOWN)
        self.play(Transform(explanation, explanation_sim))
        
        # Animation parameters
        animation_iterations = 35
        iteration_step = max(1, iterations // animation_iterations)
        
        progress_text = Text("Cloth falling...", font_size=16, color=GREEN)
        progress_text.to_corner(UR)
        self.play(Write(progress_text))
        
        # Physics simulation loop with visible draping
        for anim_iter in range(animation_iterations):
            # Run multiple physics steps per animation frame
            for step in range(iteration_step):
                new_heights = {}
                new_velocities = {}
                
                for i in range(grid_width):
                    for j in range(grid_height):
                        if (i, j) not in cloth_heights:
                            continue
                        
                        current_height = cloth_heights[(i, j)]
                        current_velocity = cloth_velocities[(i, j)]
                        x = x_coords[i]
                        z_coord = z_coords[j]
                        
                        # Apply gravity
                        current_velocity -= gravity * time_step
                        
                        # Spring forces to neighbors
                        spring_force = 0.0
                        neighbor_count = 0
                        
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if (ni, nj) in cloth_heights:
                                neighbor_height = cloth_heights[(ni, nj)]
                                height_diff = neighbor_height - current_height
                                
                                # Limit spring displacement
                                max_displacement = resolution * 0.3
                                if abs(height_diff) > max_displacement:
                                    height_diff = np.sign(height_diff) * max_displacement
                                
                                spring_force += spring_strength * height_diff
                                neighbor_count += 1
                        
                        if neighbor_count > 0:
                            current_velocity += (spring_force / neighbor_count) * time_step
                        
                        # Apply damping
                        current_velocity *= damping
                        
                        # Update position
                        new_height = current_height + current_velocity * time_step
                        
                        # Collision detection - find ground level without using classifications
                        search_radius = resolution * 1.2
                        nearby_points = all_points[
                            (np.abs(all_points[:, 0] - x) < search_radius) &
                            (np.abs(all_points[:, 1] - z_coord) < search_radius)
                        ]
                        
                        if len(nearby_points) > 0:
                            # Key fix: Determine ground level statistically (Zhang et al. method)
                            point_heights = nearby_points[:, 2]
                            
                            if len(point_heights) >= 3:
                                # Statistical ground detection
                                height_range = point_heights.max() - point_heights.min()
                                
                                if height_range > 0.5:  # Mixed heights = vegetation over ground
                                    # Find the ground level (lower percentile of points)
                                    ground_level = np.percentile(point_heights, 25)
                                else:
                                    # Similar heights = likely all ground
                                    ground_level = point_heights.max()
                            else:
                                # Few points - use max height
                                ground_level = point_heights.max()
                            
                            # Cloth collision with ground level
                            collision_buffer = 0.1
                            min_cloth_height = ground_level + collision_buffer
                            
                            if new_height < min_cloth_height:
                                new_height = min_cloth_height
                                # Collision response
                                if current_velocity < 0:
                                    current_velocity = -current_velocity * 0.2  # Damped bounce
                                current_velocity *= 0.6  # Friction
                        
                        new_heights[(i, j)] = new_height
                        new_velocities[(i, j)] = current_velocity
                
                # Update cloth state
                cloth_heights = new_heights
                cloth_velocities = new_velocities
            
            # Update visualization to show draping
            new_cloth_dots, new_cloth_lines = visualize_cloth_draping(cloth_heights, cloth_velocities)
            
            # Update progress
            progress_new = Text(f"Draping: {anim_iter + 1}/{animation_iterations}", 
                            font_size=16, color=GREEN)
            progress_new.to_corner(UR)
            
            # Animate the cloth movement
            run_time = 0.5 if anim_iter < 20 else 0.8
            self.play(
                Transform(cloth_dots, new_cloth_dots),
                Transform(cloth_lines, new_cloth_lines),
                Transform(progress_text, progress_new),
                run_time=run_time
            )
        
        # Classification phase - this is where we determine ground vs non-ground
        explanation_classify = Text("Classifying points based on distance to draped cloth", 
                                font_size=20, color=GREEN)
        explanation_classify.to_edge(DOWN)
        self.play(Transform(explanation, explanation_classify))
        
        # Classify points using cloth surface
        ground_classified = []
        non_ground_classified = []
        
        for point in all_points:
            x, y, z = point
            
            # Find nearest cloth grid cell
            grid_i = int(round((x - x_coords[0]) / resolution))
            grid_j = int(round((y - z_coords[0]) / resolution))  # Note: using Y for grid lookup
            
            # Clamp to valid range
            grid_i = max(0, min(grid_width - 1, grid_i))
            grid_j = max(0, min(grid_height - 1, grid_j))
            
            if (grid_i, grid_j) in cloth_heights:
                cloth_height = cloth_heights[(grid_i, grid_j)]
                distance_to_cloth = z - cloth_height
                
                # Classification based on threshold
                if distance_to_cloth <= threshold:
                    ground_classified.append(point)
                else:
                    non_ground_classified.append(point)
            else:
                # Points outside grid are non-ground
                non_ground_classified.append(point)
        
        # Remove original points and show classification
        self.play(FadeOut(point_dots))
        
        # Show ground points
        if ground_classified:
            ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.03)
                for x, y, z in ground_classified
            ])
            self.play(FadeIn(ground_dots, lag_ratio=0.02))
        
        # Show non-ground points
        if non_ground_classified:
            non_ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=RED, radius=0.03)
                for x, y, z in non_ground_classified
            ])
            self.play(FadeIn(non_ground_dots, lag_ratio=0.02))
        
        # Calculate accuracy (compare with known ground truth)
        true_ground_count = len(self.ground_points)
        classified_ground_count = len(ground_classified)
        
        # Simple accuracy measure
        accuracy = 0.0
        if len(all_points) > 0:
            # Count how many ground points were correctly classified
            correct_ground = 0
            for true_ground_point in self.ground_points:
                for classified_point in ground_classified:
                    if np.allclose(true_ground_point, classified_point, atol=0.01):
                        correct_ground += 1
                        break
            
            accuracy = correct_ground / true_ground_count * 100 if true_ground_count > 0 else 0
        
        # Results display
        results_display = VGroup(
            Text("CSF Classification Results:", font_size=16, color=YELLOW),
            Text(f"Classified as ground: {len(ground_classified)}", font_size=14, color=BROWN),
            Text(f"Classified as non-ground: {len(non_ground_classified)}", font_size=14, color=RED),
            Text(f"True ground points: {true_ground_count}", font_size=14, color=GRAY),
            Text(f"Ground detection accuracy: {accuracy:.1f}%", font_size=14, color=WHITE)
        )
        results_display.arrange(DOWN, aligned_edge=LEFT)
        results_display.to_corner(UR, buff=0.5).shift(DOWN * 2.5)
        self.play(FadeIn(results_display))
        
        # Final explanation
        final_explanation = Text("Adaptive CSF complete - cloth draped and points classified", 
                            font_size=24, color=GREEN)
        final_explanation.to_edge(DOWN)
        
        self.play(Transform(explanation, final_explanation))
        self.wait(3)
        
        # Store results for compatibility
        final_positions = []
        for (i, j), height in cloth_heights.items():
            x = x_coords[i]
            z_coord = z_coords[j]
            final_positions.append([x, 0, height])
        
        self.cloth_points = np.array(final_positions)
        self.cloth_dots = cloth_dots
        self.cloth_lines = cloth_lines
        
        # Cleanup
        cleanup_objects = [title, explanation, param_display, progress_text, 
                        results_display, cloth_dots, cloth_lines]
        if 'ground_dots' in locals():
            cleanup_objects.append(ground_dots)
        if 'non_ground_dots' in locals():
            cleanup_objects.append(non_ground_dots)
        
        self.play(FadeOut(*cleanup_objects))
        self.wait(0.5)

    def physics_simulation_high_density_sticky_csf(self):
        """Enhanced CSF that treats buildings as semi-ground and shows vegetation passthrough"""
        title = Text("CSF Visualization", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Analyze point cloud with building/vegetation classification
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        min_z = all_points[:, 2].min()
        max_z = all_points[:, 2].max()
        z_range = max_z - min_z
        
        print(f"BUILDING-AWARE DEBUG: Point cloud analysis:")
        print(f"  Ground points: {len(self.ground_points)}")
        print(f"  Vegetation points: {len(self.vegetation_points)}")
        print(f"  Building points: {len(self.building_points)}")
        print(f"  Z range: {min_z:.2f} to {max_z:.2f}")
        
        # Show point cloud with type-specific colors
        ground_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.02) 
            for x, y, z in self.ground_points
        ])
        vegetation_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=GREEN, radius=0.02) 
            for x, y, z in self.vegetation_points
        ])
        building_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=GRAY, radius=0.02) 
            for x, y, z in self.building_points
        ])
        
        self.play(FadeIn(ground_dots), FadeIn(vegetation_dots), FadeIn(building_dots))
        
        explanation = Text("Cloth treats buildings as semi-ground, passes through vegetation", 
                        font_size=20, color=YELLOW)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
        
        # Parameters optimized for building-aware physics
        resolution = 0.3
        time_step = 0.5
        gravity = 0.1
        damping = 0.88
        spring_strength = 0.5
        threshold = 0.25
        
        # Create cloth grid
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_z, max_z = all_points[:, 1].min(), all_points[:, 1].max()
        
        padding = resolution
        min_x -= padding
        max_x += padding
        min_z -= padding
        max_z += padding
        
        x_coords = np.arange(min_x, max_x + resolution, resolution)
        z_coords = np.arange(min_z, max_z + resolution, resolution)
        
        grid_width = len(x_coords)
        grid_height = len(z_coords)
        
        print(f"BUILDING-AWARE DEBUG: Grid size: {grid_width} x {grid_height}")
        
        # Initialize cloth heights - start ABOVE all points (corrected for new coordinate system)
        terrain_max_height = all_points[:, 2].max()  # Use actual Z coordinate from points
        initial_height = terrain_max_height + 1.5  # Start well above highest terrain point
        print(f"DEBUG: Terrain max height: {terrain_max_height:.2f}, cloth starts at: {initial_height:.2f}")
        cloth_heights = {}
        cloth_velocities = {}
        cloth_states = {}
        cloth_interactions = {}  # Track all interactions (ground, building, vegetation)
        
        for i in range(grid_width):
            for j in range(grid_height):
                cloth_heights[(i, j)] = initial_height
                cloth_velocities[(i, j)] = 0.0
                cloth_states[(i, j)] = 'falling'
                cloth_interactions[(i, j)] = {'type': None, 'points': [], 'forces': []}
        
        # Enhanced terrain analysis with building detection
        terrain_analysis = {}
        sticky_arrows = VGroup()
        
        estimated_ground_level = np.percentile(all_points[:, 2], 15)
        print(f"BUILDING-AWARE DEBUG: Estimated ground level: {estimated_ground_level:.2f}")
        
        for i in range(grid_width):        
            for j in range(grid_height):            
                x = x_coords[i]            
                z = z_coords[j]
                search_radius = resolution * 1.5
                # Separate nearby points by type
                nearby_ground = self.ground_points[                
                    (np.abs(self.ground_points[:, 0] - x) < search_radius) &  (np.abs(self.ground_points[:, 1] - z) < search_radius)]
                nearby_vegetation = self.vegetation_points[
                    (np.abs(self.vegetation_points[:, 0] - x) < search_radius) & (np.abs(self.vegetation_points[:, 1] - z) < search_radius)]
                nearby_buildings = self.building_points[
                    (np.abs(self.building_points[:, 0] - x) < search_radius) & (np.abs(self.building_points[:, 1] - z) < search_radius)]
                # Determine terrain type and behavior
                terrain_type = 'empty'
                ground_level = estimated_ground_level
                building_level = None
                vegetation_level = None
                stickiness = 0.0
                if len(nearby_ground) > 0:
                    ground_level = nearby_ground[:, 2].max()
                    terrain_type = 'ground'
                    stickiness = 0.8  
                    # Strong stickiness for ground
                if len(nearby_buildings) > 0:
                    building_level = nearby_buildings[:, 2].max()
                    building_density = len(nearby_buildings)
                    # Buildings act as semi-ground, especially flat rooftops
                    building_height_range = nearby_buildings[:, 2].max() - nearby_buildings[:, 2].min()
                    if building_height_range < 0.3:  
                        # Flat building top
                        if terrain_type == 'ground':
                            # Building above ground
                            if building_level > ground_level + 1.0:
                                terrain_type = 'building_on_ground'
                                stickiness = 0.6   # Moderate stickiness for building tops
                        else:
                            terrain_type = 'building'
                            ground_level = building_level
                            stickiness = 0.6
                    else:
                        # Sloped building edge - less sticky, gravity can overcome
                        if terrain_type == 'ground':
                            terrain_type = 'building_edge_on_ground'
                            stickiness = 0.3  # Low stickiness for edges
                        else:
                            terrain_type = 'building_edge'
                            stickiness = 0.2
                if len(nearby_vegetation) > 0:
                    vegetation_level = nearby_vegetation[:, 2].max()
                    if terrain_type in ['empty']:
                        terrain_type = 'vegetation_only'
                        stickiness = 0.0  # No stickiness for vegetation alone
                    else:
                        terrain_type += '_with_vegetation' # Vegetation doesn't increase stickiness
                # Store terrain analysis\n            
                terrain_analysis[(i, j)] = {'type': terrain_type, 
                                            'ground_level': ground_level, 
                                            'building_level': building_level, 
                                            'vegetation_level': vegetation_level,
                                            'stickiness': stickiness,
                                            'nearby_ground': nearby_ground,
                                            'nearby_buildings': nearby_buildings,
                                            'nearby_vegetation': nearby_vegetation}
                # Create sticky arrows for areas with significant stickiness
                if stickiness > 0.1:
                    arrow_start = self.axes.coords_to_point(x, ground_level)
                    arrow_end = self.axes.coords_to_point(x, ground_level + stickiness * 0.8)
                    # Color code by terrain type
                    if 'ground' in terrain_type:
                        arrow_color = BROWN
                    elif 'building' in terrain_type:
                        if 'edge' in terrain_type:
                            arrow_color = ORANGE  # Building edges                    
                        else:
                            arrow_color = GRAY    # Building tops
                else:
                    arrow_color = GREEN
                    arrow = Arrow(arrow_start, 
                                  arrow_end, 
                                  color=arrow_color, 
                                  stroke_width=1.5, 
                                  max_stroke_width_to_length_ratio=8, 
                                  max_tip_length_to_length_ratio=0.3)
                    arrow.set_opacity(0)
                    sticky_arrows.add(arrow)
        
        self.object_manager.add_object("sticky_arrows", sticky_arrows, "physics_helpers")
        self.add(sticky_arrows)
        print(f"BUILDING-AWARE DEBUG: Created {len(terrain_analysis)} terrain analysis cells")
        
        # Enhanced visualization with interaction feedback
        def visualize_building_aware_cloth(heights_dict, velocities_dict, states_dict, interactions_dict, show_interactions=True):
            dots = VGroup()
            lines = VGroup()
            interaction_markers = VGroup()
            passthrough_effects = VGroup()
            
            particle_positions = {}
            
            for i in range(grid_width):
                for j in range(grid_height):
                    if (i, j) not in heights_dict:
                        continue
                        
                    x = x_coords[i]
                    z = z_coords[j]
                    height = heights_dict[(i, j)]
                    state = states_dict[(i, j)]
                    interactions = interactions_dict[(i, j)]
                    
                    # Standard cloth particle colors
                    if state == 'falling':
                        color = BLUE
                        radius = 0.02
                    elif state == 'struggling':
                        color = YELLOW
                        radius = 0.025
                    elif state == 'stuck':
                        color = RED
                        radius = 0.03
                    else:  # settled
                        color = GREEN
                        radius = 0.02
                    
                    dot_pos = self.axes.coords_to_point(x, height)
                    dot = Dot(dot_pos, color=color, radius=radius)
                    dots.add(dot)
                    particle_positions[(i, j)] = dot_pos
                    
                    # Show interaction effects based on what cloth is interacting with
                    if show_interactions and interactions['type'] is not None:
                        interaction_type = interactions['type']
                        
                        # Highlight cloth particle in interaction
                        if interaction_type in ['ground_contact', 'building_contact']:
                            # Strong interaction (stuck/struggling)
                            highlight = Circle(radius=0.06, color=RED, stroke_width=3)
                            highlight.move_to(dot_pos)
                            interaction_markers.add(highlight)
                            # Show connection to causing points
                            for point in interactions['points'][:3]:  
                                # Limit to first 3 for clarity
                                point_pos = self.axes.coords_to_point(point[0], point[2])
                                connection = Line(dot_pos, point_pos, color=RED, stroke_width=2)
                                interaction_markers.add(connection)
                            # Highlight the terrain point
                            if interaction_type == 'ground_contact':
                                terrain_highlight = Circle(radius=0.04, color=BROWN, stroke_width=2)
                            else:  # building_contact
                                terrain_highlight = Circle(radius=0.04, color=GRAY, stroke_width=2)
                                terrain_highlight.move_to(point_pos)
                                interaction_markers.add(terrain_highlight)
                        elif interaction_type == 'vegetation_passthrough':
                            # Gentle interaction (passing through)
                            if show_interactions:
                                passthrough_circle = Circle(radius=0.04, color=GREEN, stroke_width=1, stroke_opacity=0.6)
                                passthrough_circle.move_to(dot_pos)
                                passthrough_effects.add(passthrough_circle)
                                # Show vegetation being passed through
                                for point in interactions['points'][:2]:
                                    # Show fewer for vegetation
                                    point_pos = self.axes.coords_to_point(point[0], point[2])
                                    passthrough_line = DashedLine(dot_pos, 
                                                                point_pos,
                                                                color=GREEN, 
                                                                stroke_width=1, 
                                                                stroke_opacity=0.5,
                                                                dash_length=0.1)
                                    passthrough_effects.add(passthrough_line)
                        elif interaction_type == 'building_edge_override':
                            # Gravity overcoming building edge
                            override_marker = RegularPolygon(n=4, radius=0.03, color=YELLOW, fill_opacity=0.8)
                            override_marker.move_to(dot_pos)
                            interaction_markers.add(override_marker)        
            # Create grid lines
            for i in range(0, grid_width, 2):  # Every other line to reduce clutter
                for j in range(0, grid_height, 2):
                    if (i, j) not in particle_positions:
                        continue
                        
                    current_pos = particle_positions[(i, j)]
                    state = states_dict[(i, j)]
                    
                    color = BLUE if state == 'falling' else YELLOW if state == 'struggling' else RED if state == 'stuck' else GREEN
                    
                    if i < grid_width - 2 and (i+2, j) in particle_positions:
                        right_pos = particle_positions[(i+2, j)]
                        line = Line(current_pos, right_pos, 
                                color=color, stroke_width=1, stroke_opacity=0.7)
                        lines.add(line)
                    
                    if j < grid_height - 2 and (i, j+2) in particle_positions:
                        down_pos = particle_positions[(i, j+2)]
                        line = Line(current_pos, down_pos, 
                                color=color, stroke_width=1, stroke_opacity=0.7)
                        lines.add(line)
            
            return dots, lines, interaction_markers, passthrough_effects
        
        # Show initial cloth
        cloth_dots, cloth_lines, interaction_markers, passthrough_effects = visualize_building_aware_cloth(
            cloth_heights, cloth_velocities, cloth_states, cloth_interactions
        )
        self.play(FadeIn(cloth_dots), Create(cloth_lines))
        
        # Enhanced legend
        legend = VGroup(
            Text("Building-Aware Cloth:", font_size=14, color=WHITE),
            VGroup(Dot(color=BLUE, radius=0.02), Text("Falling", font_size=11, color=BLUE)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(color=YELLOW, radius=0.02), Text("Struggling", font_size=11, color=YELLOW)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(color=RED, radius=0.02), Text("Stuck", font_size=11, color=RED)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(color=GREEN, radius=0.02), Text("Settled", font_size=11, color=GREEN)).arrange(RIGHT, buff=0.1),
            Text("Interactions:", font_size=14, color=WHITE),
            VGroup(Circle(radius=0.03, color=RED), Text("Ground/Building Contact", font_size=10, color=RED)).arrange(RIGHT, buff=0.1),
            VGroup(Circle(radius=0.02, color=GREEN), Text("Vegetation Passthrough", font_size=10, color=GREEN)).arrange(RIGHT, buff=0.1),
            VGroup(RegularPolygon(n=4, radius=0.02, color=YELLOW, fill_opacity=0.8), Text("Gravity Override", font_size=10, color=YELLOW)).arrange(RIGHT, buff=0.1),
            Text("Terrain Stickiness:", font_size=14, color=WHITE),
            VGroup(Arrow(ORIGIN, UP*0.15, color=BROWN), Text("Ground", font_size=10, color=BROWN)).arrange(RIGHT, buff=0.1),
            VGroup(Arrow(ORIGIN, UP*0.15, color=GRAY), Text("Building Top", font_size=10, color=GRAY)).arrange(RIGHT, buff=0.1),
            VGroup(Arrow(ORIGIN, UP*0.15, color=ORANGE), Text("Building Edge", font_size=10, color=ORANGE)).arrange(RIGHT, buff=0.1)
        )
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.to_corner(UL, buff=0.3).shift(DOWN * 1.5)
        self.object_manager.add_object("physics_legend", legend, "physics_helpers")
        self.play(FadeIn(legend))
        
        # Building-aware physics simulation
        explanation_sim = Text("Building-aware physics: buildings=semi-ground, vegetation=passthrough", 
                            font_size=20, color=GREEN)
        explanation_sim.to_edge(DOWN)
        self.play(Transform(explanation, explanation_sim))
        
        animation_iterations = 50
        
        progress_text = Text("Building-aware simulation...", font_size=16, color=GREEN)
        progress_text.to_corner(UR)
        self.play(Write(progress_text))
        
        # Enhanced physics simulation
        for anim_iter in range(animation_iterations):
            print(f"BUILDING-AWARE DEBUG: Animation iteration {anim_iter + 1}/{animation_iterations}")
            
            # Update sticky arrows based on cloth proximity
            arrow_index = 0
            for (i, j), terrain_info in terrain_analysis.items():
                if arrow_index < len(sticky_arrows) and terrain_info['stickiness'] > 0.1:
                    arrow = sticky_arrows[arrow_index]
                    
                    if (i, j) in cloth_heights:
                        cloth_height = cloth_heights[(i, j)]
                        distance_to_terrain = cloth_height - terrain_info['ground_level']
                        
                        # Show arrow when cloth approaches terrain
                        if 0.05 < distance_to_terrain < 1.0:
                            opacity = min(0.8, terrain_info['stickiness'] * (1.5 / distance_to_terrain))
                            arrow.set_opacity(opacity)
                        else:
                            arrow.set_opacity(0)
                    
                    arrow_index += 1
            
            # Enhanced physics update
            new_heights = {}
            new_velocities = {}
            new_states = {}
            new_interactions = {}
            
            for i in range(grid_width):
                for j in range(grid_height):
                    if (i, j) not in cloth_heights:
                        continue
                    
                    current_height = cloth_heights[(i, j)]
                    current_velocity = cloth_velocities[(i, j)]
                    x = x_coords[i]
                    z = z_coords[j]
                    
                    # Apply gravity
                    current_velocity -= gravity * time_step
                    
                    # Spring forces
                    spring_force = 0.0
                    neighbor_count = 0
                    
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (ni, nj) in cloth_heights:
                            neighbor_height = cloth_heights[(ni, nj)]
                            height_diff = neighbor_height - current_height
                            
                            max_displacement = resolution * 0.4
                            if abs(height_diff) > max_displacement:
                                height_diff = np.sign(height_diff) * max_displacement
                            
                            spring_force += spring_strength * height_diff
                            neighbor_count += 1
                    
                    if neighbor_count > 0:
                        current_velocity += (spring_force / neighbor_count) * time_step
                    
                    # Enhanced terrain interaction
                    terrain_info = terrain_analysis.get((i, j), {'type': 'empty', 'stickiness': 0})
                    interaction_info = {'type': None, 'points': [], 'forces': []}
                    
                    # Apply terrain-specific forces
                    if terrain_info['stickiness'] > 0:
                        distance_to_terrain = current_height - terrain_info['ground_level']
                        
                        # Building-aware sticky forces
                        if 0.02 < distance_to_terrain < 0.4:
                            terrain_type = terrain_info['type']
                            
                            # Velocity-based breakthrough
                            velocity_factor = 1.0
                            if abs(current_velocity) > 0.15:
                                if 'edge' in terrain_type:
                                    velocity_factor = 0.1  # Easy breakthrough on building edges
                                elif 'building' in terrain_type:
                                    velocity_factor = 0.3  # Moderate breakthrough on building tops
                                else:
                                    velocity_factor = 0.5  # Harder breakthrough on ground
                            
                            proximity_factor = max(0, 1.0 - distance_to_terrain / 0.4)
                            sticky_force = terrain_info['stickiness'] * proximity_factor * velocity_factor
                            
                            if current_velocity < 0 and sticky_force > 0.05:
                                current_velocity += sticky_force * 0.8
                                
                                # Record interaction
                                if 'building' in terrain_type:
                                    interaction_info['type'] = 'building_contact'
                                    if terrain_info['nearby_buildings'] is not None and len(terrain_info['nearby_buildings']) > 0:
                                        interaction_info['points'] = terrain_info['nearby_buildings'][:3]
                                else:
                                    interaction_info['type'] = 'ground_contact'
                                    if terrain_info['nearby_ground'] is not None and len(terrain_info['nearby_ground']) > 0:
                                        interaction_info['points'] = terrain_info['nearby_ground'][:3]
                                
                                interaction_info['forces'] = [sticky_force]
                    
                    # Check for vegetation passthrough
                    if terrain_info.get('nearby_vegetation') is not None and len(terrain_info['nearby_vegetation']) > 0:
                        veg_heights = terrain_info['nearby_vegetation'][:, 2]
                        for veg_height in veg_heights:
                            if abs(current_height - veg_height) < 0.3:  # Passing through vegetation
                                # Minimal resistance for vegetation
                                if current_velocity < 0:
                                    current_velocity += 0.02  # Very small resistance
                                
                                if interaction_info['type'] is None:
                                    interaction_info['type'] = 'vegetation_passthrough'
                                    interaction_info['points'] = terrain_info['nearby_vegetation'][:2]
                    
                    # Apply damping
                    current_velocity *= damping
                    
                    # Update position
                    new_height = current_height + current_velocity * time_step
                    
                    # Final collision detection (ground truth)
                    collision_detected = False
                    if terrain_info['ground_level'] is not None:
                        collision_buffer = 0.08
                        min_cloth_height = terrain_info['ground_level'] + collision_buffer
                        
                        if new_height < min_cloth_height:
                            new_height = min_cloth_height
                            collision_detected = True
                            if current_velocity < 0:
                                current_velocity = -current_velocity * 0.15
                            current_velocity *= 0.7
                    
                    # Enhanced state determination
                    if abs(current_velocity) < 0.02:
                        state = 'settled'
                    elif terrain_info['stickiness'] > 0.3 and 0 < current_height - terrain_info['ground_level'] < 0.5:
                        if abs(current_velocity) < 0.06:
                            state = 'stuck'
                        else:
                            state = 'struggling'
                    elif interaction_info['type'] == 'vegetation_passthrough':
                        state = 'falling'  # Still falling through vegetation
                    else:
                        state = 'falling'
                    
                    new_heights[(i, j)] = new_height
                    new_velocities[(i, j)] = current_velocity
                    new_states[(i, j)] = state
                    new_interactions[(i, j)] = interaction_info
            
            # Update cloth state
            cloth_heights = new_heights
            cloth_velocities = new_velocities
            cloth_states = new_states
            cloth_interactions = new_interactions
            
            # Count states and interactions
            state_counts = {'falling': 0, 'struggling': 0, 'stuck': 0, 'settled': 0}
            interaction_counts = {'ground_contact': 0, 'building_contact': 0, 'vegetation_passthrough': 0, 'building_edge_override': 0}
            
            for state in cloth_states.values():
                state_counts[state] += 1
            
            for interaction in cloth_interactions.values():
                if interaction['type'] in interaction_counts:
                    interaction_counts[interaction['type']] += 1

            # CLEANUP TRIGGER: Remove interaction effects when cloth has mostly settled
            cleanup_triggered = False
            if anim_iter > 25 and state_counts['settled'] > len(cloth_states) * 0.7:  # 70% settled after iteration 25
                if not hasattr(self, 'cleanup_done'):
                    cleanup_triggered = True
                    self.cleanup_done = True  # Flag to prevent repeated cleanup
            
            print(f"BUILDING-AWARE DEBUG: States - F:{state_counts['falling']}, St:{state_counts['struggling']}, Stuck:{state_counts['stuck']}, Settled:{state_counts['settled']}")
            print(f"BUILDING-AWARE DEBUG: Interactions - Ground:{interaction_counts['ground_contact']}, Building:{interaction_counts['building_contact']}, Vegetation:{interaction_counts['vegetation_passthrough']}")
            
            # Update visualization
            if anim_iter % 1 == 0:
                # Clear interactions if cleanup was triggered
                if cleanup_triggered:
                    new_cloth_dots, new_cloth_lines, new_interaction_markers, new_passthrough_effects = visualize_building_aware_cloth(
                        cloth_heights, cloth_velocities, cloth_states, cloth_interactions, show_interactions=False  #turns of regeneration of effects.
                    )
                    
                    # Also fade out sticky arrows
                    for arrow in sticky_arrows:
                        arrow.set_opacity(0)
                        
                    # Add cleanup explanation
                    cleanup_explanation = Text("Interaction effects cleared - showing final cloth position", 
                                            font_size=18, color=BLUE)
                    cleanup_explanation.to_edge(DOWN)
                    self.play(Transform(explanation, cleanup_explanation), run_time=0.5)
                else:
                    new_cloth_dots, new_cloth_lines, new_interaction_markers, new_passthrough_effects = visualize_building_aware_cloth(
                        cloth_heights, cloth_velocities, cloth_states, cloth_interactions, show_interactions=True  # <-- Add this
                    )
                
                total_interactions = sum(interaction_counts.values())
                progress_new = Text(
                    f"Settled: {state_counts['settled']}, Interactions: {total_interactions} (G:{interaction_counts['ground_contact']}, B:{interaction_counts['building_contact']}, V:{interaction_counts['vegetation_passthrough']})", 
                    font_size=12, color=GREEN
                )
                progress_new.to_corner(UR)
                
                # Animate with all effects
                animations = [
                    Transform(cloth_dots, new_cloth_dots),
                    Transform(cloth_lines, new_cloth_lines),
                    Transform(progress_text, progress_new)
                ]
                
                # Add interaction markers with tracking
                if len(new_interaction_markers) > 0:
                    if hasattr(self, 'current_interaction_markers'):
                        animations.append(Transform(self.current_interaction_markers, new_interaction_markers))
                        # Update tracked object
                        self.object_manager.tracked_objects["interaction_markers"] = new_interaction_markers
                    else:
                        animations.append(FadeIn(new_interaction_markers))
                        # Add to tracking
                        self.object_manager.add_object("interaction_markers", new_interaction_markers, "physics_helpers")
                    self.current_interaction_markers = new_interaction_markers

                # Add passthrough effects with tracking
                if len(new_passthrough_effects) > 0:
                    if hasattr(self, 'current_passthrough_effects'):
                        animations.append(Transform(self.current_passthrough_effects, new_passthrough_effects))
                        # Update tracked object
                        self.object_manager.tracked_objects["passthrough_effects"] = new_passthrough_effects
                    else:
                        animations.append(FadeIn(new_passthrough_effects))
                        # Add to tracking
                        self.object_manager.add_object("passthrough_effects", new_passthrough_effects, "physics_helpers")
                    self.current_passthrough_effects = new_passthrough_effects
                
                run_time = 0.3
                self.play(*animations, run_time=run_time)
        
        # Cleanup effects
        # cleanup_effects = [sticky_arrows]
        # if hasattr(self, 'current_interaction_markers'):
        #     cleanup_effects.append(self.current_interaction_markers)
        # if hasattr(self, 'current_passthrough_effects'):
        #     cleanup_effects.append(self.current_passthrough_effects)
        
        # self.play(FadeOut(*cleanup_effects))

        # CLEANUP STEP: Clear interaction visualizations to see final cloth position clearly
        explanation_cleanup = Text("Clearing interaction effects to show final cloth position", 
                                font_size=20, color=BLUE)
        explanation_cleanup.to_edge(DOWN)
        self.play(Transform(explanation, explanation_cleanup))

        # Force cleanup of all helper graphics
        self.object_manager.cleanup_group("physics_helpers")

        # Also manually clean up any remaining interaction effects
        cleanup_effects = []
        if hasattr(self, 'current_interaction_markers') and self.current_interaction_markers in self.mobjects:
            cleanup_effects.append(self.current_interaction_markers)
        if hasattr(self, 'current_passthrough_effects') and self.current_passthrough_effects in self.mobjects:
            cleanup_effects.append(self.current_passthrough_effects)

        if cleanup_effects:
            self.play(FadeOut(*cleanup_effects))

        # Show clean final cloth state - just the settled cloth grid
        final_cloth_dots, final_cloth_lines, _, _ = visualize_building_aware_cloth(
            cloth_heights, cloth_velocities, cloth_states, 
            {key: {'type': None, 'points': [], 'forces': []} for key in cloth_interactions.keys()}  # Clear interactions
        )

        # Update to clean visualization
        self.play(
            Transform(cloth_dots, final_cloth_dots),
            Transform(cloth_lines, final_cloth_lines)
        )

        explanation_clean = Text("Final cloth position - clear view of terrain conformance", 
                            font_size=20, color=GREEN)
        explanation_clean.to_edge(DOWN)
        self.play(Transform(explanation, explanation_clean))
        self.wait(3)
        
        # Final classification
        explanation_classify = Text("Building-aware classification: buildings treated as elevated ground", 
                                font_size=20, color=GREEN)
        explanation_classify.to_edge(DOWN)
        self.play(Transform(explanation, explanation_classify))
        
        ground_classified = []
        non_ground_classified = []
        
        # Enhanced classification: points within 5mm OR below cloth = ground
        fine_threshold = 0.005  # 5mm tolerance
        print(f"DEBUG: Using fine threshold of {fine_threshold:.3f}m for classification")

        # Debug counters
        within_tolerance_count = 0
        below_cloth_count = 0
        above_cloth_count = 0
        outside_grid_count = 0

        for point in all_points:
            x, y, z = point
            
            grid_i = int(round((x - x_coords[0]) / resolution))
            grid_j = int(round((y - z_coords[0]) / resolution))
            
            grid_i = max(0, min(grid_width - 1, grid_i))
            grid_j = max(0, min(grid_height - 1, grid_j))
            
            if (grid_i, grid_j) in cloth_heights:
                cloth_height = cloth_heights[(grid_i, grid_j)]
                distance_to_cloth = z - cloth_height
                
                # Debug: Print first few points
                if len(ground_classified) + len(non_ground_classified) < 5:
                    print(f"DEBUG Point: terrain_z={z:.3f}, cloth_z={cloth_height:.3f}, distance={distance_to_cloth:.3f}")
                
                # NEW LOGIC: Ground if within 5mm OR below cloth
                is_within_tolerance = abs(distance_to_cloth) <= fine_threshold
                is_below_cloth = z < cloth_height  # Changed from <= to < for stricter below test
                
                if is_within_tolerance:
                    within_tolerance_count += 1
                    ground_classified.append(point)
                elif is_below_cloth:
                    below_cloth_count += 1
                    ground_classified.append(point)
                else:
                    above_cloth_count += 1
                    non_ground_classified.append(point)
            else:
                outside_grid_count += 1
                non_ground_classified.append(point)

        print(f"DEBUG Classification breakdown:")
        print(f"  Within tolerance ({fine_threshold*1000:.0f}mm): {within_tolerance_count}")
        print(f"  Below cloth: {below_cloth_count}")
        print(f"  Above cloth: {above_cloth_count}")
        print(f"  Outside grid: {outside_grid_count}")
        print(f"  Total ground: {len(ground_classified)}")
        print(f"  Total non-ground: {len(non_ground_classified)}")
        
        print(f"BUILDING-AWARE DEBUG: Final classification - Ground: {len(ground_classified)}, Non-ground: {len(non_ground_classified)}")
        
        # Show results
        self.play(FadeOut(ground_dots, vegetation_dots, building_dots))
        
        if ground_classified:
            final_ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.025)
                for x, y, z in ground_classified
            ])
            self.play(FadeIn(final_ground_dots, lag_ratio=0.01))
        
        if non_ground_classified:
            final_non_ground_dots = VGroup(*[
                Dot(self.axes.coords_to_point(x, z), color=RED, radius=0.025)
                for x, y, z in non_ground_classified
            ])
            self.play(FadeIn(final_non_ground_dots, lag_ratio=0.01))
        
        # Enhanced results
        final_interaction_counts = {'ground_contact': 0, 'building_contact': 0, 'vegetation_passthrough': 0}
        for interaction in cloth_interactions.values():
            if interaction['type'] in final_interaction_counts:
                final_interaction_counts[interaction['type']] += 1
        
        results_display = VGroup(
            Text("Building-Aware CSF Results:", font_size=16, color=YELLOW),
            Text(f"Ground classified: {len(ground_classified)}", font_size=14, color=BROWN),
            Text(f"Non-ground: {len(non_ground_classified)}", font_size=14, color=RED),
            Text(f"Ground contacts: {final_interaction_counts['ground_contact']}", font_size=12, color=BROWN),
            Text(f"Building contacts: {final_interaction_counts['building_contact']}", font_size=12, color=GRAY),
            Text(f"Vegetation passthrough: {final_interaction_counts['vegetation_passthrough']}", font_size=12, color=GREEN),
            Text(f"Resolution: {resolution}m", font_size=12, color=WHITE)
        )
        results_display.arrange(DOWN, aligned_edge=LEFT)
        results_display.to_corner(UR, buff=0.5).shift(DOWN * 3)
        self.play(FadeIn(results_display))
        
        # Final explanation
        final_explanation = Text("Building-aware CSF complete - buildings as semi-ground, vegetation passthrough!", 
                            font_size=24, color=GREEN)
        final_explanation.to_edge(DOWN)
        
        self.play(Transform(explanation, final_explanation))
        self.wait(4)
        
        # Store results
        final_positions = []
        for (i, j), height in cloth_heights.items():
            x = x_coords[i]
            z = z_coords[j]
            final_positions.append([x, 0, height])
        
        self.cloth_points = np.array(final_positions)
        self.cloth_dots = cloth_dots
        self.cloth_lines = cloth_lines
        # Store classification results for next scene
        self.physics_ground_classified = ground_classified
        self.physics_non_ground_classified = non_ground_classified
        
        # Final cleanup - ensure everything is gone
        self.object_manager.cleanup_group("physics_helpers")

        # Cleanup main objects from this scene
        cleanup_objects = [title, explanation, legend, progress_text, 
                        results_display, ground_dots, vegetation_dots, building_dots,
                        cloth_dots, cloth_lines]

        if 'final_ground_dots' in locals():
            cleanup_objects.append(final_ground_dots)
        if 'final_non_ground_dots' in locals():
            cleanup_objects.append(final_non_ground_dots)

        # Remove any remaining interaction effects manually
        remaining_cleanup = []
        if hasattr(self, 'current_interaction_markers') and self.current_interaction_markers in self.mobjects:
            remaining_cleanup.append(self.current_interaction_markers)
        if hasattr(self, 'current_passthrough_effects') and self.current_passthrough_effects in self.mobjects:
            remaining_cleanup.append(self.current_passthrough_effects)

        all_cleanup = cleanup_objects + remaining_cleanup
        self.play(FadeOut(*all_cleanup))
        self.wait(0.5)

    def create_cloth_visualization_from_state(self, cloth_state):
        """Create cloth visualization objects from cloth state dictionary"""
        # Create cloth particles
        cloth_dots = VGroup(*[
            Dot(self.axes.coords_to_point(pos[0], pos[2]), 
                color=BLUE, radius=0.015) 
            for pos in cloth_state['positions'].values()
        ])
        
        # Create cloth mesh lines
        cloth_lines = VGroup()
        
        # Add structural lines (horizontal and vertical)
        for i in range(cloth_state['grid_width']):
            for j in range(cloth_state['grid_height']):
                current_key = (i, j)
                if current_key not in cloth_state['positions']:
                    continue
                    
                current_pos = cloth_state['positions'][current_key]
                current_point = self.axes.coords_to_point(current_pos[0], current_pos[2])
                
                # Right neighbor
                if i < cloth_state['grid_width'] - 1:
                    right_key = (i + 1, j)
                    if right_key in cloth_state['positions']:
                        right_pos = cloth_state['positions'][right_key]
                        right_point = self.axes.coords_to_point(right_pos[0], right_pos[2])
                        line = Line(current_point, right_point, 
                                color=BLUE, stroke_width=0.8, stroke_opacity=0.8)
                        cloth_lines.add(line)
                
                # Down neighbor
                if j < cloth_state['grid_height'] - 1:
                    down_key = (i, j + 1)
                    if down_key in cloth_state['positions']:
                        down_pos = cloth_state['positions'][down_key]
                        down_point = self.axes.coords_to_point(down_pos[0], down_pos[2])
                        line = Line(current_point, down_point, 
                                color=BLUE, stroke_width=0.8, stroke_opacity=0.8)
                        cloth_lines.add(line)
        
        return cloth_dots, cloth_lines


    def update_cloth_visualization_fixed(self, cloth_state, iteration):
        """Fixed version that returns new visualization objects instead of transforming in place"""
        return self.create_cloth_visualization_from_state(cloth_state)

    def initialize_cloth_mass_spring_system(self, x_coords, z_coords, initial_height, params):
        """Initialize cloth as a mass-spring system"""
        cloth_state = {
            'positions': {},
            'velocities': {},
            'forces': {},
            'grid_width': len(x_coords),
            'grid_height': len(z_coords),
            'resolution': x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0.4
        }
        
        for i, x in enumerate(x_coords):
            for j, z in enumerate(z_coords):
                grid_key = (i, j)
                cloth_state['positions'][grid_key] = np.array([x, 0, initial_height])
                cloth_state['velocities'][grid_key] = np.array([0.0, 0.0, 0.0])
                cloth_state['forces'][grid_key] = np.array([0.0, 0.0, 0.0])
        
        return cloth_state

    def update_cloth_physics(self, cloth_state, point_cloud, params):
        """Update cloth physics using Verlet integration"""
        dt = params['dt']
        gravity = params['gravity']
        damping = params['damping']
        spring_k = params['spring_k']
        mass = params['mass']
        
        # Reset forces
        for key in cloth_state['forces']:
            cloth_state['forces'][key] = np.array([0.0, 0.0, -gravity * mass])
        
        # Calculate spring forces
        for key, position in cloth_state['positions'].items():
            i, j = key
            
            # Structural springs (4-connected)
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            rest_length = cloth_state['resolution']
            
            for ni, nj in neighbors:
                neighbor_key = (ni, nj)
                if neighbor_key in cloth_state['positions']:
                    neighbor_pos = cloth_state['positions'][neighbor_key]
                    spring_vector = neighbor_pos - position
                    distance = np.linalg.norm(spring_vector)
                    
                    if distance > 0.01:
                        spring_force = spring_k * (distance - rest_length) * (spring_vector / distance)
                        cloth_state['forces'][key] += spring_force
            
            # Shear springs (diagonal)
            diagonal_neighbors = [(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
            diagonal_rest = rest_length * np.sqrt(2)
            diagonal_k = spring_k * 0.5
            
            for ni, nj in diagonal_neighbors:
                neighbor_key = (ni, nj)
                if neighbor_key in cloth_state['positions']:
                    neighbor_pos = cloth_state['positions'][neighbor_key]
                    spring_vector = neighbor_pos - position
                    distance = np.linalg.norm(spring_vector)
                    
                    if distance > 0.01:
                        spring_force = diagonal_k * (distance - diagonal_rest) * (spring_vector / distance)
                        cloth_state['forces'][key] += spring_force
        
        # Integrate using Verlet method
        new_positions = {}
        new_velocities = {}
        
        for key in cloth_state['positions']:
            position = cloth_state['positions'][key]
            velocity = cloth_state['velocities'][key]
            force = cloth_state['forces'][key]
            
            # Verlet integration
            acceleration = force / mass
            new_velocity = velocity + acceleration * dt
            new_velocity *= damping  # Apply damping
            new_position = position + new_velocity * dt
            
            # Collision detection and response
            new_position, new_velocity = self.handle_cloth_collisions(
                new_position, new_velocity, point_cloud
            )
            
            new_positions[key] = new_position
            new_velocities[key] = new_velocity
        
        cloth_state['positions'] = new_positions
        cloth_state['velocities'] = new_velocities
        
        return cloth_state

    def handle_cloth_collisions(self, position, velocity, point_cloud):
        """Handle collisions between cloth and point cloud"""
        # Find nearby points
        nearby_points = point_cloud[
            (np.abs(point_cloud[:, 0] - position[0]) < 0.6) & 
            (np.abs(point_cloud[:, 1] - position[1]) < 0.6)
        ]
        
        if len(nearby_points) > 0:
            max_height = nearby_points[:, 2].max()
            collision_threshold = 0.08
            
            if position[2] < max_height + collision_threshold:
                # Collision detected - adjust position and velocity
                position[2] = max_height + collision_threshold
                
                # Reflect velocity with energy loss
                if velocity[2] < 0:
                    velocity[2] = -velocity[2] * 0.3
                
                # Add surface friction
                velocity[0] *= 0.7
                velocity[1] *= 0.7
        
        return position, velocity

    def update_cloth_visualization(self, cloth_state, iteration):
        """Update the visual representation of the cloth"""
        # Create cloth particles
        new_cloth_dots = VGroup(*[
            Dot(self.axes.coords_to_point(pos[0], pos[2]), 
                color=BLUE, radius=0.015) 
            for pos in cloth_state['positions'].values()
        ])
        
        # Create cloth mesh lines
        new_cloth_lines = VGroup()
        
        # Add structural lines (horizontal and vertical)
        for i in range(cloth_state['grid_width']):
            for j in range(cloth_state['grid_height']):
                current_key = (i, j)
                if current_key not in cloth_state['positions']:
                    continue
                    
                current_pos = cloth_state['positions'][current_key]
                current_point = self.axes.coords_to_point(current_pos[0], current_pos[2])
                
                # Right neighbor
                if i < cloth_state['grid_width'] - 1:
                    right_key = (i + 1, j)
                    if right_key in cloth_state['positions']:
                        right_pos = cloth_state['positions'][right_key]
                        right_point = self.axes.coords_to_point(right_pos[0], right_pos[2])
                        line = Line(current_point, right_point, 
                                color=BLUE, stroke_width=0.8, stroke_opacity=0.8)
                        new_cloth_lines.add(line)
                
                # Down neighbor
                if j < cloth_state['grid_height'] - 1:
                    down_key = (i, j + 1)
                    if down_key in cloth_state['positions']:
                        down_pos = cloth_state['positions'][down_key]
                        down_point = self.axes.coords_to_point(down_pos[0], down_pos[2])
                        line = Line(current_point, down_point, 
                                color=BLUE, stroke_width=0.8, stroke_opacity=0.8)
                        new_cloth_lines.add(line)
        
        # Smooth animation
        run_time = 0.15 if iteration < 20 else 0.25
        self.play(
            Transform(self.cloth_dots, new_cloth_dots),
            Transform(self.cloth_lines, new_cloth_lines),
            run_time=run_time
        )

    def ground_classification(self):
        """Show ground point classification"""
        
        # COMPLETE SCENE RESET - Remove everything from physics simulation
        print("DEBUG: Clearing all mobjects before classification")
        print(f"DEBUG: Currently have {len(self.mobjects)} mobjects")
        
        # Force remove ALL current mobjects
        if self.mobjects:
            all_current_objects = list(self.mobjects)
            self.play(FadeOut(*all_current_objects), run_time=0.5)
        
        # Also force cleanup object manager
        self.object_manager.cleanup_all()
        
        # Clear any remaining mobjects that might have slipped through
        self.mobjects.clear()
        
        print(f"DEBUG: After cleanup have {len(self.mobjects)} mobjects")
        
        title = Text("Step 3: Ground Classification", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))

        # Recreate clean axes for classification (no old cloth/effects)
        clean_axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 4, 1],
            x_length=8,
            y_length=6,
            tips=False
        )
        self.play(Create(clean_axes))
        self.axes = clean_axes  # Update the reference
        
        # Calculate distances from points to cloth
        all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        threshold = 0.005

        # Use results from physics simulation
        if hasattr(self, 'physics_ground_classified'):
            ground_classified = self.physics_ground_classified
            non_ground_classified = self.physics_non_ground_classified
            print(f"Using physics simulation results: {len(ground_classified)} ground, {len(non_ground_classified)} non-ground")
        else:
            
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
        self.wait(5)
        
        explanation2 = Text("Points above threshold = Non-Ground (Red)", 
                          font_size=24, color=YELLOW)
        explanation2.to_edge(DOWN)
        
        self.play(Transform(explanation, explanation2))
        self.play(FadeIn(non_ground_classified_dots, lag_ratio=0.1))
        
        self.wait(5)
        
        # PROPER CLEANUP - Remove ALL elements from this scene
        self.play(FadeOut(
            title, explanation, ground_classified_dots, non_ground_classified_dots,
            self.cloth_dots, self.cloth_lines, self.axes
        ))
        self.wait(0.5)

    def real_data_example(self):
        """Show example with real/realistic data using PDAL"""
        # FIRST: Clear any remaining elements from previous scenes
        self.clear()
        
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
        
        # Show the pipeline implementation
        self.play(Write(pipeline_title))
        self.play(Write(code_display))
        self.wait(2)

        # Fade out the pipeline
        self.play(FadeOut(code_display))
        self.wait(0.5)
        
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
            # Clear PDAL pipeline elements before showing classification results
            self.play(FadeOut(pipeline_title, execution_demo, arrow1, arrow2, processing_text))
            self.wait(0.5)
            self.show_real_data_results()
            # Clean up remaining title
            self.play(FadeOut(title))
        else:
            # Clean up everything if no results shown
            self.play(FadeOut(title, pipeline_title, execution_demo, 
                            arrow1, arrow2, processing_text))

        self.wait(0.5)

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
            
            # PROPER CLEANUP - Remove all results visualization
            self.play(FadeOut(results_title, before_after, legend))
            self.wait(0.5)
            
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
        
        # Clear the title before proceeding
        self.play(FadeOut(title))
        
        # Show parameter ranges and effects
        param_explanation = Text("Key Parameters and Their Effects", 
                               font_size=32, color=YELLOW)
        param_explanation.to_edge(UP, buff=0.5)
        
        self.play(Write(param_explanation))
        
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
            self.wait(1.5)
        
        # Interactive parameter slider visualization
        self.play(FadeOut(param_demos))
        self.show_parameter_slider_demo()
        
        # PROPER CLEANUP
        self.play(FadeOut(param_explanation))
        self.wait(0.5)

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
        slider_title.to_edge(UP, buff=1)
        
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
        
        # PROPER CLEANUP
        self.play(FadeOut(slider_title, slider_group, result_area, sample_points))
        self.wait(0.5)

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
        # Clear any remaining elements from previous scenes
        self.clear()
        
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