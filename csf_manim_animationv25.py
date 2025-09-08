"""
CSF Algorithm Presentation - Fixed Version
Issues addressed:
1. Proper 3D visualization in problem statement
2. Fixed threshold scale (10mm vs scene units)
3. Reset point colors before classification
4. Improved distance calculation to cloth
"""

from manim import *
import numpy as np
import json

# Define custom colors
BROWN = "#8B4513"
LIGHT_GRAY = "#D3D3D3"
GROUND_COLOR = BROWN
NON_GROUND_COLOR = RED
CLOTH_COLOR = BLUE

class CSFPresentation(ThreeDScene):
    def construct(self):
        # Initialize scene management
        self.setup_scene()
        
        # Follow the presentation outline
        self.slide_1_opening()
        self.slide_2_problem_3d()
        self.slide_3_algorithm_overview()
        self.slide_4_real_data_pdal()
        self.slide_5_pdal_pipeline()
        self.slide_6_before_after()
        self.slide_7_parameter_tuning()
        self.slide_8_summary()

    def setup_scene(self):
        """Initialize coordinate system and data"""
        # Create consistent 2D coordinate system (Z vertical, X horizontal)
        self.axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 4, 1],
            x_length=10,
            y_length=6,
            tips=False
        )
        
        # Generate synthetic LiDAR data for consistency
        self.generate_lidar_data()

    def generate_lidar_data(self):
        """Generate consistent synthetic LiDAR data with realistic scale"""
        np.random.seed(42)
        
        # Ground points (terrain surface) - scale in meters
        x_ground = np.linspace(-4, 4, 60)
        y_ground = np.zeros(60)
        z_ground = 0.3 * np.sin(x_ground * 0.8) + 0.1 * np.random.normal(size=60)
        self.ground_points = np.column_stack([x_ground, y_ground, z_ground])
        
        # Vegetation points (above ground)
        veg_x = np.random.uniform(-3.5, 3.5, 40)
        veg_y = np.zeros(40)
        veg_z = np.random.uniform(0.8, 3.0, 40)
        self.vegetation_points = np.column_stack([veg_x, veg_y, veg_z])
        
        # Building points (structured above ground)
        building_x = np.random.uniform(1, 3, 25)
        building_y = np.zeros(25)
        building_z = np.random.uniform(2.5, 3.8, 25)
        self.building_points = np.column_stack([building_x, building_y, building_z])
        
        # Combined point cloud
        self.all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])

    def slide_1_opening(self):
        """Slide 1: Opening slide with title"""
        title = Text("CSF Algorithm", font_size=72, color=BLUE)
        subtitle = Text("Cloth Simulation Filter for Ground Point Detection", 
                       font_size=36, color=WHITE)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        authors = Text("Zhang et al. (2016) - Remote Sensing", 
                      font_size=24, color=GRAY)
        authors.next_to(subtitle, DOWN, buff=1)
        
        # Animated title appearance
        self.play(
            Write(title),
            FadeIn(subtitle, shift=UP),
            run_time=2
        )
        self.wait(1)
        
        self.play(Write(authors))
        self.wait(2)
        
        # Transition out
        self.play(FadeOut(title, subtitle, authors))
        self.wait(0.5)

    def slide_2_problem_3d(self):
        """Slide 2: Show 3D problem with proper 3D visualization"""
        problem_title = Text("Problem: Ground or Non-Ground?", 
                           font_size=48, color=YELLOW)
        problem_title.to_edge(UP)

        explanation = Text("Challenge: Which points represent the ground surface?", 
                     font_size=24, color=WHITE)
        explanation.to_edge(DOWN)
        
        self.play(Write(problem_title))
        self.play(Write(explanation))
    
        # Create proper 3D axes
        axes_3d = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1], 
            z_range=[-1, 4, 1],
            x_length=8,
            y_length=6,
            z_length=5,
        )
        
        # Add axis labels positioned for standard Z-vertical orientation
        x_label = Text("X", font_size=20, color=WHITE).next_to(axes_3d.x_axis, RIGHT)
        y_label = Text("Z", font_size=20, color=WHITE).next_to(axes_3d.y_axis, DOWN + RIGHT)
        z_label = Text("Y", font_size=20, color=WHITE).next_to(axes_3d.z_axis, UP)
        
        self.play(Create(axes_3d))
        self.play(Write(x_label), Write(y_label), Write(z_label))
        
        # Create 3D points - all same color initially to show the problem
        all_dots_3d = VGroup()
        for x, y, z in self.all_points:
            # Add some Y variation for true 3D spread
            y_3d = y + np.random.uniform(-1, 1)
            point_3d = Dot3D(
                point=axes_3d.coords_to_point(x, y_3d, z),
                color=WHITE,
                radius=0.03
            )
            all_dots_3d.add(point_3d)
        
        self.play(FadeIn(all_dots_3d, lag_ratio=0.02))
        
        # CRITICAL: Only group the 3D elements that should rotate
        # Do NOT include title and explanation in this group
        rotating_group = VGroup(axes_3d, all_dots_3d, x_label, y_label, z_label)
        
        # Apply modest rotations to show 3D nature while keeping Z generally vertical
        # Remove all camera manipulation to keep Z-axis vertical
        self.play(
            Rotate(rotating_group, PI/8, axis=UP),  # Smaller rotation
            run_time=2
        )
        self.play(
            Rotate(rotating_group, PI/12, axis=RIGHT),  # Smaller rotation
            run_time=2
        )
        self.play(
            Rotate(rotating_group, -PI/8, axis=UP),  # Return somewhat
            run_time=2
        )
        
        self.wait(1)
        
        # Transition: fade out for next slide
        self.play(FadeOut(problem_title, explanation, rotating_group))
        self.wait(0.5)

    def slide_3_algorithm_overview(self):
        """Slide 3: Core CSF algorithm demonstration - RESET TO 2D"""
        # CRITICAL: Reset camera to 2D orientation at the start
        #self.move_camera(phi=0, theta=0, gamma=0, run_time=0.5)
        
        # This is the main algorithm slide following steps a-j
        
        # Step 3a: Initialize point cloud on 2D coordinate view
        self.step_3a_initialize_2d()
        
        # Step 3b: Invert the point cloud by moving points
        self.step_3b_invert_cloud()
        
        # Step 3c: Initial cloth above inverted cloud
        self.step_3c_initialize_cloth()
        
        # Step 3d: Animate cloth settling with gravity and collisions
        self.step_3d_cloth_settling()
        
        # Step 3e: Identify cloth surface after settled
        self.step_3e_identify_cloth_surface()
        
        # Step 3f: Introduce threshold of 10mm
        self.step_3f_introduce_threshold()
        
        # Step 3g: Colorize points within 10mm
        self.step_3g_colorize_points()
        
        # Step 3h: Classify colorized points as 'ground'
        self.step_3h_classify_ground()
        
        # Step 3i: Remove all non-ground points
        self.step_3i_remove_non_ground()
        
        # Step 3j: Invert remaining ground points back
        self.step_3j_invert_back()

    def step_3a_initialize_2d(self):
        """Step 3a: Initialize point cloud on 2D coordinate view"""
        title = Text("CSF Algorithm: Step 1 - Initialize Point Cloud", 
                    font_size=36, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Create 2D axes (Z vertical, X horizontal)
        self.play(Create(self.axes))
        
        # Show points with original classification colors (educational)
        self.ground_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=BROWN, radius=0.03) 
            for x, y, z in self.ground_points
        ])
        
        self.vegetation_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=GREEN, radius=0.03) 
            for x, y, z in self.vegetation_points
        ])
        
        self.building_dots = VGroup(*[
            Dot(self.axes.coords_to_point(x, z), color=GRAY, radius=0.03) 
            for x, y, z in self.building_points
        ])
        
        # Add coordinate labels
        x_label = Text("X", font_size=24, color=WHITE)
        z_label = Text("Z", font_size=24, color=WHITE)
        x_label.next_to(self.axes.x_axis, RIGHT)
        z_label.next_to(self.axes.y_axis, UP)
        
        explanation = Text("2D View: X (horizontal) vs Z (vertical) - Y coordinate excluded", 
                         font_size=20, color=YELLOW)
        explanation.to_edge(DOWN)
        
        self.play(
            Write(x_label), Write(z_label),
            Write(explanation)
        )
        
        # Animate point appearance by type
        self.play(FadeIn(self.ground_dots, lag_ratio=0.05))
        self.wait(0.5)
        self.play(FadeIn(self.vegetation_dots, lag_ratio=0.05))
        self.wait(0.5)
        self.play(FadeIn(self.building_dots, lag_ratio=0.05))
        
        self.wait(2)
        
        # Store for transition
        self.current_title = title
        self.current_explanation = explanation
        self.coordinate_labels = VGroup(x_label, z_label)

    def step_3b_invert_cloud(self):
        """Step 3b: Invert the point cloud by moving points (not copying)"""
        new_title = Text("CSF Algorithm: Step 2 - Invert Point Cloud", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("Flip points upside down - ground becomes highest points", 
                             font_size=20, color=YELLOW)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # Calculate inversion parameters
        max_z = self.all_points[:, 2].max()
        min_z = self.all_points[:, 2].min()
        z_center = (max_z + min_z) / 2
        
        # Create inverted positions for each point group
        inverted_ground_positions = [
            self.axes.coords_to_point(x, 2*z_center - z) 
            for x, y, z in self.ground_points
        ]
        
        inverted_vegetation_positions = [
            self.axes.coords_to_point(x, 2*z_center - z) 
            for x, y, z in self.vegetation_points
        ]
        
        inverted_building_positions = [
            self.axes.coords_to_point(x, 2*z_center - z) 
            for x, y, z in self.building_points
        ]
        
        # Animate the inversion by moving existing points
        animations = []
        
        for i, dot in enumerate(self.ground_dots):
            animations.append(dot.animate.move_to(inverted_ground_positions[i]))
        
        for i, dot in enumerate(self.vegetation_dots):
            animations.append(dot.animate.move_to(inverted_vegetation_positions[i]))
            
        for i, dot in enumerate(self.building_dots):
            animations.append(dot.animate.move_to(inverted_building_positions[i]))
        
        # Execute inversion animation
        self.play(*animations, run_time=3)
        
        # Store inverted coordinates for cloth physics
        self.inverted_points = self.all_points.copy()
        self.inverted_points[:, 2] = 2*z_center - self.inverted_points[:, 2]
        
        self.wait(2)

    def step_3c_initialize_cloth(self):
        """Step 3c: Initial cloth above inverted cloud"""
        new_title = Text("CSF Algorithm: Step 3 - Initialize Cloth", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("Place virtual cloth above inverted surface", 
                             font_size=20, color=YELLOW)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # Calculate cloth parameters
        min_x, max_x = self.inverted_points[:, 0].min(), self.inverted_points[:, 0].max()
        inverted_max_z = self.inverted_points[:, 2].max()
        
        resolution = 0.4
        padding = resolution
        x_coords = np.arange(min_x - padding, max_x + padding, resolution)
        cloth_start_height = inverted_max_z + 1.5
        
        # Create cloth grid
        self.cloth_positions = {}
        self.cloth_velocities = {}
        
        for i, x in enumerate(x_coords):
            self.cloth_positions[i] = np.array([x, cloth_start_height])
            self.cloth_velocities[i] = 0.0
        
        # Visualize cloth
        self.cloth_dots = VGroup(*[
            Dot(self.axes.coords_to_point(pos[0], pos[1]), color=CLOTH_COLOR, radius=0.025)
            for pos in self.cloth_positions.values()
        ])
        
        # Create cloth connections
        self.cloth_lines = VGroup()
        for i in range(len(x_coords) - 1):
            start_pos = self.cloth_positions[i]
            end_pos = self.cloth_positions[i + 1]
            line = Line(
                self.axes.coords_to_point(start_pos[0], start_pos[1]),
                self.axes.coords_to_point(end_pos[0], end_pos[1]),
                color=CLOTH_COLOR, stroke_width=2
            )
            self.cloth_lines.add(line)
        
        self.play(
            FadeIn(self.cloth_dots, lag_ratio=0.1),
            Create(self.cloth_lines)
        )
        
        self.wait(2)
        
        # Store for physics simulation
        self.x_coords = x_coords
        self.resolution = resolution

    def step_3d_cloth_settling(self):
        """Step 3d: Animate cloth settling with gravity and collisions"""
        new_title = Text("CSF Algorithm: Step 4 - Cloth Physics Simulation", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("Cloth falls under gravity and collides with inverted points", 
                             font_size=20, color=YELLOW)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # Physics parameters
        gravity = 0.1
        damping = 0.9
        time_step = 0.5
        iterations = 20
        collision_buffer = 0.1
        
        # Run cloth physics simulation
        for iteration in range(iterations):
            new_positions = {}
            new_velocities = {}
            
            for i, position in self.cloth_positions.items():
                x, z = position
                velocity = self.cloth_velocities[i]
                
                # Apply gravity
                velocity -= gravity * time_step
                
                # Collision detection with inverted points
                nearby_points = self.inverted_points[
                    np.abs(self.inverted_points[:, 0] - x) < self.resolution
                ]
                
                if len(nearby_points) > 0:
                    max_nearby_z = nearby_points[:, 2].max()
                    
                    # Calculate new position
                    new_z = z + velocity * time_step
                    
                    # Handle collision
                    if new_z <= max_nearby_z + collision_buffer:
                        new_z = max_nearby_z + collision_buffer
                        velocity = 0.0  # Stop at collision
                else:
                    new_z = z + velocity * time_step
                
                # Apply damping
                velocity *= damping
                
                new_positions[i] = np.array([x, new_z])
                new_velocities[i] = velocity
            
            # Update positions
            self.cloth_positions = new_positions
            self.cloth_velocities = new_velocities
            
            # Update visualization every few iterations
            if iteration % 2 == 0:
                self.update_cloth_visualization()
        
        self.wait(2)

    def update_cloth_visualization(self):
        """Update cloth visualization during physics simulation"""
        # Update cloth dots
        new_cloth_dots = VGroup(*[
            Dot(self.axes.coords_to_point(pos[0], pos[1]), color=CLOTH_COLOR, radius=0.025)
            for pos in self.cloth_positions.values()
        ])
        
        # Update cloth lines
        new_cloth_lines = VGroup()
        for i in range(len(self.x_coords) - 1):
            start_pos = self.cloth_positions[i]
            end_pos = self.cloth_positions[i + 1]
            line = Line(
                self.axes.coords_to_point(start_pos[0], start_pos[1]),
                self.axes.coords_to_point(end_pos[0], end_pos[1]),
                color=CLOTH_COLOR, stroke_width=2
            )
            new_cloth_lines.add(line)
        
        self.play(
            Transform(self.cloth_dots, new_cloth_dots),
            Transform(self.cloth_lines, new_cloth_lines),
            run_time=0.3
        )

    def step_3e_identify_cloth_surface(self):
        """Step 3e: Identify cloth surface after settled"""
        new_title = Text("CSF Algorithm: Step 5 - Cloth Surface Identified", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("Cloth has settled - now represents approximated ground surface", 
                             font_size=20, color=GREEN)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # Highlight the settled cloth
        self.play(
            self.cloth_dots.animate.set_color(GREEN),
            self.cloth_lines.animate.set_color(GREEN),
            run_time=1
        )
        
        self.wait(2)

    def step_3f_introduce_threshold(self):
        """Step 3f: Introduce threshold - FIXED SCALE"""
        new_title = Text("CSF Algorithm: Step 6 - Distance Threshold", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("Set threshold: 0.5m distance from cloth surface (scaled for visibility)", 
                             font_size=20, color=YELLOW)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # FIXED: Use realistic threshold relative to scene scale
        # Scene spans ~8 units in Z, so 0.5 units = reasonable threshold
        threshold = 0.5  # Much larger than 0.01!
        
        # Visualize threshold zone around cloth
        threshold_zone = VGroup()
        
        for i in range(len(self.x_coords) - 1):
            pos1 = self.cloth_positions[i]
            pos2 = self.cloth_positions[i + 1]
            
            # Create threshold zone as transparent rectangles
            zone_height = threshold * 2  # Above and below cloth
            zone_rect = Rectangle(
                width=self.resolution * self.axes.x_axis.unit_size,
                height=zone_height * self.axes.y_axis.unit_size,
                color=YELLOW,
                fill_opacity=0.2,
                stroke_opacity=0.5
            )
            zone_center = self.axes.coords_to_point(
                (pos1[0] + pos2[0]) / 2,
                (pos1[1] + pos2[1]) / 2
            )
            zone_rect.move_to(zone_center)
            threshold_zone.add(zone_rect)
        
        self.play(FadeIn(threshold_zone, lag_ratio=0.1))
        
        # Store threshold for next step
        self.threshold = threshold
        self.threshold_zone = threshold_zone
        
        self.wait(2)

    def step_3g_colorize_points(self):
        """Step 3g: FIXED - Reset colors first, then colorize points within threshold"""
        new_title = Text("CSF Algorithm: Step 7 - Identify Ground Points", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("First reset all points to white, then mark ground points brown", 
                             font_size=20, color=YELLOW)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # FIXED: First reset all points to white (as if unclassified)
        all_dots = [*self.ground_dots, *self.vegetation_dots, *self.building_dots]
        reset_animations = [dot.animate.set_color(WHITE) for dot in all_dots]
        self.play(*reset_animations, run_time=1)
        
        self.wait(1)
        
        # Now update explanation for classification
        new_explanation2 = Text("Points within 0.5m threshold = Ground (Brown)", 
                              font_size=20, color=BROWN)
        new_explanation2.to_edge(DOWN)
        self.play(Transform(self.current_explanation, new_explanation2))
        
        # FIXED: Calculate which points are within threshold with proper distance calculation
        self.ground_classified_indices = []
        self.non_ground_classified_indices = []
        
        for idx, point in enumerate(self.inverted_points):
            x, y, z = point
            
            # Find distance to cloth surface at this X position
            min_distance = float('inf')
            
            # Check distance to all cloth positions
            for cloth_pos in self.cloth_positions.values():
                cloth_x, cloth_z = cloth_pos
                # Distance calculation: primarily Z-distance (vertical)
                if abs(x - cloth_x) < self.resolution * 2:  # Only nearby cloth points
                    distance = abs(z - cloth_z)
                    min_distance = min(min_distance, distance)
            
            # Classify based on threshold
            if min_distance <= self.threshold:
                self.ground_classified_indices.append(idx)
            else:
                self.non_ground_classified_indices.append(idx)
        
        print(f"DEBUG: Ground classified: {len(self.ground_classified_indices)}, Non-ground: {len(self.non_ground_classified_indices)}")
        
        # Animate colorization of ground points
        ground_animations = []
        for idx in self.ground_classified_indices:
            ground_animations.append(all_dots[idx].animate.set_color(GROUND_COLOR))
        
        if ground_animations:
            self.play(*ground_animations, lag_ratio=0.02)
        else:
            # If no points classified, show warning
            warning = Text("Warning: No points within threshold!", font_size=20, color=RED)
            warning.to_edge(DOWN)
            self.play(Write(warning))
            self.wait(2)
            self.play(FadeOut(warning))
        
        self.wait(2)

    def step_3h_classify_ground(self):
        """Step 3h: Classify colorized points as 'ground'"""
        new_title = Text("CSF Algorithm: Step 8 - Ground Classification Complete", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text(f"Classified {len(self.ground_classified_indices)} points as ground", 
                             font_size=20, color=GREEN)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # Add classification statistics
        stats = Text(
            f"Ground: {len(self.ground_classified_indices)} points\n"
            f"Non-ground: {len(self.non_ground_classified_indices)} points\n"
            f"Threshold: 0.5m",
            font_size=16,
            color=WHITE
        )
        stats.to_corner(UR)
        
        self.play(Write(stats))
        self.classification_stats = stats
        
        self.wait(2)

    def step_3i_remove_non_ground(self):
        """Step 3i: Remove all non-ground points from view"""
        new_title = Text("CSF Algorithm: Step 9 - Remove Non-Ground Points", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("Filter out all non-ground points - keep only ground", 
                             font_size=20, color=RED)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # Identify non-ground dots to remove
        all_dots = [*self.ground_dots, *self.vegetation_dots, *self.building_dots]
        non_ground_dots = [all_dots[idx] for idx in self.non_ground_classified_indices]
        
        # Remove cloth and threshold zone first
        self.play(
            FadeOut(self.cloth_dots, self.cloth_lines, self.threshold_zone),
            run_time=1
        )
        
        # Then remove non-ground points
        if non_ground_dots:
            self.play(FadeOut(*non_ground_dots, lag_ratio=0.01))
        
        self.wait(2)

    def step_3j_invert_back(self):
        """Step 3j: Invert remaining ground points to original position"""
        new_title = Text("CSF Algorithm: Step 10 - Return to Original Orientation", 
                        font_size=36, color=BLUE)
        new_title.to_edge(UP)
        
        new_explanation = Text("Flip ground points back to original positions - Algorithm Complete!", 
                             font_size=20, color=GREEN)
        new_explanation.to_edge(DOWN)
        
        self.play(
            Transform(self.current_title, new_title),
            Transform(self.current_explanation, new_explanation)
        )
        
        # Calculate original positions for remaining ground points
        max_z = self.all_points[:, 2].max()
        min_z = self.all_points[:, 2].min()
        z_center = (max_z + min_z) / 2
        
        # Get remaining dots (ground-classified points)
        all_dots = [*self.ground_dots, *self.vegetation_dots, *self.building_dots]
        remaining_dots = [all_dots[idx] for idx in self.ground_classified_indices]
        
        # Calculate original positions
        original_animations = []
        for idx, dot in zip(self.ground_classified_indices, remaining_dots):
            original_point = self.all_points[idx]
            x, y, z = original_point
            original_pos = self.axes.coords_to_point(x, z)  # Back to original Z
            original_animations.append(dot.animate.move_to(original_pos))
        
        # Animate return to original orientation
        if original_animations:
            self.play(*original_animations, run_time=3)
        
        # Final result summary
        final_stats = Text(
            f"CSF Algorithm Complete!\n"
            f"Ground points detected: {len(self.ground_classified_indices)}\n"
            f"Success: Ground surface identified",
            font_size=18,
            color=GREEN
        )
        final_stats.next_to(self.classification_stats, DOWN, aligned_edge=LEFT)
        
        self.play(Write(final_stats))
        
        self.wait(3)
        
        # Clear for next slide
        self.play(FadeOut(*self.mobjects))

    def slide_4_real_data_pdal(self):
        """Slide 4: Real Data processing with PDAL csf filter"""
        title = Text("Real Data Processing with PDAL", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Show transition from algorithm to real implementation
        transition_text = VGroup(
            Text("From Algorithm to Implementation:", font_size=32, color=YELLOW),
            Text("• CSF implemented in PDAL (Point Data Abstraction Library)", font_size=24, color=WHITE),
            Text("• Production-ready for real LiDAR data", font_size=24, color=WHITE),
            Text("• Configurable parameters for different terrains", font_size=24, color=WHITE)
        )
        transition_text.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        transition_text.center()
        
        for item in transition_text:
            self.play(Write(item))
            self.wait(0.5)
        
        self.wait(2)
        self.play(FadeOut(title, transition_text))

    def slide_5_pdal_pipeline(self):
        """Slide 5: Show PDAL processing pipeline"""
        title = Text("PDAL Processing Pipeline", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Create PDAL pipeline visualization
        pipeline_json = {
            "pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {
                    "type": "filters.csf",
                    "resolution": 0.5,
                    "threshold": 0.01,
                    "rigidness": 2,
                    "iterations": 500
                },
                {"type": "writers.las", "filename": "output_ground.las"}
            ]
        }
        
        pipeline_text = Text(
            json.dumps(pipeline_json, indent=2),
            font_size=14,
            color=WHITE,
            font="monospace"
        )
        pipeline_text.center()
        
        self.play(Write(pipeline_text))
        self.wait(3)
        
        self.play(FadeOut(title, pipeline_text))

    def slide_6_before_after(self):
        """Slide 6: Before/After views of point classification"""
        title = Text("PDAL CSF Results: Before vs After", font_size=48, color=BLUE)
        title.to_edge(UP)

        self.all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        self.play(Write(title))
        
        # Create side-by-side comparison
        before_title = Text("Before CSF Filter: Unclassified", font_size=24, color=WHITE)
        after_title = Text("After CSF Filter: Ground Classified", font_size=24, color=WHITE)
        
        before_title.to_corner(UL, buff=1)
        after_title.to_corner(UR, buff=1)
        
        # Before view - all points same color
        before_ground = VGroup(*[
            Dot([-3 + p[0]*0.5, p[2]*0.5, 0], color=WHITE, radius=0.02)
            for p in self.ground_points[:50]
        ])
        
        before_vegetation = VGroup(*[
            Dot([-3 + p[0]*0.5, p[2]*0.5, 0], color=WHITE, radius=0.02)
            for p in self.vegetation_points[:50]
        ])

        before_building = VGroup(*[
            Dot([-3 + p[0]*0.5, p[2]*0.5, 0], color=WHITE, radius=0.02)
            for p in self.building_points[:50]
        ])
        
        # After view - classified points
        after_ground = VGroup(*[
            Dot([3 + p[0]*0.5, p[2]*0.5, 0], color=GROUND_COLOR, radius=0.02)
            for p in self.ground_points[:50]
        ])
        
        after_vegetation = VGroup(*[
            Dot([3 + p[0]*0.5, p[2]*0.5, 0], color=WHITE, radius=0.02)
            for p in self.vegetation_points[:50]
        ])

        after_building = VGroup(*[
            Dot([3 + p[0]*0.5, p[2]*0.5, 0], color=WHITE, radius=0.02)
            for p in self.building_points[:50]
        ])
        
        self.play(
            Write(before_title), Write(after_title),
            FadeIn(before_ground, lag_ratio=0.02),
            FadeIn(before_vegetation, lag_ratio=0.02),
            FadeIn(before_building, lag_ratio=0.02)
        )
        
        self.wait(1)
        
        self.play(
            FadeIn(after_ground, lag_ratio=0.02),
            FadeIn(after_vegetation, lag_ratio=0.02),
            FadeIn(after_building, lag_ratio=0.02)
        )
        
        self.wait(3)
        self.play(FadeOut(*self.mobjects))

    def slide_7_parameter_tuning(self):
        """Slide 7: Parameter tuning with sliders showing dynamic results"""
        title = Text("CSF Parameter Tuning Effects", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Pre-compute different classification results for different parameter sets
        self.precompute_parameter_results()
        
        # Create parameter sliders
        parameters = ["Resolution", "Threshold", "Rigidness", "Iterations"]
        slider_group = VGroup()
        
        for i, param in enumerate(parameters):
            label = Text(param, font_size=20, color=WHITE)
            track = Rectangle(width=4, height=0.2, color=GRAY)
            handle = Circle(radius=0.15, color=BLUE, fill_opacity=1)
            slider = VGroup(label, track, handle)
            slider.arrange(RIGHT, buff=0.5)
            slider_group.add(slider)
        
        slider_group.arrange(DOWN, buff=0.8)
        slider_group.move_to(LEFT * 3)
        
        result_area = Rectangle(width=6, height=4, color=WHITE, fill_opacity=0.1)
        result_area.to_edge(RIGHT, buff=1)
        
        # Start with config 0
        self.current_result_points = self.create_result_display(0)
        
        self.play(
            Create(slider_group),
            Create(result_area),
            FadeIn(self.current_result_points)
        )
        
        explanation = Text("Parameter adjustments affect ground classification results", 
                        font_size=20, color=YELLOW)
        explanation.to_edge(DOWN)
        self.play(Write(explanation))

        # Define slider positions and corresponding thresholds
        config_settings = [
            {"positions": [0.1, 0.2, 0.3, 0.4], "threshold": 0.05},  # Conservative
            {"positions": [0.3, 0.5, 0.6, 0.7], "threshold": 0.12},  # Moderate
            {"positions": [0.6, 0.8, 0.4, 0.2], "threshold": 0.18},   # Standard  
            {"positions": [0.8, 0.3, 0.7, 0.9], "threshold": 0.25},   # Aggressive
            {"positions": [0.4, 0.6, 0.2, 0.5], "threshold": 0.35}   # Mixed
        ]
        
        for config_idx in range(1, 5):
            # Move sliders based on positions
            slider_animations = []
            for slider_idx, slider in enumerate(slider_group):
                handle = slider[2]
                track = slider[1]
                position_ratio = config_settings[config_idx-1]["positions"][slider_idx]
                new_x = track.get_left()[0] + position_ratio * track.width
                target_pos = [new_x, handle.get_center()[1], 0]
                slider_animations.append(handle.animate.move_to(target_pos))
            
            # Compute classification based on the threshold for this config
            threshold = config_settings[config_idx-1]["threshold"]
            ground_indices = self.compute_classification_for_threshold(threshold)
            
            # Create display with the computed results
            new_result_points = self.create_result_display_from_indices(
                ground_indices, 
                config_settings[config_idx-1], 
                config_idx
            )
            
            self.play(
                *slider_animations,
                Transform(self.current_result_points, new_result_points),
                run_time=2
            )
            self.wait(1)
        self.wait(2)
        self.play(FadeOut(*self.mobjects))

    def compute_classification_for_threshold(self, threshold):
        """Compute ground classification for a specific threshold"""
        ground_indices = []
        for idx, point in enumerate(self.all_points):
            x, y, z = point
            ground_height = 0.3 * np.sin(x * 0.8)
            distance_to_ground = abs(z - ground_height)
            if distance_to_ground <= threshold:
                ground_indices.append(idx)
        return ground_indices

    def create_result_display_from_indices(self, ground_indices, config_settings, config_idx):
        """Create display from computed indices"""
        result_points = VGroup()
        
        config_label = Text(f"Threshold: {config_settings['threshold']}", font_size=16, color=YELLOW)
        config_label.move_to([4, 3, 0])
        result_points.add(config_label)
        
        stats_text = Text(
            f"Ground: {len(ground_indices)} pts\n"
            f"Non-ground: {len(self.all_points) - len(ground_indices)} pts",
            font_size=14, color=WHITE
        )
        stats_text.move_to([4, 2.3, 0])
        result_points.add(stats_text)
        
        # Create point visualization
        subset_size = 60
        for i in range(min(subset_size, len(self.all_points))):
            point = self.all_points[i]
            x, y, z = point
            
            vis_x = 2 + x * 0.3
            vis_y = z * 0.4
            
            if i in ground_indices:
                color = GROUND_COLOR
                radius = 0.025
            else:
                color = WHITE
                radius = 0.02
            
            dot = Dot([vis_x, vis_y, 0], color=color, radius=radius)
            result_points.add(dot)
        
        return result_points

    def precompute_parameter_results(self):
        """Pre-compute classification results for different parameter sets"""
        self.all_points = np.vstack([self.ground_points, self.vegetation_points, self.building_points])
        
        # Use thresholds appropriate for your data range
        # Your data has Z roughly from -1 to 4, so use smaller, more realistic thresholds
        self.parameter_configs = [
            {"threshold": 0.15, "resolution": 0.5, "name": "Very Conservative"},  # Very few ground points
            {"threshold": 0.3, "resolution": 0.4, "name": "Conservative"},       # Some ground points  
            {"threshold": 0.5, "resolution": 0.3, "name": "Standard"},           # Moderate ground points
            {"threshold": 0.7, "resolution": 0.3, "name": "Aggressive"},         # More ground points
            {"threshold": 0.9, "resolution": 0.5, "name": "Very Aggressive"}     # Most ground points
        ]
        
        self.classification_results = []
        
        for config in self.parameter_configs:
            threshold = config["threshold"]
            ground_indices = []
            
            for idx, point in enumerate(self.all_points):
                x, y, z = point
                ground_height = 0.3 * np.sin(x * 0.8)  # This gives values roughly -0.3 to +0.3
                distance_to_ground = abs(z - ground_height)
                
                if distance_to_ground <= threshold:
                    ground_indices.append(idx)
            
            self.classification_results.append(ground_indices)
            print(f"Config {config['name']}: {len(ground_indices)} ground points out of {len(self.all_points)}")

    def create_result_display(self, config_idx):
        """Create point cloud visualization for a specific parameter configuration"""
        if config_idx >= len(self.classification_results):
            config_idx = 0
            
        ground_indices = self.classification_results[config_idx]
        config_name = self.parameter_configs[config_idx]["name"]
        
        # Create points for visualization
        result_points = VGroup()
        
        # Add configuration label
        config_label = Text(f"Config: {config_name}", font_size=16, color=YELLOW)
        config_label.move_to([4, 3, 0])
        result_points.add(config_label)
        
        # Add stats
        stats_text = Text(
            f"Ground: {len(ground_indices)} pts\n"
            f"Non-ground: {len(self.all_points) - len(ground_indices)} pts",
            font_size=14, color=WHITE
        )
        stats_text.move_to([4, 2.3, 0])
        result_points.add(stats_text)
        
        # Create point visualization - FIX: Use actual indices from all_points
        subset_size = 60
        for i in range(min(subset_size, len(self.all_points))):
            point = self.all_points[i]
            x, y, z = point
            
            # Position in result area
            vis_x = 2 + x * 0.3
            vis_y = z * 0.4
            
            # FIX: Check if the actual point index (i) is in ground_indices
            if i in ground_indices:
                color = GROUND_COLOR
                radius = 0.025
            else:
                color = WHITE  # Non-ground points stay white
                radius = 0.02
            
            dot = Dot([vis_x, vis_y, 0], color=color, radius=radius)
            result_points.add(dot)
        
        return result_points

    def slide_8_summary(self):
        """Slide 8: Summary"""
        title = Text("CSF Algorithm Summary", font_size=48, color=BLUE)
        title.to_edge(UP)
        
        summary_points = [
            "✓ Physics-based approach using cloth simulation",
            "✓ Robust ground detection without bias",
            "✓ Handles complex terrain and vegetation",
            "✓ Implemented in production tools (PDAL)",
            "✓ Tunable parameters for different environments"
        ]
        
        self.play(Write(title))
        
        summary_group = VGroup()
        for point in summary_points:
            text = Text(point, font_size=28, color=WHITE)
            summary_group.add(text)
        
        summary_group.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        summary_group.center()
        
        for point in summary_group:
            self.play(Write(point))
            self.wait(0.8)
        
        # Final thank you
        thank_you = Text("Thank You!", font_size=48, color=YELLOW)
        thank_you.to_edge(DOWN)
        
        self.play(Write(thank_you))
        self.wait(3)
        
        # Final fade out
        self.play(FadeOut(*self.mobjects))


# To run this presentation:
# manim -pql csf_manim_animationv**.py CSFPresentation