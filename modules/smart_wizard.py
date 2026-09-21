# Smart Selection Wizard for TEM Image Analysis
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches

class SmartSelectionWizard:
    def __init__(self, raw_image):
        self.raw_image = raw_image
        self.display_image = raw_image.copy()
        self.steps = [
            {"name": "Particle Area", "color": (0, 100, 255), "type": "rectangle", "instruction": "Step 1/4: Particle Analysis Area"},
            {"name": "Scale Bar Line", "color": (100, 255, 255), "type": "line", "instruction": "Step 2/4: Scale Bar (2 points)"},
            {"name": "Minimum Size", "color": (255, 100, 100), "type": "circle", "instruction": "Step 3/4: Minimum Particle (drag circle)"},
            {"name": "Advanced Options", "color": (200, 100, 255), "type": "checkbox", "instruction": "Step 4/4: Advanced Config"},
        ]
        self.current_step = 0
        self.selections = {}

        # Mouse interaction variables
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_image = None
        self.circle_center = None
        self.circle_radius = 0

        # SAM parameter settings (FIXED - No user interaction)
        self.sam_params = {
            "cost_time": 1,
            "data_quality": 2,
            "hardware_spec": 1
        }

        # Advanced options (checkboxes)
        self.advanced_options = {
            "small_particles": False,
            "gpu_quality": "medium",
            "high_noise": False,
            "analysis_types": ["size", "distribution", "shape"]
        }

        # Clickable checkbox regions (x1, y1, x2, y2, action_key)
        self.checkbox_regions = []
        self.checkbox_window_name = "Advanced Configuration"

    def reset_temp_image(self):
        """Reset temporary image for drawing"""
        self.temp_image = self.display_image.copy()

    def draw_all_selections(self, image):
        """Draw all completed selections on the image"""
        result_image = image.copy()

        for i, step in enumerate(self.steps):
            step_name = step["name"]
            step_type = step["type"]
            color = step["color"]

            if step_name not in self.selections:
                continue

            selection = self.selections[step_name]

            if step_type == "rectangle" and len(selection) == 2:
                pt1, pt2 = selection
                cv2.rectangle(result_image, pt1, pt2, color, 3)

            elif step_type == "line" and len(selection) == 2:
                pt1, pt2 = selection
                cv2.line(result_image, pt1, pt2, color, 4)

            elif step_type == "circle" and len(selection) == 2:
                center, radius = selection
                cv2.circle(result_image, center, radius, color, 3)

        return result_image

    def draw_instruction_overlay(self, image):
        """Return image as-is (no overlay). Title is set separately via window title."""
        return image.copy()

    def get_window_title(self):
        """Get window title string for current step"""
        if self.current_step < len(self.steps):
            instruction = self.steps[self.current_step]["instruction"]
            return f"Smart Selection Wizard - {instruction} - [SPACE: Done | BACKSPACE: Back | ESC: Cancel]"
        else:
            return "Smart Selection Wizard - All steps completed! Press SPACE to finish"

    def mouse_callback(self, event, x, y, flags, param):
        """Unified mouse callback for all selection types"""
        if self.current_step >= len(self.steps):
            return

        step = self.steps[self.current_step]
        step_type = step["type"]
        color = step["color"]

        if step_type == "rectangle":
            self.handle_rectangle_selection(event, x, y, color)
        elif step_type == "line":
            self.handle_line_selection(event, x, y, color)
        elif step_type == "circle":
            self.handle_circle_selection(event, x, y, color)
        elif step_type == "checkbox":
            pass

    def handle_rectangle_selection(self, event, x, y, color):
        """Handle rectangle selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.reset_temp_image()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = (x, y)
                temp_img = self.temp_image.copy()
                cv2.rectangle(temp_img, self.start_point, self.current_point, color, 3)
                display = self.draw_instruction_overlay(temp_img)
                cv2.imshow("Smart Selection Wizard", display)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_point = (x, y)
            step_name = self.steps[self.current_step]["name"]
            self.selections[step_name] = [self.start_point, self.current_point]
            self.display_image = self.draw_all_selections(self.raw_image.copy())

    def handle_line_selection(self, event, x, y, color):
        """Handle line selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not hasattr(self, 'line_points'):
                self.line_points = []

            self.line_points.append((x, y))

            if len(self.line_points) == 1:
                temp_img = self.draw_all_selections(self.display_image)
                cv2.circle(temp_img, (x, y), 8, color, -1)
                display = self.draw_instruction_overlay(temp_img)
                cv2.imshow("Smart Selection Wizard", display)

            elif len(self.line_points) == 2:
                step_name = self.steps[self.current_step]["name"]
                self.selections[step_name] = self.line_points.copy()
                self.display_image = self.draw_all_selections(self.raw_image.copy())
                del self.line_points

        elif event == cv2.EVENT_MOUSEMOVE:
            if hasattr(self, 'line_points') and len(self.line_points) == 1:
                temp_img = self.draw_all_selections(self.display_image)
                cv2.line(temp_img, self.line_points[0], (x, y), color, 4)
                display = self.draw_instruction_overlay(temp_img)
                cv2.imshow("Smart Selection Wizard", display)

    def handle_circle_selection(self, event, x, y, color):
        """Handle circle selection for minimum particle size"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.circle_center = (x, y)
            self.circle_radius = 0
            self.reset_temp_image()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.circle_center:
                dx = x - self.circle_center[0]
                dy = y - self.circle_center[1]
                self.circle_radius = int(np.sqrt(dx**2 + dy**2))
                temp_img = self.temp_image.copy()
                cv2.circle(temp_img, self.circle_center, self.circle_radius, color, 3)
                display = self.draw_instruction_overlay(temp_img)
                cv2.imshow("Smart Selection Wizard", display)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.circle_center and self.circle_radius > 0:
                step_name = self.steps[self.current_step]["name"]
                self.selections[step_name] = [self.circle_center, self.circle_radius]
                self.display_image = self.draw_all_selections(self.raw_image.copy())

    def create_checkbox_ui_window(self):
        """Create checkbox selection UI window with clickable regions"""
        ui_height = 600
        ui_width = 800
        checkbox_image = np.zeros((ui_height, ui_width, 3), dtype=np.uint8)

        self.checkbox_regions = []

        cv2.putText(checkbox_image, "Advanced Configuration",
                   (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(checkbox_image, "Click checkboxes or press keys [1-6], then SPACE to finish",
                   (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Option 1: Small Particles
        y_start = 120
        cv2.putText(checkbox_image, "1. Small Particle Detection:",
                   (50, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        checkbox_checked = self.advanced_options["small_particles"]
        box_x, box_y = 80, y_start + 15
        box_size = 20
        cv2.rectangle(checkbox_image, (box_x, box_y), (box_x + box_size, box_y + box_size),
                     (0, 255, 0) if checkbox_checked else (100, 100, 100), 2)
        if checkbox_checked:
            cv2.line(checkbox_image, (box_x + 5, box_y + 10), (box_x + 8, box_y + 15), (0, 255, 0), 2)
            cv2.line(checkbox_image, (box_x + 8, box_y + 15), (box_x + 15, box_y + 5), (0, 255, 0), 2)

        color = (0, 255, 0) if checkbox_checked else (150, 150, 150)
        cv2.putText(checkbox_image, "Enable (points_per_side=64 for particles <10nm)",
                   (box_x + box_size + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        self.checkbox_regions.append((box_x, box_y, box_x + box_size, box_y + box_size, 'small_particles'))

        # Option 2: GPU Quality
        y_start = 200
        cv2.putText(checkbox_image, "2. GPU Quality (Press 2 to cycle):",
                   (50, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        gpu_options = {"high": "High (256 batch, 12GB+ VRAM)",
                      "medium": "Medium (128 batch, 8GB VRAM)",
                      "low": "Low (64 batch, 6GB VRAM)"}
        for i, (key, desc) in enumerate(gpu_options.items()):
            y_pos = y_start + 30 + i * 25
            if key == self.advanced_options["gpu_quality"]:
                cv2.rectangle(checkbox_image, (70, y_pos-20), (750, y_pos+5), (0, 100, 0), -1)
                color = (255, 255, 255)
            else:
                color = (150, 150, 150)
            cv2.putText(checkbox_image, f"  {desc}",
                       (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Option 3: High Noise
        y_start = 330
        cv2.putText(checkbox_image, "3. High Noise Mode:",
                   (50, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)

        checkbox_checked = self.advanced_options["high_noise"]
        box_x, box_y = 80, y_start + 15
        box_size = 20
        cv2.rectangle(checkbox_image, (box_x, box_y), (box_x + box_size, box_y + box_size),
                     (0, 255, 0) if checkbox_checked else (100, 100, 100), 2)
        if checkbox_checked:
            cv2.line(checkbox_image, (box_x + 5, box_y + 10), (box_x + 8, box_y + 15), (0, 255, 0), 2)
            cv2.line(checkbox_image, (box_x + 8, box_y + 15), (box_x + 15, box_y + 5), (0, 255, 0), 2)

        color = (0, 255, 0) if checkbox_checked else (150, 150, 150)
        cv2.putText(checkbox_image, "Enable Noise2SR preprocessing (adds 2-5min)",
                   (box_x + box_size + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        self.checkbox_regions.append((box_x, box_y, box_x + box_size, box_y + box_size, 'high_noise'))

        # Option 4-6: Analysis Types
        y_start = 410
        cv2.putText(checkbox_image, "4-6. Analysis Types (at least one required):",
                   (50, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 200), 2)

        analysis_options = [
            ("size", "4", "Projected particle area"),
            ("distribution", "5", "PF-SUI"),
            ("shape", "6", "Projected morphology")
        ]

        for i, (key, num, desc) in enumerate(analysis_options):
            y_pos = y_start + 30 + i * 30

            checkbox_checked = key in self.advanced_options["analysis_types"]
            box_x, box_y = 80, y_pos - 15
            box_size = 20
            cv2.rectangle(checkbox_image, (box_x, box_y), (box_x + box_size, box_y + box_size),
                         (0, 255, 0) if checkbox_checked else (100, 100, 100), 2)
            if checkbox_checked:
                cv2.line(checkbox_image, (box_x + 5, box_y + 10), (box_x + 8, box_y + 15), (0, 255, 0), 2)
                cv2.line(checkbox_image, (box_x + 8, box_y + 15), (box_x + 15, box_y + 5), (0, 255, 0), 2)

            color = (0, 255, 0) if checkbox_checked else (150, 150, 150)
            cv2.putText(checkbox_image, f"{num}. {desc}",
                       (box_x + box_size + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            self.checkbox_regions.append((box_x, box_y, box_x + box_size, box_y + box_size, f'analysis_{key}'))

        instructions = [
            "Click checkboxes or press keys [1-6] to toggle",
            "[SPACE] Finish | [BACKSPACE] Previous Step | [ESC] Cancel"
        ]

        for i, instruction in enumerate(instructions):
            y_pos = 550 + i * 20
            cv2.putText(checkbox_image, instruction, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return checkbox_image

    def handle_checkbox_click(self, x, y):
        """Handle mouse click on checkbox regions"""
        for (x1, y1, x2, y2, action_key) in self.checkbox_regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action_key == 'small_particles':
                    self.advanced_options["small_particles"] = not self.advanced_options["small_particles"]
                    return True
                elif action_key == 'high_noise':
                    self.advanced_options["high_noise"] = not self.advanced_options["high_noise"]
                    return True
                elif action_key.startswith('analysis_'):
                    analysis_type = action_key.replace('analysis_', '')
                    if analysis_type in self.advanced_options["analysis_types"]:
                        self.advanced_options["analysis_types"].remove(analysis_type)
                    else:
                        self.advanced_options["analysis_types"].append(analysis_type)
                    return True
        return False

    def checkbox_mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for checkbox window"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.handle_checkbox_click(x, y):
                checkbox_img = self.create_checkbox_ui_window()
                cv2.imshow(self.checkbox_window_name, checkbox_img)

    def handle_checkbox_key(self, key):
        """Handle checkbox toggle keys"""
        if key == ord('1'):
            self.advanced_options["small_particles"] = not self.advanced_options["small_particles"]
            return True
        elif key == ord('2'):
            gpu_cycle = {"medium": "high", "high": "low", "low": "medium"}
            self.advanced_options["gpu_quality"] = gpu_cycle[self.advanced_options["gpu_quality"]]
            return True
        elif key == ord('3'):
            self.advanced_options["high_noise"] = not self.advanced_options["high_noise"]
            return True
        elif key == ord('4'):
            if "size" in self.advanced_options["analysis_types"]:
                self.advanced_options["analysis_types"].remove("size")
            else:
                self.advanced_options["analysis_types"].append("size")
            return True
        elif key == ord('5'):
            if "distribution" in self.advanced_options["analysis_types"]:
                self.advanced_options["analysis_types"].remove("distribution")
            else:
                self.advanced_options["analysis_types"].append("distribution")
            return True
        elif key == ord('6'):
            if "shape" in self.advanced_options["analysis_types"]:
                self.advanced_options["analysis_types"].remove("shape")
            else:
                self.advanced_options["analysis_types"].append("shape")
            return True
        return False

    def calculate_sam_parameters(self):
        """Calculate SAM parameters based on selections and options"""
        config = {
            "points_per_side": 32,
            "points_per_batch": 256,
            "pred_iou_thresh": 0.95,
            "stability_score_thresh": 0.80,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2
        }

        if self.advanced_options["small_particles"]:
            config["points_per_side"] = 64

        return config

    def show_current_instruction(self):
        """Display current step instruction"""
        if self.current_step < len(self.steps):
            print("\n" + "="*80)
            print(f"  {self.steps[self.current_step]['instruction']}")
            print("  Press SPACE when done | BACKSPACE to go back | ESC to cancel")
            print("="*80)
        else:
            print("\n🎉 All steps completed! Press SPACE to finish")

    def run_wizard(self):
        """Run the complete selection wizard"""
        print("🧙‍♂️ Smart Selection Wizard Started!")

        cv2.namedWindow("Smart Selection Wizard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Selection Wizard", 1400, 900)
        cv2.setMouseCallback("Smart Selection Wizard", self.mouse_callback)

        self.display_image = self.raw_image.copy()
        self.show_current_instruction()

        while True:
            if (self.current_step < len(self.steps) and
                self.steps[self.current_step]["type"] == "checkbox"):

                checkbox_img = self.create_checkbox_ui_window()
                cv2.imshow(self.checkbox_window_name, checkbox_img)
                cv2.setMouseCallback(self.checkbox_window_name, self.checkbox_mouse_callback)

                key = cv2.waitKey(30) & 0xFF

                if key == 27:  # ESC
                    print("❌ Wizard cancelled.")
                    cv2.destroyAllWindows()
                    return None

                elif key == 32:  # SPACE
                    if len(self.advanced_options["analysis_types"]) == 0:
                        print("⚠️  Please select at least one analysis type!")
                        continue
                    cv2.destroyWindow(self.checkbox_window_name)
                    self.current_step += 1
                    break

                elif key == 8:  # BACKSPACE
                    cv2.destroyWindow(self.checkbox_window_name)
                    self.current_step -= 1
                    if self.current_step < 0:
                        self.current_step = 0
                    self.show_current_instruction()
                    continue

                elif self.handle_checkbox_key(key):
                    checkbox_img = self.create_checkbox_ui_window()
                    cv2.imshow(self.checkbox_window_name, checkbox_img)

                continue

            temp_display = self.draw_all_selections(self.display_image)
            display = self.draw_instruction_overlay(temp_display)
            cv2.setWindowTitle("Smart Selection Wizard", self.get_window_title())
            cv2.imshow("Smart Selection Wizard", display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                print("❌ Wizard cancelled.")
                cv2.destroyAllWindows()
                return None

            elif key == 32:  # SPACE
                if self.current_step >= len(self.steps):
                    break

                step_name = self.steps[self.current_step]["name"]

                if step_name not in self.selections:
                    print(f"⚠️  Please complete {step_name} selection first!")
                    continue

                self.current_step += 1
                self.show_current_instruction()

                if self.current_step >= len(self.steps):
                    break

            elif key == 8:  # BACKSPACE
                self.current_step -= 1
                if self.current_step < 0:
                    self.current_step = 0
                self.show_current_instruction()

        cv2.destroyAllWindows()

        # Extract results
        particle_coords = self.selections["Particle Area"]
        scale_bar_points = self.selections["Scale Bar Line"]
        min_size_selection = self.selections.get("Minimum Size", [None, 0])

        x1, y1 = particle_coords[0]
        x2, y2 = particle_coords[1]
        particle_part = self.raw_image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]

        pt1, pt2 = scale_bar_points
        scale_bar_length = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

        sam_config = self.calculate_sam_parameters()

        return {
            "particle_part": particle_part,
            "scale_bar_points": scale_bar_points,
            "scale_bar_length": scale_bar_length,
            "minimum_size_radius": min_size_selection[1],
            "sam_config": sam_config,
            "particle_coords": particle_coords,
            "advanced_options": self.advanced_options
        }


def run_smart_selection_wizard(raw_image):
    """Wrapper function to run the wizard"""
    wizard = SmartSelectionWizard(raw_image)
    return wizard.run_wizard()
