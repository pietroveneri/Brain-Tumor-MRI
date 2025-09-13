import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from typing import Tuple, Optional, Union

class ScaleBar:
    """
    A class to add scale bars to images with flexible positioning and styling options.
    """
    
    def __init__(self, image_path: str, output_path: Optional[str] = None):
        """
        Initialize the ScaleBar with an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path for the output image (optional)
        """
        self.image_path = image_path
        self.output_path = output_path or self._generate_output_path()
        self.img = self._load_image()
        
    def _load_image(self) -> np.ndarray:
        """Load the image and handle errors."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
        return img
    
    def _generate_output_path(self) -> str:
        """Generate output path based on input path."""
        base, ext = os.path.splitext(self.image_path)
        return f"{base}_with_scalebar{ext}"
    
    def add_scale_bar(self, 
                     scale_length: float,
                     scale_unit: str = "cm",
                     pixel_to_unit_ratio: Optional[float] = None,
                     position: str = "bottom_left",
                     bar_color: Tuple[int, int, int] = (255, 255, 255),
                     text_color: Tuple[int, int, int] = (255, 255, 255),
                     bar_thickness: int = 2,
                     font_scale: float = 0.5,
                     font_thickness: int = 1,
                     margin: int = 20) -> str:
        """
        Add a scale bar to the image.
        
        Args:
            scale_length: Length of the scale bar in the specified unit
            scale_unit: Unit for the scale bar (e.g., "cm", "mm", "μm", "px")
            pixel_to_unit_ratio: Pixels per unit (if None, will be estimated)
            position: Position of scale bar ("bottom_left", "bottom_right", "top_left", "top_right")
            bar_color: RGB color of the scale bar
            text_color: RGB color of the text
            bar_thickness: Thickness of the scale bar line
            font_scale: Scale of the font
            font_thickness: Thickness of the font
            margin: Margin from edges in pixels
            
        Returns:
            Path to the output image
        """
        # Calculate pixel length of scale bar
        if pixel_to_unit_ratio is None:
            pixel_to_unit_ratio = self._estimate_pixel_ratio()
        
        bar_length_pixels = int(scale_length * pixel_to_unit_ratio)
        
        # Get position coordinates
        start_point, text_position = self._get_positions(position, bar_length_pixels, margin)
        end_point = (start_point[0] + bar_length_pixels, start_point[1])
        
        # Create a copy of the image
        img_with_scalebar = self.img.copy()
        
        # Draw the scale bar
        cv2.line(img_with_scalebar, start_point, end_point, bar_color, bar_thickness)
        
        # Add text
        scale_text = f"{scale_length} {scale_unit}"
        cv2.putText(img_with_scalebar, scale_text, text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 
                   font_thickness, cv2.LINE_AA)
        
        # Save the image
        cv2.imwrite(self.output_path, img_with_scalebar)
        print(f"Scale bar added successfully! Output saved to: {self.output_path}")
        
        return self.output_path
    
    def _estimate_pixel_ratio(self) -> float:
        """
        Estimate pixels per unit based on image dimensions.
        This is a rough estimation - for accurate results, provide pixel_to_unit_ratio.
        """
        # Assume a reasonable default for medical images (roughly 0.1mm per pixel)
        # This should be adjusted based on your specific imaging setup
        return 10.0  # 10 pixels per mm
    
    def _get_positions(self, position: str, bar_length: int, margin: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get start point and text position based on the specified position."""
        h, w = self.img.shape[:2]
        
        if position == "bottom_left":
            start_point = (margin, h - margin)
            text_position = (margin, h - margin - 10)
        elif position == "bottom_right":
            start_point = (w - margin - bar_length, h - margin)
            text_position = (w - margin - bar_length, h - margin - 10)
        elif position == "top_left":
            start_point = (margin, margin)
            text_position = (margin, margin + 20)
        elif position == "top_right":
            start_point = (w - margin - bar_length, margin)
            text_position = (w - margin - bar_length, margin + 20)
        else:
            raise ValueError(f"Invalid position: {position}. Use 'bottom_left', 'bottom_right', 'top_left', or 'top_right'")
        
        return start_point, text_position
    
    def add_scale_bar_with_known_dimensions(self, 
                                          image_width_mm: float,
                                          image_height_mm: float,
                                          scale_length: float,
                                          scale_unit: str = "mm",
                                          position: str = "bottom_left",
                                          **kwargs) -> str:
        """
        Add scale bar when you know the actual dimensions of the image.
        
        Args:
            image_width_mm: Actual width of the image in mm
            image_height_mm: Actual height of the image in mm
            scale_length: Length of the scale bar in the specified unit
            scale_unit: Unit for the scale bar
            position: Position of scale bar
            **kwargs: Additional arguments passed to add_scale_bar
            
        Returns:
            Path to the output image
        """
        h, w = self.img.shape[:2]
        
        # Calculate pixel to unit ratio
        pixel_to_unit_ratio = w / image_width_mm
        
        return self.add_scale_bar(
            scale_length=scale_length,
            scale_unit=scale_unit,
            pixel_to_unit_ratio=pixel_to_unit_ratio,
            position=position,
            **kwargs
        )


def main():
    """
    Example usage of the ScaleBar class.
    """
    # Example 1: Basic usage with estimated scale
    try:
        scalebar = ScaleBar("cs-119589-Figure7.png")
        scalebar.add_scale_bar(
            scale_length=1.0,
            scale_unit="cm",
            position="bottom_left"
        )
    except FileNotFoundError:
        print("Example image 'cs-119589-Figure5.png' not found. Please provide a valid image path.")
        return
    
    # Example 2: Usage with known image dimensions
    # scalebar = ScaleBar("your_image.png")
    # scalebar.add_scale_bar_with_known_dimensions(
    #     image_width_mm=200,  # Image is 200mm wide
    #     image_height_mm=200,  # Image is 200mm tall
    #     scale_length=10,
    #     scale_unit="mm",
    #     position="bottom_right"
    # )
    
    # Example 3: Custom styling
    # scalebar = ScaleBar("your_image.png")
    # scalebar.add_scale_bar(
    #     scale_length=5.0,
    #     scale_unit="mm",
    #     pixel_to_unit_ratio=2.5,  # 2.5 pixels per mm
    #     position="top_right",
    #     bar_color=(0, 255, 0),  # Green scale bar
    #     text_color=(0, 255, 0),  # Green text
    #     bar_thickness=3,
    #     font_scale=0.7
    # )


if __name__ == "__main__":
    main()