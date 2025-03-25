import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, Blip2ForConditionalGeneration
import re

class ImprovedYoloBlipColorExtractor:
    def __init__(self, use_blip2=True):
        """YOLO and improved BLIP for object detection and color extraction"""
        # Load YOLO model
        self.yolo_model = YOLO('yolov8m.pt')
        
        # Load BLIP model
        if use_blip2:
            # BLIP-2 has better caption generation capabilities
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
            self.use_blip2 = True
        else:
            # Original BLIP
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")  # Using large model
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            self.use_blip2 = False
            
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.blip_model.to(self.device)
        
        # Define color list with more variations
        self.color_list = [
            # Basic colors
            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 
            'brown', 'black', 'white', 'gray', 'grey', 'silver', 'gold', 
            # Additional colors
            'beige', 'tan', 'navy', 'maroon', 'olive', 'teal', 'violet',
            'turquoise', 'cyan', 'magenta', 'lime', 'indigo', 'crimson',
            # Shades
            'dark blue', 'light blue', 'dark green', 'light green',
            'dark red', 'light red', 'dark brown', 'light brown',
            'dark gray', 'light gray'
        ]
        
        # Special color variations that should be mapped to main colors
        self.color_mapping = {
            'dark blue': 'blue',
            'light blue': 'blue',
            'navy blue': 'blue',
            'dark green': 'green',
            'light green': 'green',
            'dark red': 'red',
            'light red': 'red',
            'dark brown': 'brown',
            'light brown': 'brown',
            'dark gray': 'gray',
            'dark grey': 'gray',
            'light gray': 'gray',
            'light grey': 'gray'
        }

    def detect_object(self, image_path):
        """
        Detect objects using YOLO
        
        Args:
            image_path: Path to image file
            
        Returns:
            cropped_img: Cropped object image
            class_name: Detected object class
            img_rgb: Original RGB image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
        """
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        results = self.yolo_model(img_rgb, conf=0.25)
        
        # If no objects detected
        if len(results[0].boxes) == 0:
            print("No objects detected.")
            return None, None, img_rgb, None
        
        # Select largest bounding box
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_box_idx = np.argmax(areas)
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = boxes[largest_box_idx].astype(int)
        
        # Get object class
        if hasattr(results[0].boxes, 'cls'):
            cls_id = int(results[0].boxes.cls[largest_box_idx].item())
            class_name = results[0].names[cls_id]
        else:
            class_name = "unknown"
        
        # Crop object area
        cropped_img = img_rgb[y1:y2, x1:x2]
        
        return cropped_img, class_name, img_rgb, (x1, y1, x2, y2)

    def generate_caption(self, image, prompt=None):
        """
        Generate image caption using BLIP
        
        Args:
            image: PIL image or numpy array
            prompt: Caption generation prompt
            
        Returns:
            caption: Generated caption text
        """
        # Convert to PIL image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Use different prompts for BLIP and BLIP-2
        if prompt is None:
            if self.use_blip2:
                prompt = "Describe this object in detail, focusing on its color, shape, and material."
            else:
                prompt = "A photo of"
        
        # Process image and generate caption
        if self.use_blip2:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            outputs = self.blip_model.generate(**inputs, max_new_tokens=75)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        else:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            outputs = self.blip_model.generate(**inputs, max_length=75)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # For original BLIP, strip the prompt from the output
            if caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
        
        return caption

    def extract_color_from_caption(self, caption):
        """
        Extract colors from caption
        
        Args:
            caption: Image caption text
            
        Returns:
            detected_colors: List of extracted colors
            raw_matches: Original color matches before mapping
        """
        detected_colors = []
        raw_matches = []
        caption_lower = caption.lower()
        
        # First try to match multi-word colors (e.g., "dark blue")
        for color in self.color_list:
            if ' ' in color:
                pattern = r'\b' + color + r'\b'
                if re.search(pattern, caption_lower):
                    raw_matches.append(color)
                    
                    # Map to main color if needed
                    if color in self.color_mapping:
                        mapped_color = self.color_mapping[color]
                        if mapped_color not in detected_colors:
                            detected_colors.append(mapped_color)
                    else:
                        if color not in detected_colors:
                            detected_colors.append(color)
        
        # Then match single-word colors
        for color in self.color_list:
            if ' ' not in color:
                pattern = r'\b' + color + r'\b'
                if re.search(pattern, caption_lower):
                    raw_matches.append(color)
                    if color not in detected_colors:
                        detected_colors.append(color)
        
        return detected_colors, raw_matches

    def try_multiple_captions(self, image, num_tries=3):
        """
        Generate multiple captions with different prompts and combine color results
        
        Args:
            image: Image to caption
            num_tries: Number of caption attempts with different prompts
            
        Returns:
            best_caption: Best caption from attempts
            all_colors: Combined list of detected colors
        """
        prompts = [
            "Describe this object, especially its color, material, and type.",
            "What color is this object? Describe it in detail.",
            "Describe the appearance of this item, including its color and material."
        ]
        
        all_colors = []
        all_raw_matches = []
        captions = []
        
        # Try multiple prompts
        for i in range(min(num_tries, len(prompts))):
            caption = self.generate_caption(image, prompts[i])
            captions.append(caption)
            
            colors, raw_matches = self.extract_color_from_caption(caption)
            all_raw_matches.extend(raw_matches)
            
            for color in colors:
                if color not in all_colors:
                    all_colors.append(color)
        
        # Find the best caption (the one with the most color mentions)
        best_caption_idx = 0
        most_colors = 0
        
        for i, caption in enumerate(captions):
            _, raw_matches = self.extract_color_from_caption(caption)
            if len(raw_matches) > most_colors:
                most_colors = len(raw_matches)
                best_caption_idx = i
        
        return captions[best_caption_idx], all_colors

    def process_image(self, image_path, output_path=None, show_result=True):
        """
        Process image pipeline: detect object, generate caption, extract colors
        
        Args:
            image_path: Input image path
            output_path: Output image save path
            show_result: Whether to visualize results
            
        Returns:
            main_color: Primary color
            caption: Generated caption
            detected_colors: All detected colors
        """
        # Detect object
        cropped_img, detected_class, orig_img, bbox = self.detect_object(image_path)
        
        if cropped_img is None:
            return None, None, None
        
        # Generate caption with multiple attempts
        caption, detected_colors = self.try_multiple_captions(cropped_img)
        
        # Determine main color (first detected color)
        main_color = detected_colors[0] if detected_colors else "unknown"
        
        # Visualize results
        if show_result:
            plt.figure(figsize=(10, 8))
            
            # Original image
            plt.subplot(2, 2, 1)
            plt.imshow(orig_img)
            plt.title("Original Image")
            plt.axis('off')
            
            # Bounding box display
            plt.subplot(2, 2, 2)
            bbox_img = orig_img.copy()
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.imshow(bbox_img)
            plt.title(f"Detected Object: {detected_class}")
            plt.axis('off')
            
            # Cropped image
            plt.subplot(2, 2, 3)
            plt.imshow(cropped_img)
            plt.title("Cropped Object")
            plt.axis('off')
            
            # Caption and color info
            plt.subplot(2, 2, 4)
            plt.axis('off')
            plt.title("BLIP Analysis Results")
            
            info_text = f"Generated Caption:\n{caption}\n\n"
            info_text += f"Detected Colors: {', '.join(detected_colors) if detected_colors else 'None'}\n\n"
            info_text += f"Main Color: {main_color}"
            
            plt.text(0.1, 0.5, info_text, fontsize=10, va='center', wrap=True)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
            
            plt.show()
        
        return main_color, caption, detected_colors

if __name__ == "__main__":
    # Initialize the extractor
    extractor = ImprovedYoloBlipColorExtractor(use_blip2=True)  # Set to False to use original BLIP
    
    # Test image path
    test_image_path = "test_images/blueumb.jpg"  # Set path to test image
    
    # Process image and extract colors
    main_color, caption, all_colors = extractor.process_image(
        image_path=test_image_path,
        output_path="improved_blip_analysis.jpg",
        show_result=True
    )
    
    print("\n===== Improved BLIP Analysis Results =====")
    print(f"Generated Caption: {caption}")
    print(f"All Detected Colors: {all_colors}")
    print(f"Main Color: {main_color}")