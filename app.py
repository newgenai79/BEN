import gradio as gr
import torch
import model
from PIL import Image
import os
from datetime import datetime
import glob

class ImageProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.BEN_Base().to(self.device).eval()
        self.model.loadcheckpoints("./BEN_Base.pth")

    def process_single_image(self, input_image):
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.splitext(os.path.basename(input_image))[0]
        output_dir = os.path.join("output", f"{timestamp}_{image_name}")
        os.makedirs(output_dir, exist_ok=True)

        # Process image with torch.no_grad() for memory efficiency
        image = Image.open(input_image)
        with torch.no_grad():
            mask, foreground = self.model.inference(image)

        # Save outputs
        mask_path = os.path.join(output_dir, "mask.png")
        # foreground_path = os.path.join(output_dir, f"{image_name}.png")
        foreground_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.png")
        
        mask.save(mask_path)
        foreground.save(foreground_path, format="PNG")

        return foreground_path, mask_path

    def process_batch(self, input_path, output_dir="output", simplified_output=False):
        processed_files = []
        
        # Get all image files from the input path
        image_files = []
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        if os.path.isdir(input_path):
            for ext in valid_extensions:
                image_files.extend(glob.glob(os.path.join(input_path, f'*{ext}')))
                image_files.extend(glob.glob(os.path.join(input_path, f'*{ext.upper()}')))
        else:
            return "Error: Input path is not a valid directory"
        
        if not image_files:
            return "No image files found in the specified directory"
        
        # Create base output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            
            # Process image
            image = Image.open(image_path)
            with torch.no_grad():
                mask, foreground = self.model.inference(image)

            if simplified_output:
                # Save only foreground directly in output directory as PNG
                foreground_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.png")
                foreground.save(foreground_path, format="PNG")
                processed_files.append(f"Processed {image_name} -> {foreground_path}")
            else:
                # Create timestamped subdirectory and save both mask and foreground
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_output_dir = os.path.join(output_dir, f"{timestamp}_{os.path.splitext(image_name)[0]}")
                os.makedirs(current_output_dir, exist_ok=True)
                
                mask_path = os.path.join(current_output_dir, "mask.png")
                foreground_path = os.path.join(current_output_dir, f"{os.path.splitext(image_name)[0]}.png")
                
                mask.save(mask_path)
                foreground.save(foreground_path, format="PNG")
                processed_files.append(f"Processed {image_name} -> {current_output_dir}")


        return "\n".join(processed_files)

def create_ui():
    processor = ImageProcessor()

    with gr.Blocks() as app:
        gr.Markdown("# Image Processing Interface")
        
        with gr.Tabs():
            # Tab 1: Single Image Processing
            with gr.Tab("Process Single Image"):
                with gr.Row():
                    input_image = gr.Image(type="filepath", label="Input Image")
                
                with gr.Row():
                    process_btn = gr.Button("Process Image")
                
                with gr.Row():
                    with gr.Column():
                        processed_output = gr.Image(label="Processed Output")
                    with gr.Column():
                        mask_output = gr.Image(label="Mask Output")

                process_btn.click(
                    fn=processor.process_single_image,
                    inputs=[input_image],
                    outputs=[processed_output, mask_output]
                )

            # Tab 2: Batch Processing
            with gr.Tab("Batch Processing"):
                with gr.Row():
                    input_path = gr.Textbox(
                        label="Input Directory Path",
                        placeholder="Enter the path to directory containing images"
                    )
                    output_dir = gr.Textbox(
                        value="output",
                        label="Output Directory"
                    )
                
                with gr.Row():
                    simplified_output = gr.Checkbox(
                        label="Save without mask and subfolders",
                        value=False,
                        info="When checked, saves only foreground images directly in output folder"
                    )
                
                with gr.Row():
                    batch_process_btn = gr.Button("Process Batch")
                
                with gr.Row():
                    output_text = gr.Textbox(
                        label="Processing Results",
                        lines=10
                    )
                
                batch_process_btn.click(
                    fn=processor.process_batch,
                    inputs=[input_path, output_dir, simplified_output],
                    outputs=[output_text]
                )

    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()