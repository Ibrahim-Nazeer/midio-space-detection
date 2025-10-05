#app.py
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
logger.info("Loading YOLOv8s model...")
model = YOLO("best.pt")
logger.info(f"Model loaded! Classes: {model.names}")

def predict_image(image):
    """
    Perform object detection on the input image
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Annotated image with detections
    """
    if image is None:
        return None
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Run inference
        results = model(image_bgr, conf=0.25)
        
        # Get annotated image
        annotated = results[0].plot()
        
        # Convert BGR back to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Get detection info
        boxes = results[0].boxes
        detections_text = f"**Detections Found: {len(boxes)}**\n\n"
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detections_text += f"{i+1}. **{class_name}** - Confidence: {confidence:.2%}\n"
        
        if len(boxes) == 0:
            detections_text = "No objects detected. Try adjusting the image or confidence threshold."
        
        return annotated_rgb, detections_text
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return image, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="YOLOv8s Safety Detection") as demo:
    gr.Markdown(
        """
        # 🦺 YOLOv8s Safety Object Detection
        Upload an image or use your webcam to detect safety objects in real-time.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                sources=["upload", "webcam"]
            )
            predict_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
            
            gr.Markdown("### Available Classes")
            classes_text = ", ".join([f"{k}: {v}" for k, v in model.names.items()])
            gr.Markdown(f"```\n{classes_text}\n```")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results", type="numpy")
            output_text = gr.Markdown(label="Detections")
    
    # Examples
    gr.Markdown("### 📸 Example Images")
    gr.Markdown("Upload your own images or use the webcam for real-time detection!")
    
    # Connect the function
    predict_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )
    
    # Auto-detect on image change
    input_image.change(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True
    )