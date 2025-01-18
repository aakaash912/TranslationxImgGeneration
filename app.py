import google.generativeai as genai
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Initialize the models without GPU dependencies
genai.configure(api_key="AIzaSyCK-tdqs-O7sMUR938ZZy0kF-2vblNJKDo")
model = genai.GenerativeModel("gemini-1.5-flash-8b")

# Initialize Stable Diffusion for CPU only
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,  # Use float32 for CPU
    safety_checker=None,
    requires_safety_checking=False
)

# Memory optimizations that work on CPU
pipeline.enable_attention_slicing()

def translate_tamil_to_english_gemini(tamil_text):
    try:
        response = model.generate_content(f"Translate this Tamil text to English: {tamil_text}")
        return response.text
    except Exception as e:
        return f"Translation error: {str(e)}"

def generate_image_gemini(prompt):
    try:
        # Reduced size and steps for CPU performance
        image = pipeline(
            prompt,
            num_inference_steps=30,  # Further reduced for CPU
            height=512,              # Smaller size for CPU
            width=512, 
        ).images[0]
        
        return image
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return None

def process_input_gemini(tamil_text):
    if not tamil_text.strip():
        return "Please enter some text", None
        
    try:
        # Translate Tamil to English
        english_translation = translate_tamil_to_english_gemini(tamil_text)
        print(f"Translation completed: {english_translation}")
        
        if english_translation.startswith("Translation error"):
            return english_translation, None
            
        # Generate an image based on the translation
        print("Starting image generation...")
        image = generate_image_gemini(english_translation)
        print("Image generation completed")
        return english_translation, image
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return str(e), None

# Create Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### Tamil to English Translator and Image Generator (CPU Optimized)")
    
    with gr.Row():
        tamil_input = gr.Textbox(
            label="Enter Tamil Text",
            placeholder="Type a scene description in Tamil...",
            lines=3
        )
    
    with gr.Row():
        english_output = gr.Textbox(
            label="English Translation",
            interactive=False
        )
    
    with gr.Row():
        image_output = gr.Image(label="Generated Image")
        
    with gr.Row():
        status_output = gr.Markdown("Status: Ready")
    
    # Button to process
    btn_process = gr.Button("Generate")
    btn_process.click(
        fn=process_input_gemini,
        inputs=[tamil_input],
        outputs=[english_output, image_output]
    )

# Launch the interface
if __name__ == "__main__":
    print("Initializing application...")
    demo.launch()
