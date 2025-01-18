---
title: TranslationxImgGeneration
emoji: 😻
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Tamil to English Text translation and Image generation
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Tamil to English Image Generator

A multimodal AI application that translates Tamil text descriptions into English and generates corresponding images using Stable Diffusion. The application combines Google's Gemini API for translation and Stable Diffusion for image generation, wrapped in a user-friendly Gradio interface.

## 🌟 Features

- Tamil to English text translation using Google's Gemini API
- Image generation from translated descriptions using Stable Diffusion
- User-friendly web interface
- CPU-optimized performance
- Real-time status updates

## 🔧 Technical Architecture

### Components
1. **Translation Engine**: Google Gemini 1.5 Flash
2. **Image Generation**: Stable Diffusion v1.4
3. **Interface**: Gradio Web UI
4. **Deployment**: Hugging Face Spaces

## 💻 Source Code Breakdown

### Key Components

1. **Model Initialization**
```python
# Initialize Gemini for translation
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash-8b")

# Initialize Stable Diffusion for CPU
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checking=False
)
pipeline.enable_attention_slicing()
```

2. **Translation Function**
```python
def translate_tamil_to_english_gemini(tamil_text):
    try:
        response = model.generate_content(
            f"Translate this Tamil text to English: {tamil_text}"
        )
        return response.text
    except Exception as e:
        return f"Translation error: {str(e)}"
```

3. **Image Generation Function**
```python
def generate_image_gemini(prompt):
    try:
        image = pipeline(
            prompt,
            num_inference_steps=30,
            height=512,
            width=512
        ).images[0]
        return image
    except Exception as e:
        return None
```

## 📦 Installation

1. **Dependencies**
```bash
pip install google-generativeai gradio diffusers torch transformers pillow
```

2. **Environment Variables**
Create a `.env` file with:
```
GOOGLE_API_KEY=your_gemini_api_key
```

## 🚀 Deployment Guide

### Hugging Face Spaces Deployment

1. Create a new Space:
   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Select "Gradio" as the SDK
   - Choose "CPU" as the hardware

2. Upload Files:
   - Upload `app.py` and `requirements.txt`
   - Add your API key in Settings → Secrets → New Secret:
     - Name: `GOOGLE_API_KEY`
     - Value: Your Gemini API key

3. Configuration:
   - Space hardware: CPU
   - Python version: 3.9
   - SDK: Gradio

### Local Deployment
```bash
python app.py
```

## 📘 User Manual

1. **Accessing the Application**
   - Visit your Hugging Face Space URL or local deployment URL

2. **Using the Interface**
   - Enter Tamil text in the input box
   - Click "Generate"
   - View the English translation and generated image

3. **Example Inputs**
```tamil
மலை உச்சியில் பனி மூடிய குடிசை. சுற்றிலும் பச்சை மரங்கள்
```

## 📊 Performance Analysis

### Optimization Decisions

1. **Resolution Selection**
   - Optimal resolution: 512x512
   - Rationale: Base training resolution of Stable Diffusion
   - Balance between quality and performance

2. **Generation Parameters**
   - Inference steps: 30
     - Lower values (<20): Poor quality
     - Higher values (>40): Diminishing returns
   - Guidance scale: Default (7.5)
     - Lower values: Less adherence to prompt
     - Higher values: Increased computation time

3. **CPU Optimizations**
   - Attention slicing enabled
   - Safety checker disabled
   - Float32 precision for CPU compatibility

### Performance Metrics
- Average translation time: ~2-3 seconds
- Average image generation time: 12 to 15 mins (CPU)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache2.0 License

## 🙏 Acknowledgments

- Google Gemini API for translation
- Stable Diffusion by CompVis
- Hugging Face for hosting
- Gradio team for the UI framework
