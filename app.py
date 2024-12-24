import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion model
def load_model():
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    pipe.enable_attention_slicing()
    pipe.safety_checker = None
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

# Define the image generation function
def generate_image(prompt, height, width, steps, guidance_scale):
    try:
        # Convert inputs to integers and floats as needed
        height = int(height)
        width = int(width)
        steps = int(steps)
        guidance_scale = float(guidance_scale)
        
        # Generate the image
        result = pipe(
            prompt, 
            height=height, 
            width=width, 
            num_inference_steps=steps, 
            guidance_scale=guidance_scale
        )
        image = result.images[0]
        return image
    except Exception as e:
        return f"An error occurred: {e}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Image Generator")
    
    with gr.Row():
        prompt = gr.Textbox(label="Enter your prompt", value="eg. a cute kitten")
    
    with gr.Row():
        height = gr.Textbox(label="Image Height (e.g., 512)", value="512")
        width = gr.Textbox(label="Image Width (e.g., 512)", value="512")
    
    with gr.Row():
        steps = gr.Textbox(label="Inference Steps (e.g., 50)", value="50")
        guidance_scale = gr.Textbox(label="Guidance Scale (e.g., 7.5)", value="7.5")
    
    generate_button = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Image")
    
    generate_button.click(
        generate_image,
        inputs=[prompt, height, width, steps, guidance_scale],
        outputs=output_image,
    )

# Launch the app
demo.launch()
