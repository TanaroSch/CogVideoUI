"""
This script is used to create a Streamlit web application for generating videos using the CogVideoX model.

Run the script using Streamlit:
    $ export OPENAI_API_KEY=your OpenAI Key or ZhiupAI Key
    $ export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/  # using with ZhipuAI, Not using this when using OpenAI
    $ streamlit run web_demo.py
"""

import base64
import json
import os
import time
import gc
from datetime import datetime
from typing import List

import imageio
import numpy as np
import streamlit as st
import torch
from convert_demo import convert_prompt
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

# List of available models
AVAILABLE_MODELS = [
    "THUDM/CogVideoX-5b",
    "THUDM/CogVideoX-2b",
]

# Function to check if models exist locally
def get_local_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    return [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

# Function to download model
def download_model(model_name):
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name.split("/")[-1])
    snapshot_download(repo_id=model_name, local_dir=model_path)
    return model_path

# Load the model at the start
@st.cache_resource
def load_model(model_path: str, dtype: torch.dtype) -> CogVideoXPipeline:
    """
    Load the CogVideoX model.

    Args:
    - model_path (str): Path to the model.
    - dtype (torch.dtype): Data type for model.
    - device (str): Device to load the model on.

    Returns:
    - CogVideoXPipeline: Loaded model pipeline.
    """
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    
    # Set Scheduler
    if "2b" in model_path.lower():
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    else:
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    # Enable CPU offload and tiling
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


# Define a function to generate video based on the provided prompt and model path
def generate_video(
    pipe: CogVideoXPipeline,
    prompt: str,
    model_path: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    output_path: str = "./output.mp4",
) -> List[np.ndarray]:
    """
    Generate a video based on the provided prompt and model path.

    Args:
    - pipe (CogVideoXPipeline): The pipeline for generating videos.
    - prompt (str): Text prompt for video generation.
    - num_inference_steps (int): Number of inference steps.
    - guidance_scale (float): Guidance scale for generation.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - device (str): Device to run the generation on.
    - dtype (torch.dtype): Data type for the model.

    Returns:
    - List[np.ndarray]: Generated video frames.
    """
    use_dynamic_cfg = "2b" not in model_path.lower()
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=use_dynamic_cfg,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(42),
    ).frames[0]
    
    export_to_video(video, output_path, fps=8)
    return video

def save_metadata(
    prompt: str,
    converted_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_videos_per_prompt: int,
    path: str,
) -> None:
    """
    Save metadata to a JSON file.

    Args:
    - prompt (str): Original prompt.
    - converted_prompt (str): Converted prompt.
    - num_inference_steps (int): Number of inference steps.
    - guidance_scale (float): Guidance scale.
    - num_videos_per_prompt (int): Number of videos per prompt.
    - path (str): Path to save the metadata.
    """
    metadata = {
        "prompt": prompt,
        "converted_prompt": converted_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_videos_per_prompt": num_videos_per_prompt,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

def generate_detailed_prompt(brief_prompt):
    template = """You are an AI assistant specialized in creating detailed, vivid video descriptions from brief prompts. Your task is to take short video concepts and expand them into highly descriptive, cinematic narratives that could guide video generation. Follow these rules:
1. Provide only one detailed video description per user request.
2. When asked for modifications, don't simply add to the existing description. Instead, completely rewrite the description to seamlessly incorporate the new elements.
3. If the user requests a new video concept, disregard any previous conversations and start fresh.
4. Aim for descriptions of about 100-150 words.
5. Focus on visual details, actions, settings, lighting, and atmosphere.
6. Describe the video as a sequence of scenes or shots, capturing the dynamic nature of video.
7. Use vivid, specific language to paint a clear picture in the mind's eye.
Here are three examples of how to transform brief prompts into detailed video descriptions:
User: "A girl is on the beach"
Assistant: A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.
User: "A man jogging on a football field"
Assistant: A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.
User: "A woman is dancing, HD footage, close-up"
Assistant: A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.
Now, please transform this brief concept it into a detailed but very short, cinematic description suitable for video generation:
User: "{}"
Assistant: """
    return template.format(brief_prompt)

def main() -> None:
    st.set_page_config(page_title="CogVideoX-Demo", page_icon="🎥", layout="wide")
    st.write("# CogVideoX 🎥")

    # Check for local models
    local_models = get_local_models()

    with st.sidebar:
        if local_models:
            model_choice = st.selectbox("Select Model", local_models + ["Download new model"])
            if model_choice == "Download new model":
                new_model = st.selectbox("Select model to download", AVAILABLE_MODELS)
                if st.button("Download"):
                    with st.spinner(f"Downloading {new_model}..."):
                        model_path = download_model(new_model)
                        st.success(f"Model downloaded to {model_path}")
                    local_models = get_local_models()
                    model_choice = new_model.split("/")[-1]
            model_path = os.path.join("models", model_choice)
        else:
            st.warning("No local models found.")
            download_model_choice = st.radio("Do you want to download a model?", ["Yes", "No"], index=1)
            if download_model_choice == "Yes":
                new_model = st.selectbox("Select model to download", AVAILABLE_MODELS)
                if st.button("Download"):
                    with st.spinner(f"Downloading {new_model}..."):
                        model_path = download_model(new_model)
                        st.success(f"Model downloaded to {model_path}")
                    local_models = get_local_models()
                    model_choice = new_model.split("/")[-1]
                    model_path = os.path.join("models", model_choice)
            else:
                st.error("A model is required to generate videos. Please download a model to continue.")
                st.stop()

        st.info("It will take some time to generate a video (~90 seconds per videos in 50 steps).", icon="ℹ️")
        num_inference_steps: int = st.number_input("Inference Steps", min_value=1, max_value=100, value=50)
        guidance_scale: float = st.number_input("Guidance Scale", min_value=0.0, max_value=20.0, value=6.0)
        num_videos_per_prompt: int = st.number_input("Videos per Prompt", min_value=1, max_value=10, value=1)
        
        # Add slider for native prompt enhancement
        use_native_enhancement = st.slider("Use Native Prompt Enhancement", min_value=0, max_value=1, value=0)

        share_links_container = st.empty()

    # Determine dtype based on model
    dtype = torch.bfloat16 if "5b" in model_path.lower() else torch.float16

    global pipe
    pipe = load_model(model_path, dtype)

    prompt: str = st.text_input("Prompt")

    col1, col2 = st.columns(2)
    with col1:
        generate_button = st.button("Generate Video")
    with col2:
        detailed_prompt_button = st.button("Generate Detailed Prompt")

    if generate_button and prompt:
        # Modify the video generation process to use native enhancement only if selected
        with st.spinner("Refining prompts..."):
            if use_native_enhancement:
                converted_prompt = convert_prompt(prompt=prompt, retry_times=1)
                if converted_prompt is None:
                    st.error("Failed to refine the prompt. Using the original one.")
                    converted_prompt = prompt
            else:
                converted_prompt = prompt

        st.info(f"**Original prompt:**  \n{prompt}  \n  \n**Converted prompt:**  \n{converted_prompt}")
        torch.cuda.empty_cache()

        with st.spinner("Generating Video..."):
            start_time = time.time()
            video_paths = []

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./output/{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            metadata_path = os.path.join(output_dir, "config.json")
            save_metadata(
                prompt, converted_prompt, num_inference_steps, guidance_scale, num_videos_per_prompt, metadata_path
            )

            for i in range(num_videos_per_prompt):
                video_path = os.path.join(output_dir, f"output_{i + 1}.mp4")

                video = generate_video(
                    pipe, converted_prompt, model_path, num_inference_steps, guidance_scale, 1, video_path
                )
                video_paths.append(video_path)
                with open(video_path, "rb") as video_file:
                    video_bytes: bytes = video_file.read()
                    st.video(video_bytes, autoplay=True, loop=True, format="video/mp4")
                
                # Clear CUDA cache and force garbage collection after each video
                torch.cuda.empty_cache()
                gc.collect()

            used_time: float = time.time() - start_time
            st.success(f"Videos generated in {used_time:.2f} seconds.")

            # Create download links in the sidebar
            with share_links_container:
                st.sidebar.write("### Download Links:")
                for video_path in video_paths:
                    video_name = os.path.basename(video_path)
                    with open(video_path, "rb") as f:
                        video_bytes: bytes = f.read()
                    b64_video = base64.b64encode(video_bytes).decode()
                    href = f'<a href="data:video/mp4;base64,{b64_video}" download="{video_name}">Download {video_name}</a>'
                    st.sidebar.markdown(href, unsafe_allow_html=True)

        # After all videos are generated, move the pipe to CPU and clear GPU memory
        pipe.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    if detailed_prompt_button and prompt:
        detailed_prompt = generate_detailed_prompt(prompt)
        st.text_area("Detailed Prompt (Copy this)", detailed_prompt, height=400)

if __name__ == "__main__":
    main()