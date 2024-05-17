import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import gradio as gr
import cv2
import json

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = hf_hub_download(repo_id='h94/IP-Adapter-FaceID', filename="ip-adapter-faceid_sd15.bin", repo_type="model")

device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
)

ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

def generate_image(selected_state: gr.SelectData, image: gr.Image, gr_styles: gr.State, progress=gr.Progress(track_tqdm=True)):
    prompt = gr_styles[selected_state.index]["prompt"]
    negative_prompt = "naked, bikini, skimpy, scanty, bare skin, lingerie, swimsuit, exposed, see-through"
    print("[INFO] Prompt: ", prompt)
    print("[INFO] Negative Prompt: ", prompt)

    temp_img_path = "./temp/image.jpg"
    image.save(temp_img_path)

    pipe.to(device)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faceid_all_embeds = []
    face = cv2.imread(temp_img_path)
    faces = app.get(face)
    faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    faceid_all_embeds.append(faceid_embed)
    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)
    
    print("[INFO] Generating image...")
    image = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding, width=480, height=720, num_inference_steps=30
    )
    print("[INFO] Generated")
    return image


with open("styles.json", "r") as file:
    data = json.load(file)
    styles_raw = [
        {
            "image": item["image"],
            "title": item["title"],
            "prompt": item["prompt"],
        }
        for item in data
    ]

with gr.Blocks(css="custom.css") as demo:
    gr.Image("./static/aws_logo.png", height=100, width=100, show_download_button=False, show_label=False)
    gr.Markdown(
    """
    # AWS HK Summit 2024 - IP Adapter FaceID Demo
    This demo showcases the IP Adapter FaceID model, which generates images based on a given prompt.
    """)
    with gr.Row(elem_id="main_app"):
        gr_styles = gr.State(value=styles_raw)
        with gr.Column(scale=12, elem_id="box_column"):
            with gr.Group(elem_id="gallery_box"):
                photo = gr.Image(label="1. Take a picture of yourself", sources=["webcam"], interactive=True, type="pil", height=500)
                style_gallery = gr.Gallery(
                    value=[(item["image"], item["title"]) for item in styles_raw],
                    label="2. Pick a style from the gallery",
                    allow_preview=False,
                    columns=4,
                    elem_id="style-gallery",
                    show_share_button=False,
                    height=550
                )
        with gr.Column(scale=12, elem_id="box_column"):
            result_gallery = gr.Gallery(
                    label="Generated Image", 
                    columns=2, 
                    selected_index=0,
                    object_fit="contain", 
                    elem_id="result-gallery",
                    allow_preview=True,
                    show_share_button=False,
                    show_download_button=True,
                    height=1050
                )

        style_gallery.select(
            fn=generate_image,
            inputs=[photo, gr_styles],
            outputs=[result_gallery],
            queue=False,
            show_progress=True,        
            )
            
demo.launch(share=True)

