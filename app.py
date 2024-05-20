import json
import gradio as gr
import cv2
###########################################
#### Comment for local testing ###########
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis

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
###########################################

with open("stylesMale.json", "r") as file:
    data = json.load(file)
    male_styles_raw = [
        {
            "image": item["image"],
            "title": item["title"],
            "prompt": item["prompt"],
        }
        for item in data
    ]

with open("stylesFemale.json", "r") as file:
    data = json.load(file)
    female_styles_raw = [
        {
            "image": item["image"],
            "title": item["title"],
            "prompt": item["prompt"],
        }
        for item in data
    ]

with gr.Blocks(css="custom.css") as demo:

    with gr.Row(elem_id=""):
        gr.Image("./static/aws_logo.png", height=100, width=80, show_download_button=False, show_label=False, scale=0)
        gr.Markdown(
        """
        # AWS HK Summit 2024 - Pixel Perfection by Stable Diffusion
        This demo showcases Stable Diffusion with IP Adapter FaceID model, which generates images based on a given prompt.
        """)

    selected_gender = gr.State([])
    selected_style = gr.State([])
    styles_raw = male_styles_raw

    with gr.Row(elem_id=""):
        with gr.Column(scale=1, elem_id="box_column"):
            with gr.Group(elem_id="gallery_box"):
                photo = gr.Image(label="1. Take a picture of yourself", sources=["webcam"], interactive=True, type="pil", height=800)
        with gr.Column(scale=1, elem_id="box_column"):
            with gr.Group(elem_id="gallery_box"):
                gender_radio = gr.Radio(
                    label="2. Select your gender",
                    choices=["Male", "Female"],
                    value="Male",
                    elem_id="gender-radio"
                )
                styles_gallery = gr.Gallery(
                    value=[(item["image"], item["title"]) for item in male_styles_raw],
                    label="3. Pick a style from the gallery",
                    allow_preview=False,
                    columns=3,
                    elem_id="style-gallery",
                    show_share_button=False,
                    height=565
                )
                selected_style_text_box = gr.Textbox(
                    label="Selected Style",
                    lines=4,
                    max_lines=4
                    )
        with gr.Column(scale=1, elem_id="box_column"):
            with gr.Group(elem_id="gallery_box"):
                generate_btn = gr.Button(
                    value="Generate Image",
                    variant="primary"
                )
                result_gallery = gr.Gallery(
                        label="Generated Image", 
                        columns=2, 
                        selected_index=0,
                        object_fit="contain", 
                        elem_id="result-gallery",
                        allow_preview=True,
                        show_share_button=False,
                        show_download_button=True,
                        height=755
                    )
                
    def select_style(selected_state: gr.SelectData):
        return {
            selected_style: styles_raw[selected_state.index],
            selected_style_text_box: f"Title: {styles_raw[selected_state.index]['title']}\nPrompt: {styles_raw[selected_state.index]['prompt']}"
        }
    styles_gallery.select(
        fn=select_style,
        inputs=[],
        outputs=[selected_style, selected_style_text_box],
        queue=False,
        show_progress=True,        
    )

    def update_styles(gender):
        global styles_raw
        if gender == "Male":
            styles_raw = male_styles_raw
        else:
            styles_raw = female_styles_raw
        return [(item["image"], item["title"]) for item in styles_raw]
    gender_radio.change(
        fn=update_styles,
        inputs=[gender_radio],
        outputs=[styles_gallery]
    )

    def generate_image(photo: gr.Image, style, progress=gr.Progress(track_tqdm=True)):
        prompt = style['prompt']
        print("[Generate] Prompt: ", prompt)
        negative_prompt = "naked, bikini, skimpy, scanty, bare skin, lingerie, swimsuit, exposed, see-through, nsfw, nudity, porn, adult, explicit, sexual, erotic, nude, naked, lingerie, underwear, panties, bra, nipples, genitals, cleavage, suggestive, provocative, fetish, bondage, xxx, hentai"
        print("[INFO] Negative Prompt: ", negative_prompt)
        if photo is None:
            return "Please select an image in step 1 first.", None
        temp_img_path = "./temp/image.jpg"
        photo.save(temp_img_path)
        
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
        images = ip_model.generate(
            prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding, width=480, height=720, num_inference_steps=30
        )
        print("[INFO] Generated")
        return images
    generate_btn.click(
        fn=generate_image,
        inputs=[photo, selected_style],
        outputs=[result_gallery],
    )

demo.launch(share=False)