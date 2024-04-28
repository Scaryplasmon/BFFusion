bl_info = {
    "name": "BlenderDiffusion",
    "blender": (4, 0, 2),
    "category": "Render",
    "description": "Generate images using Diffusion models within Blender",
    "author": "Your Name",
    "version": (1, 0, 0),
    "location": "Properties > Render"
}
import os
import sys
from io import BytesIO
import requests
import imageio
import bpy
import time
from bpy.props import StringProperty, EnumProperty, BoolProperty, IntProperty
from PIL import Image, ImageOps
from diffusers.utils import load_image, export_to_video
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableVideoDiffusionPipeline,
    MotionAdapter, 
    AnimateDiffVideoToVideoPipeline
)
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler, 
    UniPCMultistepScheduler,
    AutoencoderKL
)

import torch
import numpy as np
import threading
print(torch.cuda.is_available())


# params
pipeline_instance = None

class BlenderDiffusionProperties(bpy.types.PropertyGroup):
    sd15: bpy.props.StringProperty(
        name="sd15",
        description="paste in the model path",
        default="runwayml/stable-diffusion-v1-5"
    ) # type: ignore
    sd15_bool: bpy.props.BoolProperty(
        name="sd15_bool",
        description="paste in the model path",
        default=True
    ) # type: ignore
    ctrl_net: bpy.props.StringProperty(
        name="ctrl_net",
        description="paste in the model path",
        default="None"
    ) # type: ignore
    ctrl_net_bool: bpy.props.BoolProperty(
        name="ctrl_net_bool",
        description="paste in the model path",
        default=False
    ) # type: ignore
    sdxl: bpy.props.StringProperty(
        name="sdxl",
        description="paste in the model path",
        default="None"
    ) # type: ignore
    sdxl_bool: bpy.props.BoolProperty(
        name="sdxl_bool",
        description="paste in the model path",
        default=False
    ) # type: ignore

    lora_id: bpy.props.StringProperty(
        name="lora_id",
        description="paste in the model path",
        default=""
    ) # type: ignore
    FreeU: bpy.props.BoolProperty(
        name="FreeU",
        description="Enable FreeU",
        default=False
    ) # type: ignore
    use_loaded: bpy.props.BoolProperty(
        name="use_loaded",
        description="use_loaded",
        default=False
    ) # type: ignore
    four_seeds: bpy.props.BoolProperty(
        name="four_seeds",
        description="four_seeds",
        default=False
    ) # type: ignore
    diffusion_scheduler: bpy.props.EnumProperty(
        name="Scheduler",
        description="Choose the scheduler",
        items=[('LCM', "LCM", ""),
               ('DDIM', "DDIM", ""),
               ('DDPM', "DDPM", ""),
               ('PNDMS', "PNDMS", ""),
               ('LMS', "LMSDiscrete", ""),
               ('EulerAncestral', "EulerAncestral", ""),
               ('DPMSolverMultistep', "DPMSolverMultistep", ""),
               ('UNIPC', "UniPC", "")]
    ) # type: ignore
    diffusion_device: bpy.props.EnumProperty(
        name="Device",
        description="Choose the device",
        items=[('CUDA', "Cuda", ""),
               ('CPU', "CPU", "")]
    ) # type: ignore
    diffusion_input_image_path: bpy.props.StringProperty(
        name="Input Image Path",
        description="Path to the input image",
        subtype='FILE_PATH'
    ) # type: ignore
    rendered_image_path: bpy.props.StringProperty(
        name="Rendered Image Path",
        description="Path to save and load the rendered images",
        subtype='DIR_PATH',
        default="//rendered_images/"
    ) # type: ignore
    diffusion_prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Enter a prompt",
        default=""
    ) # type: ignore
    diffusion_neg_prompt: bpy.props.StringProperty(
        name="Negative Prompt",
        description="Enter a negative prompt",
        default=""
    ) # type: ignore
    batch_size: bpy.props.IntProperty(
        name="batch_size",
        description="batch_size",
        default=1,
        min=1,
        max=8
    ) # type: ignore
    num_inference_steps: bpy.props.IntProperty(
        name="Num Inference Steps",
        description="Number of inference steps",
        default=20,
        min=1,
        max=100
    ) # type: ignore
    strength: bpy.props.FloatProperty(
        name="Strength",
        description="Strength of the image modification",
        default=0.8,
        min=0.0,
        max=1.0
    ) # type: ignore
    guidance_scale: bpy.props.FloatProperty(
        name="Guidance Scale",
        description="Guidance scale",
        default=6.5,
        min=0.0,
        max=20.0
    ) # type: ignore
    seed: bpy.props.IntProperty(
        name="Seed",
        description="Seed for the random number generator",
        default=12345
    ) # type: ignore
    
class BlenderDiffusionPanel(bpy.types.Panel):
    bl_label = "BlenderDiffusion"
    bl_idname = "RENDER_PT_blender_diffusion"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        scene = context.scene.sd15

        layout.prop(scene, "sd15_bool")
        layout.prop(scene, "sd15")
        layout.prop(scene, "ctrl_net_bool")
        layout.prop(scene, "ctrl_net")
        layout.prop(scene, "sdxl_bool")
        layout.prop(scene, "sdxl")
        layout.prop(scene, "lora_id")
        layout.prop(scene, "FreeU")
        layout.prop(scene, "use_loaded")
        layout.prop(scene, "four_seeds")
        layout.prop(scene, "diffusion_scheduler")
        layout.prop(scene, "diffusion_device")
        layout.prop(scene, "diffusion_input_image_path")
        layout.prop(scene, "rendered_image_path")
        layout.prop(scene, "diffusion_prompt")
        layout.prop(scene, "diffusion_neg_prompt")
        layout.prop(scene, "batch_size")
        layout.prop(scene, "num_inference_steps")
        layout.prop(scene, "strength")
        layout.prop(scene, "guidance_scale")
        layout.prop(scene, "seed")

        layout.operator("wm.blenderdiffusion_generate")
        layout.operator("wm.blenderdiffusion_generate_animation")
        layout.operator("wm.blenderdiffusion_generate_video")
        layout.operator("wm.blenderdiffusion_offload")

      
class BlenderDiffusionGenerateOperator(bpy.types.Operator):
    bl_idname = "wm.blenderdiffusion_generate"
    bl_label = "Generate Image"

    def execute(self, context):
        props = context.scene.sd15
        global pipeline_instance

        context.scene.render.image_settings.file_format = 'PNG'
        print("Rendering Image...")
        
        rendered_image_path = bpy.path.abspath(props.rendered_image_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        rendered_image_file = f"rendered_image_{timestamp}.png"
        full_path = os.path.join(rendered_image_path, rendered_image_file)

        if not os.path.exists(rendered_image_path):
            os.makedirs(rendered_image_path)

        bpy.context.scene.render.filepath = full_path
        bpy.ops.render.render(write_still=True)

        props.diffusion_input_image_path = full_path
        if props.use_loaded == False:
            pipeline_instance = setup_pipeline(props.diffusion_device.lower(), props.diffusion_scheduler, props.sd15, props.sd15_bool, props.ctrl_net, props.ctrl_net_bool, props.sdxl, props.sdxl_bool, props.FreeU, props.lora_id)
        elif pipeline_instance == None:
            pipeline_instance = setup_pipeline(props.diffusion_device.lower(), props.diffusion_scheduler, props.sd15, props.sd15_bool, props.ctrl_net, props.ctrl_net_bool, props.sdxl, props.sdxl_bool, props.FreeU, props.lora_id)
        else:
            print("using loaded")

        print("Generating Image...")

        def generate_in_background():
            global pipeline_instance
            generated_image = generate_image(
                pipe=pipeline_instance,
                device="cuda" if torch.cuda.is_available() else "cpu",
                input_image_path=full_path,
                prompt=props.diffusion_prompt,
                negative_prompt=props.diffusion_neg_prompt,
                num_inference_steps=props.num_inference_steps,
                strength=props.strength,
                guidance_scale=props.guidance_scale,
                batch_size=props.batch_size,
                four_seeds=props.four_seeds,
                seed=props.seed
            )

        threading.Thread(target=generate_in_background).start()

        return {'FINISHED'}
    
class BlenderDiffusionGenerateAnimationOperator(bpy.types.Operator):
    bl_idname = "wm.blenderdiffusion_generate_animation"
    bl_label = "Generate Animation"

    def execute(self, context):
        props = context.scene.sd15
        print("Generating Animation...")
        threading.Thread(target=self.generate_animation_in_background, args=(props, context)).start()
        return {'FINISHED'}

    def generate_animation_in_background(self, props, context):
        try:
            # Assuming generate_animation function is defined elsewhere and works correctly
            generated_animation = generate_animation(
                device="cuda" if torch.cuda.is_available() else "cpu",
                input_image_path=props.diffusion_input_image_path,
                prompt=props.diffusion_prompt,
                negative_prompt=props.diffusion_neg_prompt,
                num_inference_steps=props.num_inference_steps,
                strength=props.strength,
                guidance_scale=props.guidance_scale,
                seed=props.seed
            )
            filepath = os.path.join(bpy.path.abspath(context.scene.render.filepath), "generated_animation.mp4")
            export_to_video(generated_animation, filepath, fps=7)
            print("Animation saved successfully.", filepath)
        except Exception as e:
            print(f"Failed to generate or save animation: {e}")
            self.save_partial_results(generated_animation, context)

    def save_partial_results(self, generated_animation, context):
        output_path = os.path.join(bpy.path.abspath(context.scene.render.filepath), "partial_generated_frames")
        os.makedirs(output_path, exist_ok=True)
        try:
            for i, frame in enumerate(generated_animation):
                frame_path = os.path.join(output_path, f"frame_{i:04d}.png")
                frame.save(frame_path)
            print("Saved partial frames due to error.")
        except Exception as e:
            print(f"Failed to save partial frames: {e}")
            
class BlenderDiffusionGenerateVideoOperator(bpy.types.Operator):
    bl_idname = "wm.blenderdiffusion_generate_video"
    bl_label = "Generate Video"

    def execute(self, context):
        props = context.scene.sd15
        print("Generating Video...")

        def generate_video_in_background():
            generated_video = generate_video(
                device="cuda" if torch.cuda.is_available() else "cpu",
                strength=props.strength,
                seed=props.seed
            )
            try:
                bpy.app.timers.register(lambda: export_to_video(generated_video, "generated.mp4", fps=7))
            except Exception as e:
                print(e)
                export_to_video(generated_video, "generated.mp4", fps=7)

        
        threading.Thread(target=generate_video_in_background).start()

        return {'FINISHED'}
    
class BlenderDiffusionOffloadOperator(bpy.types.Operator):
    bl_idname = "wm.blenderdiffusion_offload"
    bl_label = "Offload Model and Refresh VRAM"

    @staticmethod
    def offload_model():
        torch.cuda.empty_cache()
        
    def execute(self, context):
        self.offload_model()
        self.report({'INFO'}, "Model offloaded and VRAM refreshed.")
        return {'FINISHED'}    
    
def setup_pipeline(device, scheduler, sd15 = "runwayml/stable-diffusion-v1-5", sd15_bool = True, ctrl_net = "", ctrl_net_bool = False,  sdxl = "",  sdxl_bool = False, freeU = False, lora_id = "None"):

    device = "cuda" if torch.cuda.is_available() else "NOcpuPlease"
    pipe = None
    #PIPELINE
    print(device)
    if sd15_bool == True and sd15 != "":
        model_id = sd15
        if ctrl_net_bool == True:
            if ctrl_net != "None":
                controlnet = ControlNetModel.from_pretrained(ctrl_net, torch_dtype=torch.float16, use_safetensors=True)
                pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16).to(device)
                pipe.enable_xformers_memory_efficient_attention()
        else:
            if model_id != "None":          
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to(device)
                pipe.enable_xformers_memory_efficient_attention()
    elif sd15_bool== False and sdxl == True:
        model_id = sdxl
        if model_id != "None":
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
            pipe.enable_xformers_memory_efficient_attention()
        else:
            print("No model provided")
            
    if pipe is None:
        raise ValueError("No valid model configuration provided. Please check your settings.")

    #SCHEDULERs
    if scheduler == 'LCM':
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'UniPC':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPMSolver':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'EulerDiscrete':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'EulerAncestralDiscrete':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'LMSDiscrete':
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'PNDM':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
#LORA
    if lora_id == "None":
        print("No LORA ID provided")
    else:
        pipe.load_lora_weights(lora_id)
        pipe.fuse_lora()
#FREEU
    if freeU == True:
        #tends to saturate the pictures a lot! great for stylized outputs
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    #just making sure xformers kicks in, everyday is a good day for xformers    
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
        pipe.set_vae(vae)
    except Exception as e:
        print(e)

    pipe.enable_xformers_memory_efficient_attention()
    print(pipe)
    return pipe

def generate_image(pipe, device, input_image_path, prompt, negative_prompt, num_inference_steps=20, strength=0.8, guidance_scale=6.5, batch_size=1, four_seeds=False, seed=12345):  

    images= []

    # InputIMAGE
    input_image = Image.open(input_image_path).convert("RGB")
    if four_seeds == True:
        prompt = [prompt + t for t in [", highly realistic", ", artsy", ", trending", ", colorful"]]
        negative_prompt = [negative_prompt + p for p in [", surrealistic", ", bland", ", old", ", B&W"]]
        for i in range(4):
            generator = torch.Generator(device).manual_seed(seed + i)
            with torch.no_grad():
                temp_image = pipe(
                    prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    image=input_image, 
                    strength=strength, num_inference_steps=num_inference_steps, generator=generator, 
                    guidance_scale=guidance_scale,
                    requires_safety_checker= False,
                    num_images_per_prompt=batch_size,
                ).images[0]
            temp_image = ImageOps.flip(temp_image)
            images.append(temp_image)
    elif batch_size > 1:
        for p in range(batch_size):
            prompt = [prompt]
            generator = torch.Generator(device).manual_seed(seed)
            # GenerateIMAGE
            with torch.no_grad():
                batched_image = pipe(
                    prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    image=input_image, 
                    strength=strength, num_inference_steps=num_inference_steps, generator=generator, 
                    guidance_scale=guidance_scale,
                    requires_safety_checker= False,
                    num_images_per_prompt=batch_size,
                ).images[0]
            batched_image = ImageOps.flip(batched_image)
            images.append(batched_image)
    else:
        prompt = [prompt]
        generator = torch.Generator(device).manual_seed(seed)
        # GenerateIMAGE
        with torch.no_grad():
            image = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                image=input_image, 
                strength=strength, num_inference_steps=num_inference_steps, generator=generator, 
                guidance_scale=guidance_scale,
                requires_safety_checker= False,
            ).images[0]
        image = ImageOps.flip(image)
        images.append(image)

    print("Image generated successfully! yay (❁´◡`❁)")
    
    if four_seeds or batch_size > 1:
        image = create_image_grid(images, grid_size=(2, 2))
    else:
        image = images[0]
    schedule_image_update(image)
    return image

def generate_animation(device, input_image_path, prompt, negative_prompt, num_inference_steps=20, strength=0.8, guidance_scale=6.5, seed=12345):

#   # Load the motion adapter
#    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
#    model_id = "stablediffusionapi/realistic-vision-v60"
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    # helper function to load videos
    def load_video(file_path: str):
        images = []

        if file_path.startswith(('http://', 'https://')):
            # If the file_path is a URL
            response = requests.get(file_path)
            response.raise_for_status()
            content = BytesIO(response.content)
            vid = imageio.get_reader(content)
        else:
            # Assuming it's a local file path
            vid = imageio.get_reader(file_path)

        for frame in vid:
            pil_image = Image.fromarray(frame)
            images.append(pil_image)

        return images

    video = load_video(input_image_path)

    try:
        pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors", adapter_name="lcm-lora")
    except:
        print("exception")
    try:
        pipe.load_lora_weights("wangfuyun/AnimateLCM/sd15_lora_beta.safetensors", adapter_name="lcm-lora")
    except:
        print("exception")
        
    # enable memory savings!!!! 
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        video = video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator = torch.Generator(device).manual_seed(seed)
    )
    frames = output.frames[0]
    export_to_video(frames, "animation.mp4", fps=10)
    return frames

def generate_video(device, strength=0.8, seed=12345):

    pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()
    generator = torch.Generator(device).manual_seed(seed)   
    pipe.unet.enable_forward_chunking()
    
    # Load the conditioning image
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
    image = image.resize((1024, 576))
    # GenerateIMAGE
    with torch.no_grad():
        frames = pipe(
            image, 
            decode_chunk_size=2, 
            generator=generator, 
            motion_bucket_id=180, 
            noise_aug_strength=strength).frames[0]
        
    export_to_video(frames, "generated.mp4", fps=7)

    
    print("Video generated successfully! yay (❁´◡`❁)")
    return frames


def update_image_in_blender(image_pil, image_name="generated_image"):
    image_data = np.array(image_pil.convert("RGBA"))
    image_data = image_data / 255.0 
    flat_image_data = image_data.ravel()

    if image_name in bpy.data.images:
        bpy.data.images[image_name].name = "old_" + image_name  # Rename the old image

    image = bpy.data.images.new(name=image_name, width=image_pil.width, height=image_pil.height)
    image.pixels = list(flat_image_data)
    image.update()


    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            for space in area.spaces:
                if space.type == 'IMAGE_EDITOR':
                    space.image = image
    print("Image updated")

def schedule_image_update(image_pil):
    bpy.app.timers.register(lambda: update_image_in_blender(image_pil))

def create_image_grid(images, grid_size=(2, 2)):
    print("Combining images into a grid.")
    width, height = images[0].size
    grid = Image.new('RGB', (width * grid_size[0], height * grid_size[1]))
    for index, image in enumerate(images):
        x = index % grid_size[0] * width
        y = index // grid_size[0] * height
        grid.paste(image, (x, y))
    return grid

def register():
    bpy.utils.register_class(BlenderDiffusionProperties)
    bpy.types.Scene.sd15 = bpy.props.PointerProperty(type=BlenderDiffusionProperties)
    bpy.utils.register_class(BlenderDiffusionPanel)
    bpy.utils.register_class(BlenderDiffusionGenerateOperator)
    bpy.utils.register_class(BlenderDiffusionGenerateAnimationOperator)
    bpy.utils.register_class(BlenderDiffusionGenerateVideoOperator)
    bpy.utils.register_class(BlenderDiffusionOffloadOperator)

def unregister():
    del bpy.types.Scene.sd15
    bpy.utils.unregister_class(BlenderDiffusionProperties)
    bpy.utils.unregister_class(BlenderDiffusionPanel)
    bpy.utils.unregister_class(BlenderDiffusionGenerateOperator)
    bpy.utils.unregister_class(BlenderDiffusionGenerateAnimationOperator)
    bpy.utils.unregister_class(BlenderDiffusionGenerateVideoOperator)
    bpy.utils.unregister_class(BlenderDiffusionOffloadOperator)

if __name__ == "__main__":
    register()
