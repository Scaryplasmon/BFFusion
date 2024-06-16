### BFFusion (❁´◡`❁)
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/7804f293-ac68-499d-9459-0fec5a949c30" height="256">

## Integrating diffusers Functionalities within Blender

<video src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/120d83ee-7567-483a-ae0b-4c79310c6a58" width="400" controls></video>
<video src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/bf9840dc-7b47-4dfc-8efd-7d9ffd57c4f8" width="400" controls></video>
Inference Time

- Less than 1 second

## Features

<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/3f5ca57d-f7b0-43e1-9245-22876df00290" width="400">

### BFFusion (●'◡'●)

#### Blender Scene
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/30ceb5cc-85d5-4d0d-bfff-e718fec2beb0" width="400">

#### Img2Img
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/89756170-4e4e-4f75-9632-cf97b549f65c" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/4421f19a-f94e-4e5d-95c6-74bb07875248" width="400">

#### ctrl_net:

-canny / -depth / -normal

<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/8dd29056-f63e-4bc0-90d5-cd0269165e67" width="250">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/d9b99a33-96f3-426f-90d3-4f538eec2386" width="250">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/a9df95f9-b07d-4d8d-ae78-5bcde74753aa" width="250">

++

### GPFFusion (❁´◡`❁)

#### Render
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/9d7ec03d-3325-4b7c-930f-3d0a6d1c6643" width="400">

#### outputs with different Loras and combinations
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/e0dbe95b-1524-4236-94e8-f91c3fc1a961" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/dfb27e46-dbae-4499-bc04-98afb0afeb33" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/cfef7527-b422-489b-a251-5ea5ea8cfb8c" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/14b1a3d2-2efb-4550-92ee-54b293aae12a" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/2b562d6e-ff6b-4251-aa51-7a4bb402e62c" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/d66da0a4-ab5c-4668-90d5-fa7202e17d00" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/19fd4a08-5eb4-422b-a24d-dde7a0e7dc32" width="400">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/9df82af0-fb62-4868-b2e2-4766005cb5df" width="400">

#### Render
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/f3d863f5-9b17-4d58-9f0b-b03828a0765a" width="200">

#### outputs with different Loras and combinations
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/14c10029-c8a2-491c-a780-9b9d5bbbe851" width="250">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/4be9c7fb-85d9-470c-aa72-4be02f7d44bd" width="250">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/d2db1fe9-a89c-47f7-beb0-8444c13ceea3" width="250">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/91521ca4-69a3-40fb-b30e-ff51b322d4bb" width="250">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/e5ed91bb-da46-49e6-9fb7-2be16db0eb39" width="250">
<img src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/4fc8cffd-efe5-4673-9cdc-149e35178ac1" width="250">

### AnimFFusion (❁´◡`❁)
The addon supports also video generation from video inputs, but it´s still limited and currently in development.

<video src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/6835dc62-6916-4320-b649-f5f1164ccfea" width="400" controls></video>
<video src="https://github.com/Scaryplasmon/BFFusion/assets/90010990/02cd8374-ac4c-4c7c-9996-264f646dc1cb" width="400"></video>

## Setup

1. **Download Blender:**
   - Download Blender from the [official website](https://www.blender.org/download/lts/3-6/).
   - Use the zip version.

2. **Prepare Blender:**
   - Rename the downloaded folder to your preferred name and unzip it.
   - Navigate to the `python` folder within the unzipped directory and delete it.

3. **Python Installation:**
   - Ensure you have Python 3.10 installed on your device.

4. **Install CUDA:**
   - Download and install CUDA 12.1 (or newer) from the [NVIDIA CUDA Archive](https://developer.nvidia.com/cuda-12-1-0-download-archive).

5. **Install Dependencies:**
   - Open a terminal and navigate to your Blender folder.
   - Run the following command to install the required Python packages:
     ```sh
     pip install -r requirements.txt
     ```

6. **Install PyTorch:**
   - Visit the [PyTorch website](https://pytorch.org/).
   - Select and install the PyTorch version compatible with CUDA 12.1 (tested version 2.2).

7. **Start Blender:**
   - Launch Blender from the executable within the unzipped folder.

8. **Install the Addon:**
   - Either install the Python script from the Addon Preferences menu in Blender or run it as a script directly. Both methods should work.

You're now ready to use BFFusion!

## Performance
- Timing based on an NVIDIA RTX 4070 12GB GPU

## Integrated using the diffusers docs
https://huggingface.co/docs/diffusers/v0.27.2/en/index
