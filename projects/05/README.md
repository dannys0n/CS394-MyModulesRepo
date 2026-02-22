# LLamaSharp Unity Scene (Project 05)

Simple Unity scene that runs a local LLama model with LLamaSharp and CUDA.

## Prerequisites

- Windows
- Unity `2022.3.12f1`
- NVIDIA GPU with CUDA 12 support

## Setup

1. Open this project folder in Unity Hub.
2. From the project root, run bat script:
```bat
setup_cuda.bat
```
3. install any model into StreamingAssets/models

4. In Unity, open NuGetForUnity and restore packages if prompted.
5. Confirm these files exist:
   - `Assets/Plugins/x86_64/llama.dll`
   - `Assets/Plugins/x86_64/cudart64_12.dll`
   - `Assets/Plugins/x86_64/cublas64_12.dll`
   - `Assets/Plugins/x86_64/cublasLt64_12.dll`

## Run

1. Open `Assets/Scenes/SampleScene.unity`.
2. Enter Play mode.
3. Use the in-scene calculator UI to send a prompt.

## Troubleshoot
1. if cannot find model, look at Gameobject Calculator/Logic and change model name (SampleScene assumes qwen2.5-3b-instruct-q4_k_m.gguf).
2. if token generation appears unusual, look at Gameobject Calculator/Logic and change Gpu Layer Count (0=CPU)

## Screenshots

### Model running in scene

![Model running screenshot 1](./Screenshot%202026-02-22%20062503.png)

![Model running screenshot 2](./Screenshot%202026-02-22%20062523.png)

![Model running screenshot 3](./Screenshot%202026-02-22%20062543.png)