# Setup

This repo uses `uv`. 

Run:
```bash
uv sync
```

This installs stable-audio-tools with Python 3.10, and updates jupyter and tqdm packages (for `tqdm.notebook` progress bars).

## Getting Started

Run the `sao.ipynb` notebook which provides detailed instructions.

### First Run Requirements
- Specify which model to download
- Enter your HuggingFace token when prompted  
- Visit the model's HF page to request access (fill out usage form)
- Links to each model are provided in the notebook

### Using torch.save
Torch.save for .wav files requires torchcodec, we don't need to use torch save if we just view it in jupyter notebook 
using the audio display feature. But its fun to save wav files so I made a shell script `patch_torchcodec_mac.sh` that 
fixes the issue.

#### **To Fix it**
Run **`./patch_torchcodec_mac.sh`** in terminal if there is an execution issue run `chmod +x patch_torchcodec_mac.sh` then run the shell script again.

#### **More Detailed Explanation**
Basically, it's an issue with where torch can't find the ffmpeg module (the thing that writes it into audio). Pretty sure they used conda at Stability, but I use Homebrew and uv, and it wasn't making it easy.

Yes we could have worked around using torch.save() by installing a lib like soundfile and used numpy to work around this which I can show how if needed, but torch.save() is more fun so here is a workaround!

So what we do is we clone the version of ffmpeg that is 7.1.2 and then install it into the .local dir on mac. We then update the torchcodec lib to point to this installation and sign it. Lots of workarounds but yk what? It works.

If you aren't on mac and are still reading this probably just use conda, I might have to figure this out when I run it on my desktop but I hope to not run into the issue. We'll see. (I'll update this if I run into the issue there and make a workaround)