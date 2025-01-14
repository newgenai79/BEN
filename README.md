# BEN - Background Erase Network (Beta Base Model)

BEN is a deep learning model designed to automatically remove backgrounds from images, producing both a mask and a foreground image. 
- For access to commercial model email at sales@pramadevelopment.com
- Website: https://pramadevelopment.com/
- Follow on X: https://x.com/PramaResearch/


# BEN SOA Benchmarks on Disk 5k Eval

![Demo Results](https://huggingface.co/PramaLLC/BEN/resolve/main/demo.jpg?download=true)


### BEN_Base + BEN_Refiner (commercial model please contact us for more information):
- MAE: 0.0283
- DICE: 0.8976
- IOU: 0.8430
- BER: 0.0542
- ACC: 0.9725

### BEN_Base (94 million parameters):
- MAE: 0.0331
- DICE: 0.8743
- IOU: 0.8301
- BER: 0.0560
- ACC: 0.9700

### MVANet (old SOTA):
- MAE: 0.0353
- DICE: 0.8676
- IOU: 0.8104
- BER: 0.0639
- ACC: 0.9660


### BiRefNet(not tested in house):
- MAE: 0.038


### InSPyReNet (not tested in house):
- MAE: 0.042



## Features
- Background removal from images
- Generates both binary mask and foreground image
- CUDA support for GPU acceleration
- Simple API for easy integration

## Installation
Step 1: Clone the repository
```	
git clone https://github.com/newgenai79/BEN
```

Step 2: Navigate inside the cloned repository
```
cd BEN
```

Step 3: Create virtual environment
```
python -m venv venv
```

Step 4: Activate virtual environment
```
venv\scripts\activate
```

Step 5: Install wheel package
```
pip install wheel
```

Step 6: Install requirements
```	
pip install -r requirements.txt
```

Step 7: Download model
https://huggingface.co/PramaLLC/BEN/resolve/main/BEN_Base.pth?download=true


Step 8: Launch Gradio WebUI
```	
venv\scripts\activate
```

```
python app.py
```