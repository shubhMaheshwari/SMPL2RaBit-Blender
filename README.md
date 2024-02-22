# SMPL2RABIT-Blender 
    - Retarget motion from RGB video RaBit models using SMPL   

## A. Installation

1. Install OS 
    
    #### Ubuntu 
    ```
    TODO
    ```
    
    #### Windows
    1. Install [WSL2](https://www.omgubuntu.co.uk/how-to-install-wsl2-on-windows-10)
    2. Install [CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)  
    3.  

2. Ubuntu Modules
```
sudo apt install unzip cmake
```     

2. Download the codebase
    ```
        git --recursive https://github.com/shubhMaheshwari/SMPL2RaBit-Blender.git
    ```

4. Rabit installation
    <details>
    <summary>  Details </summary>
    1. Clone RaBit Library

    ```
        git clone https://github.com/kulendu/RaBit.git 
        cd RaBit 
    ```
    2. Download model data from [link](https://drive.google.com/file/d/1yvweTYPKtmuMt5Eu7CHZ4-Do4CRYLFtp/view?usp=sharing) to `<HOME_PATH>/RaBit`

    3. Unzip 
    ```
    unzip rabit_data.zip
    ```
    4. Python dependencies
    ```
        pip install joblib torch openmesh
    ```
        or
   ```
    pip install -r requirements.txt
   ```

   </details>


    
6. Download [blender](https://www.blender.org/download/)


*Note- Raise an issue if you are having trouble installing any of the above packages*

## B. SMPL from RGB video  

### Pose estimation 
To setup [VIBE](https://github.com/mkocabas/VIBE), run the following code chunks:
<hr>

*NOTE: This is the fine-tined version of VIBE, maintained by [Kulendu](https://github.com/kulendu).*

<hr>

1. Clone the repo:
```shell
git clone https://github.com/kulendu/VIBE.git
```

2. Install the requirements using `virtualenv` or  `conda`:
```shell
# pip (virtualenv)
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```

3. To run **VIBE** on any arbitary video, download the required data(i.e. the trained model and SMPL model parameters). To do this you can just run:
```shell
source scripts/prepare_data.sh
```

4. Then, for running the demo:
```shell
# Run on a local video
python demo.py --vid_file sample_video.mp4 --output_folder output/ --display

# Run on a YouTube video
python demo.py --vid_file https://www.youtube.com/watch?v=wPZP8Bwxplo --output_folder output/ --display
```

Refer to [VIBE/doc/demo.md](https://github.com/mkocabas/VIBE/blob/master/doc/demo.md) for more details about the demo code.

Sample demo output with the `--sideview` flag:

5. For running the demo on CPU:
```python
''' 
for demo.py:
loading the checkpoints and locating on the 'CPU' device
'''
ckpt = torch.load(pretrained_file, map_location=torch.device('cpu'))

''' 
for lib/models/vibe.py: 
loading the pretrained dictionary and checkpoints and then locating on the 'CPU' device
'''
#ln :96
pretrained_dict = torch.load(pretrained, map_location=torch.device('cpu'))['model']
# ln: 147
checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
# ln: 154
pretrained_dict = torch.load(pretrained, map_location=torch.device('cpu'))['model']
```
For more installation and running inferences, refer to the official [VIBE](https://github.com/mkocabas/VIBE) documentation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fZCIhiL4CwDNiMUCvePzBFbMVsA3LOZd?usp=sharing)

### Segmentation / POSA 
To setup [DECO](https://github.com/sha2nkt/deco), refer the following steps:

1. First clone the repo, then create a [conda](https://docs.conda.io/) env and install the necessary dependencies:
```shell
git clone https://github.com/sha2nkt/deco.git
cd deco
conda create -n deco python=3.9 -y
conda activate deco
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
It creates an conda env of python 3.9, with compatible dependencies

2. Install **PyTorch3D** from source:
```shell
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install .
cd ..
```

3. Install the other dependancies and download the required data:
```shell
pip install -r requirements.txt
sh fetch_data.sh
```

4. Please download [SMPL](https://smpl.is.tue.mpg.de/) (version 1.1.0) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) (v1.1) files into the data folder. Please rename the SMPL files to ```SMPL_FEMALE.pkl```, ```SMPL_MALE.pkl``` and ```SMPL_NEUTRAL.pkl```. The directory structure for the ```data``` folder has been elaborated below:

```
├── preprocess
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   ├── SMPL_NEUTRAL.pkl
│   ├── smpl_neutral_geodesic_dist.npy
│   ├── smpl_neutral_tpose.ply
│   ├── smplpix_vertex_colors.npy
├── smplx
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_FEMALE.pkl
│   ├── SMPLX_MALE.npz
│   ├── SMPLX_MALE.pkl
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_NEUTRAL.pkl
│   ├── smplx_neutral_tpose.ply
├── weights
│   ├── pose_hrnet_w32_256x192.pth
├── J_regressor_extra.npy
├── base_dataset.py
├── mixed_dataset.py
├── smpl_partSegmentation_mapping.pkl
├── smpl_vert_segmentation.json
└── smplx_vert_segmentation.json
```

**NOTE**: Sometimes running inferences on CPU might cause some hardware defination issue, which can be resolved by defining the following on **ln:30** in `inference.py`:
```python
checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
```


**NOTE**: Sometimes `Mapping` might cause some comapatibilty issue with *python 3.9*, to resolve this, use the following import:
```python
from collections.abc import Mapping
```

### Running demo on images/per-frame:
```shell
python inference.py \
    --img_src example_images \
    --out_dir demo_out
```
This command runs DECO on the images stored in the `example_images/` directory specified in `--img_src`, saving a rendering and a colored mesh in `demo_out/` directory

For referring more in-depth Training and Testing directions, refer to the official [DECO implementation](https://github.com/kulendu/deco/blob/main/README.md).

### C. Retargetting


###  D. Rendering 
Renders a video of motion transfer from smpl file dataset to RaBit Model.  


- Installation 
    <details>
    <summary>  Details </summary>
    * Command Line
    
    * Open terminal using `Cntr-shift-T` or `Cmd-shift-T` then paste

    ```
        <blender-python-path> pip install meshio
    ```
            
    * Example in Linux
            
        ```
        /home/shubh/blender-4.0.1-linux-x64/4.0/python/bin/python3.10 pip install meshio
        ```

- Input details

    `<sample-filepath>` is the path to the `.pkl` file containing the smpl:
        
        pose params - TxJx3, Rotation of SMPL joints (24 Joints version)
        body params - vec(10), Body parametes of SMPL Mesh (10 dimension version )
        camera ext - Tx6 or None, 6D camera pose (10 dimension version )
        camera int - K or None, Camera Intrinsic params



- Command Line

    ```
     blender --background --python rabit_render.py # For complete dataset
    ```
    Or 
    ```
    python3 renderer.py <smpl-filepath> # Specific file
    ```


