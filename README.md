# SMPL2RABIT-Blender 
    - Retarget motion from SMPL to RaBit module  

## A. Installation

1. Download the codebase
    ```
        git --recursive 
    ```

2. Download [blender](https://www.blender.org/download/)

3. Python packages 
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
4. Rabit installation
    <details>
    <summary>  Details </summary>
    1. Clone RaBit Library

    ```
        git clone https://github.com/zhongjinluo/RaBit.git
        cd RaBit 
    ```
    2. Download model data from [link](https://drive.google.com/file/d/1yvweTYPKtmuMt5Eu7CHZ4-Do4CRYLFtp/view?usp=sharing) to `<HOME_PATH>/RaBit`

    3. Unzip 
    ```
    unzip rabit_data.zip
    ```
    </details>



*Note- Raise an issue if you are having trouble installing any of the above packages*





###  B. Rendering 
Renders a video of motion transfer from smpl file dataset to RaBit Model.  

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



## C. Retargetting  
To retarget .trc file to SMPL format  
```
python3 retarget.py # For complete dataset
```
Or 
```
python3 retarget.py <sample-filepath> # Specific file
```



## D. From RGB video  

### Pose estimation 
    TODO (Nishant) 

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


## E. Overlaying 3D character over video 
    ASK Grace

