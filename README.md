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
    TODO (Nishant) 

## E. Overlaying 3D character over video 
    ASK Grace

