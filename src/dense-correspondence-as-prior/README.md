# Dense correspondence priors 
    This folder contains the experiments used to determine whether dense correspondence between 3D objects can be used to determine whether dense shape correspondence methods, such as [SpiderMatch](!https://paulroetzer.github.io/publications/2024-06-19-spidermatch.html), or [functional maps](!https://dl.acm.org/doi/10.1145/2185520.2185526) can be used to retarget motion. 

## Setup: 
    In the current setup we use [SMPL](!https://smpl.is.tue.mpg.de/) as our source mesh. We perform 4 experiments with different of target mesh. These are with the following as the target mesh:
    
    SMPL 
        - with different shape
        - with different pose 
        - with different shape and pose 
    
    [RaBit](!https://github.com/zhongjinluo/RaBit) 
        - with different shape
        - with different pose 
        - with different shape and pose 

## Assumptions 
    We assume that the source mesh is SMPL and complete and the skeleton is provided for the source experiments. 


## Setup: 
    . We take SMPL mesh and perform  smple based of the shape


## Experiement Environment: 
1. save/<Exp-Name>
    1. Arguments
    2. Log 
    2. Source
    3. Target
    4. Animation
    4. Images 
    5. Videos 
    6. Pinnochio Retargetting
    7. SpiderMatch Retargetting 
    8. Evaluation  