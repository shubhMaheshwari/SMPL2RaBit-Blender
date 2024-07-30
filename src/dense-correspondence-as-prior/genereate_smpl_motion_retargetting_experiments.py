## Code to generate SMPL motion retargetting experiments. 
## Usage: python genereate_smpl_motion_retargetting_experiments.py 

import os
import sys
import argparse
import numpy as np


import polyscope as ps
import polyscope.imgui as psim

# Insert the local directory at the beginning of sys.path
sys.path.insert(0, 'smplx/smplx')

# Now import the package
import smplx
import torch


# Load from modules 
from constants import get_logger
# from renderer import Visualizer


class Mesh: 
    def __init__(self,parametric_model,**kwargs):

        self.parametric_model = parametric_model
        for k in kwargs: 
            self.__dict__[k] = kwargs[k]

        if parametric_model == 'SMPL':
            model_path = 'parametric_models/SMPL_NEUTRAL.pkl'
            self.model = smplx.create(model_path, model_type='smpl', gender='neutral', ext='pkl',
                                num_betas=10, num_pca_comps=12, batch_size=self.cmd_args.batch_size, 
                                create_global_orient=True, create_transl=True,                                
                                create_body_pose=True, create_betas=True,

                                create_left_hand_pose=False, create_right_hand_pose=False,
                                create_expression=False, create_jaw_pose=False,
                                create_leye_pose=False, create_reye_pose=False,
                                ).to(self.device)
            
        
        elif parametric_model == 'RABIT': 
            raise NotImplementedError("Write code to extract vertices and faces for RABIT")   
    

        self.initialize_mesh_sequence()

    def initialize_mesh_sequence(self):
        
        if self.parametric_model == 'SMPL':
            with torch.no_grad():

                model_output = self.model(return_verts=True)

                # if not hasattr(self,'pose') or self.pose is None:
                #     model_output = self.model(return_verts=True)
                # elif os.path.exists(self.pose): 
                    
                #     smpl_params = self.load_pose(self.pose)       
                #     shape_params = smpl_params['shape_params'].repeat(smpl_params["pose_params"].shape[0],1)
                #     print(smpl_params['pose_params'].shape, shape_params.shape)
                #     model_output = self.model(global_orient=smpl_params['pose_params'][:,:3].reshape(-1,1,3), body_pose=smpl_params['pose_params'][:,3:].reshape(-1,23,3), th_betas=shape_params, return_verts=True)

                # Dynamic variables
                self.vertices = model_output.vertices.detach().cpu().numpy()
                self.joints = model_output.joints.detach().cpu().numpy()
                self.joints = self.joints[:,:24] # Need only the first for now
 

            # Static variables
            self.faces = self.model.faces.copy()
            self.lbs_weights = self.model.lbs_weights.detach().cpu().numpy()
            self.vertex2joints = self.model.J_regressor.detach().cpu().numpy()
            self.kinematic_tree = self.model.parents.detach().cpu().numpy()
            self.kinematic_tree[0] = 0 # Root node is parent to itself

        else: 
            raise NotImplementedError("Only works for SMPL. Not for parametric model:{self.parametric_model}")  

        # Auxillary variables
        self.nT, self.nV, self.D = self.vertices.shape        
        assert self.D == 3, 'Only 3D vertices are supported got: {}'.format(self.D)

        # Initial 1-1 mapping between SMPL and vertices
        self.mapping = np.arange(self.nV) 

        return self.get_mesh_sequence()

    def get_mesh_sequence(self):

        return {'vertices': self.vertices, 'faces': self.faces,
                'mapping': self.mapping,
                'lbs_weights': self.lbs_weights,
                'joints': self.joints,
                'kinematic_tree' : self.kinematic_tree,
                'vertex2joints': self.vertex2joints
                }

    def save(self,save_path): 
        
        if os.path.isdir(save_path): 
            save_path = os.path.join(save_path,f'{self.parametric_model}_gt.npy')

        self_dict = self.get_mesh_sequence()

        np.save(save_path,self_dict)    

    def load_pose(self,pose_path): 
        import pickle
        try: 
            with open(pose_path, 'rb') as f:
                smpl_params = pickle.load(f)
        except Exception as e: 
            print(f"Unable to open smpl file:{pose_path} Try deleting the file and rerun retargetting. Error:{e}")
            raise

        for k in smpl_params: 
            smpl_params[k] = torch.from_numpy(smpl_params[k]).to(self.device)	

        return smpl_params


    @staticmethod
    def invert_mapping(new_indices): 
        # Step 2: Create an empty array of the same length for the inverse mapping
        inverse_mapping = np.empty_like(new_indices)

        # Step 3: Fill the inverse mapping array
        for i, p in enumerate(new_indices):
            inverse_mapping[p] = i

        return inverse_mapping

    def distorte(self,method):
        if method == 'remesh':
            new_indices = np.random.permutation(self.vertices.shape[1])
            
            
            self.mapping = new_indices
            self.vertices = self.vertices[:,new_indices]
            self.vertex2joints = self.vertex2joints[:,new_indices]
            self.lbs_weights = self.lbs_weights[new_indices]

            inverse_mapping = self.invert_mapping(new_indices)

            # Inverse map the faces
            self.faces = inverse_mapping[self.faces]

            # Rearrange faces 
            new_face_indices = np.random.permutation(self.faces.shape[0])
            self.faces = self.faces[new_face_indices]


        elif method == 'decimate':
            simplified_mesh = mesh.simplify_quadratic_decimation(target_faces=desired_number_of_faces)

        elif method == 'subdivide': 
            # Subdivide the mesh to increase the number of faces
            subdivided_mesh = mesh.subdivide()

        elif method == 'gaussian noise':
            args.gaussian_noise_std = 0.01 * self.get_bounding_box() # Distrort vertices by 1% of the bounding box. 
            noise = np.random.normal(0, args.gaussian_noise_std, self.nV)
        else: 
            raise NotImplementedError("Distortion Method:{} not implemented:".format(method))




def save_experiment(args,source,target):


    source_dict = source.get_mesh_sequence()
    target_dict = target.get_mesh_sequence()

    # Create input for the experiment
    save_dict = {  
        'source_vertices':source_dict['vertices'], 
        'source_faces': source_dict['faces'], 
        'target_vertices': target_dict['vertices'], 
        'target_faces': target_dict['faces'],
        'mapping': target_dict['mapping']
        }

    np.save(os.path.join(args.out_dir,f'{args.source}_to_{args.source}_gt.npy'),save_dict)    

    # store the compelete information 
    source.save(os.path.join(args.out_dir,f'source_gt.npy'))
    target.save(os.path.join(args.out_dir,f'target_gt.npy'))


class Visualizer:

    def __init__(self):
        
        # Create scene variables
        self.ps_scene = {
            'dist':1.0,
            'minT': 0, 
            'maxT': 1, 
            "is_true1": False,
            "is_true2": True,
            "ui_int": 7,
            "ui_float1": -3.2,
            "ui_float2": 0.8,
            "ui_color3": (1., 0.5, 0.5),
            "ui_color4": (0.3, 0.5, 0.5, 0.8),
            "ui_angle_rad": 0.2,
            "exp_name": "Enter instructions here",
            "experiment_options": [],
            "experiment_options_selected": None,

            "category_options": [],
            "category_options_selected": None,

            "rank": 1,

            "is_paused": False
        }

        ps.init()
        ps.set_view_projection_mode("orthographic")

        # Time slideer
        self.t = 0



    @staticmethod
    def generate_color_map(N, colormap_name='viridis'):
        import matplotlib.pyplot as plt
        # Get the colormap
        colormap = plt.get_cmap(colormap_name)
        
        # Normalize indices to the range [0, 1]
        normalized_indices = np.linspace(0, 1, N)
        
        # Generate colors using the colormap
        colors = colormap(normalized_indices)[:, :3]  # Ignore the alpha channel
        
        return colors



    def callback(self):


        self.t = self.ps_scene['minT'] if self.t < self.ps_scene['minT'] else self.t
        self.t %= self.ps_scene['maxT']



        if hasattr(self,'ps_source'): 
            self.ps_source.update_vertex_positions(self.ps_scene['source_vertices'][self.t] - np.array([1,0,0])*(self.ps_scene['dist'])*self.ps_scene['bbox']) 

        if hasattr(self,'ps_target'):
            self.ps_target.update_vertex_positions(self.ps_scene['target_vertices'][self.t] + np.array([1,0,0])*(self.ps_scene['dist'])*self.ps_scene['bbox']) 
        

        if not self.ps_scene['is_paused']: 
            self.t += 1 


        # Check keyboards for inputs
        
        # Check for spacebar press to toggle pause
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)):
            
            self.ps_scene['is_paused'] = not self.ps_scene['is_paused']

        # Left arrow pressed
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)):
            self.t -= 1

        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)):
            self.t += 1

        # Input text
        changed, self.ps_scene["exp_name"] = psim.InputText("- Experiment Name", self.ps_scene["exp_name"])


        ############## Create the GUI to update the animations 
        # psim.Begin("Video Controller",True)


        # psim.SetWindowPos((1340,100.0),1) # Set the position the window at the bottom of the GUI
        # psim.SetWindowSize((500.0,700.0),1)

        # Create a floater to show the timestep and adject self.t accordingly
        changed, self.t = psim.SliderInt("", self.t, v_min=self.ps_scene['minT'], v_max=self.ps_scene['maxT'])
        psim.SameLine()

        # Create a render button which when pressed will create a .mp4 file
        if psim.Button("<"):
            self.t -= 1
        
        psim.SameLine()
        if psim.Button("Play Video" if self.ps_scene['is_paused'] else "Pause Video"):
            self.ps_scene['is_paused'] = not self.ps_scene['is_paused']

        psim.SameLine()
        if psim.Button(">"):
            self.t += 1

        # psim.SameLine()
        if psim.Button("Render Video"):
            self.render()        

        if(psim.TreeNode("Load Experiment")):

            # psim.TextUnformatted("Load Optimized samples")

            changed = psim.BeginCombo("- Experiement", self.ps_scene["experiment_options_selected"])
            if changed:
                for val in self.ps_scene["experiment_options"]:
                    _, selected = psim.Selectable(val, selected=self.ps_scene["experiment_options_selected"]==val)
                    if selected:
                        self.ps_scene["experiment_options_selected"] = val
                psim.EndCombo()

            changed = psim.BeginCombo("- Category", self.ps_scene["category_options_selected"])
            if changed:
                for val in self.ps_scene["category_options"]:
                    _, selected = psim.Selectable(val, selected=self.ps_scene["category_options_selected"]==val)
                    if selected:
                        self.ps_scene["category_options_selected"] = val
                psim.EndCombo()



            changed, new_rank = psim.InputInt("- rank", self.ps_scene["rank"], step=1, step_fast=10) 
            if changed: 
                self.ps_scene["rank"] = new_rank # Only change values when button is pressed. Otherwise will be continously update like self.t 
                
                if self.ps_scene["rank"] > 100:
                    self.ps_scene['rank'] = 100
                elif self.ps_scene["rank"] < 1: 
                    self.ps_scene['rank'] = 1 
                else: 
                    pass

            
            if(psim.Button("Load Optimized samples")):
                self.update_skeleton()
            psim.TreePop()


        psim.End()

        # == Set parameters

        # These commands allow the user to adjust the value of variables.
        # It is important that we assign the return result to the variable to
        # update it. 
        # For most elements, the return is actually a tuple `(changed, newval)`, 
        # where `changed` indicates whether the setting was modified on this 
        # frame, and `newval` gives the new value of the variable (or the same 
        # old value if unchanged).
        #
        # For numeric inputs, ctrl-click on the box to type in a value.


        # Checkbox
        # changed, self.ps_scene["is_true1"] = psim.Checkbox("flag1", self.ps_scene["is_true1"]) 
        # if(changed): # optionally, use this conditional to take action on the new value
            # psim.SetWindowSize((0.0,200.0),1)
            # psim.SetWindowPos("Some view",(0.0,0.0),1)

            # print("Checkbox changed to ", self.ps_scene["is_true1"])
            # pass 
        # psim.SameLine() 
        # changed, self.ps_scene["is_true2"] = psim.Checkbox("flag2", self.ps_scene["is_true2"]) 

        # Input ints



        # Input floats using two different styles of widget
        # changed, self.ps_scene["ui_float1"] = psim.InputFloat("ui_float1", self.ps_scene["ui_float1"]) 
        # psim.SameLine() 


        # Input colors
        # changed, self.ps_scene["ui_color3"] = psim.ColorEdit3("ui_color3", self.ps_scene["ui_color3"])
        # psim.SameLine() 
        # changed, self.ps_scene["ui_color4"] = psim.ColorEdit4("ui_color4", self.ps_scene["ui_color4"])


        # Combo box to choose from options
        # There, the options are a list of strings in `ui_options`,
        # and the currently selected element is stored in `ui_options_selected`.
        # psim.PushItemWidth(200)

        # psim.PopItemWidth()


        # Use tree headers to logically group options

        # This a stateful option to set the tree node below to be open initially.
        # The second argument is a flag, which works like a bitmask.
        # Many ImGui elements accept flags to modify their behavior.
        # psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)

        # # The body is executed only when the sub-menu is open. Note the push/pop pair!


        # psim.PopItemWidth()



    def render_experiment(self,args,dist=1.0): 
        data = np.load(os.path.join(args.out_dir,f'{args.source}_to_{args.source}_gt.npy'),allow_pickle=True).item()

        # Get the bounding box as the max of all bounding boxes in the scene
        bbox = max([np.max( vertices, axis=tuple(range(len(vertices.shape) - 1))) for data_key,vertices in data.items() if 'vertices' in data_key], key =  lambda x: np.linalg.norm(x)) 
        self.ps_scene['bbox'] = bbox

        # Set the distance between objects
        self.ps_scene['dist'] = dist # Distance between objects
        self.ps_scene["exp_name"] = args.exp_name





        self.ps_source = ps.register_surface_mesh("Source", data['source_vertices'][0] - np.array([1,0,0])*(dist)*bbox , data['source_faces'])
        self.ps_target = ps.register_surface_mesh("Target", data['target_vertices'][0] + np.array([1,0,0])*(dist)*bbox, data['target_faces'])

        # Save for future operations
        self.ps_scene['source_vertices'] = data['source_vertices']
        self.ps_scene['target_vertices'] = data['target_vertices']
        self.t = 0
        self.ps_scene['maxT'] = min([ x.shape[0] for x in  [ data['source_vertices'],  data['target_vertices']]])


        source_color_map = self.generate_color_map(data['source_vertices'].shape[1])
        self.ps_source.add_color_quantity("Indices", source_color_map)
        self.ps_source.add_color_quantity("Mapping", source_color_map,enabled=True)

        target_color_map = self.generate_color_map(data['target_vertices'].shape[1])
        self.ps_target.add_color_quantity("Indices", target_color_map,)
        self.ps_target.add_color_quantity("Mapping", target_color_map[data['mapping']],enabled=True)

        ps.set_user_callback(self.callback)

        ps.show()




def main(args):

    # Create experiement directory 
    args.out_dir  = os.path.join(args.out_dir,args.exp_name)
    os.makedirs(args.out_dir,exist_ok=True)

    # Create a logger to print and log resutlts
    logger = get_logger(out_dir=args.out_dir,debug=args.debug)
    logger.info("Creating Experiment with arguements: {}".format(args))

    if os.path.exists(os.path.join(args.out_dir,f'{args.exp_name}.pkl')) and not args.force: # If data already exists and we are not forcing to overwriting
        logger.info(f"Setup already exists at:{os.path.join(args.out_dir,f'{args.exp_name}.pkl')}. Skipping")    
        return 



    # visualizer = Visualizer() if args.render else None #Create a polyscope visualizer if we are rendering
    visualizer = None
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    

    # Create source mesh
    # Create arguments for source mesh 

    source = Mesh(args.source,pose=args.pose,visualizer=visualizer,logger=logger,cmd_args=args,device=device)

    # Create target mesh
    target = Mesh(args.target,pose=args.pose,visualizer=visualizer,logger=logger,cmd_args=args,device=device)
    target.distorte('remesh')
    
    source.mapping = target.invert_mapping(target.mapping)


    # Retarget Everything
    if args.out_dir != '':
        save_experiment(args,source,target)
    

    if args.render:
        visualizer = Visualizer()
        visualizer.render_experiment(args)    



if __name__ == '__main__':

    ############################ Command Line Arguments ############################################
    # Create the argument parser
    parser = argparse.ArgumentParser(
                    prog=' Retargetting Parameteric models',
                    description='This folder contains the experiments used to determine whether dense correspondence between 3D objects can be used to determine whether dense shape correspondence methods, such as [SpiderMatch](!https://paulroetzer.github.io/publications/2024-06-19-spidermatch.html), or [functional maps](!https://dl.acm.org/doi/10.1145/2185520.2185526) can be used to retarget motion.',
                    epilog='')


    # Source and target parametric models
    parser.add_argument('--source',type=str, default="SMPL", help="Source mesh")  
    parser.add_argument('--target',type=str, default="SMPL", help="Target mesh")  

    # Experiment params
    parser.add_argument('--exp-name',type=str, default="Exp-1", help="Experiment name")  
    parser.add_argument('--out-dir',type=str, default="./save", help="Location where all experiements are saved")  
    parser.add_argument('--batch-size',type=int, default=10, help="Location where all experiements are saved")  

    # Experiment Variables 
    parser.add_argument('--pose',type=str, default="", help="Motion to be retargetted")  
    parser.add_argument('--shape', type=str, default="", help="Whether to change the shape of the target model")

    parser.add_argument('--distortions', dest='distortions', nargs='?', const=None, default=None, help="Distortions allowed")
    parser.add_argument('--distorion', type=str, default="", help="Whether to change the shape of the target model")

    parser.add_argument('-f', '--force',action='store_true',help="forces a re-run on retargetting even if pkl file containg smpl data is already present.")  # on/off flag

    parser.add_argument('--render', action='store_true', default=True, help="Render a video and save it it in RENDER_DIR. Can also be set in the utils.py")  # on/off flag

    parser.add_argument('--debug', dest='debug', action='store_true',  help="Debug and run on less number of frames")
    parser.add_argument('--no-debug', dest='debug', action='store_false',default=False, help="Debug and run on less number of frames")

    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True,help="Whether to use CUDA or CPU")
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',help="Whether to use CUDA or CPU")

    cmd_line_args = parser.parse_args()
    print("Arguments:",cmd_line_args)
    main(cmd_line_args)    
    