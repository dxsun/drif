import os

from parameters.parameter_server import get_current_parameters


# Simulator
import json
import numpy as np
# Configs
# --------------------------------------------------------------------------------------------

def get_sim_executable_path():
    return get_current_parameters()["Environment"]["simulator_path"]


def get_sim_config_dir():
    directory = get_current_parameters()["Environment"]["sim_config_dir"]
    print("get_sim_config_dir:", directory)
    return directory

# Configs
# --------------------------------------------------------------------------------------------
def get_env_config_path(env_id):
    directory = os.path.join(get_config_dir(), "configs", "random_config_%d.json" % env_id)
    print("get_env_config_path:", directory)
    return directory
#    return os.path.join(get_env_config_dir(), "config_%d.json" % env_id)

def get_template_path(env_id):
    return os.path.join(get_config_dir(), "templates", "random_template_%d.json" % env_id)


def get_instructions_path(env_id):
    return os.path.join(get_config_dir(), "instructions", "instructions_%d.txt" % env_id)


def get_curve_path(env_id):
    return os.path.join(get_config_dir(), "paths", "random_curve_%d.json" % env_id)


def get_anno_curve_path(env_id):
    return os.path.join(get_config_dir(), "anno_paths", "anno_curve_%d.json" % env_id)


def get_fpv_img_dir(real=True):
    subdir = "drone_img" if real else "sim_img"
    return os.path.join(get_config_dir(), subdir)


def get_fpv_img_flight_dir(env_id, real=True):
    return os.path.join(get_fpv_img_dir(real), "flight_%d" % env_id)


def get_all_poses_dir():
    return os.path.join(get_config_dir(), "poses")


def get_poses_dir(env_id):
    return os.path.join(get_all_poses_dir(), "flight_" + str(env_id))


def get_all_real_images_dir():
    return os.path.join(get_config_dir(), "drone_img")


def get_all_sim_images_dir():
    return os.path.join(get_config_dir(), "sim_img")


def get_real_images_dir(env_id):
    return os.path.join(get_all_real_images_dir(), "flight_" + str(env_id))


def get_sim_images_dir(env_id):
    return os.path.join(get_all_sim_images_dir(), "flight_" + str(env_id))


def get_env_config_dir():
    return os.path.join(get_config_dir(), "configs")


def get_pose_path(env_id, pose_id):
    return os.path.join(get_poses_dir(env_id), "pose_%d.json" % pose_id)


def get_real_img_path(env_id, pose_id):
    return os.path.join(get_real_images_dir(env_id), "usb_cam_%d.jpg" % pose_id)


def get_sim_img_path(env_id, pose_id):
    return os.path.join(get_sim_images_dir(env_id), "usb_cam_%d.png" % pose_id)


def get_plots_dir():
    return os.path.join(get_config_dir(), "plots")


def get_samples_dir():
    return os.path.join(get_config_dir(), "samples")


def get_rollout_plots_dir():
    return os.path.join(get_config_dir(), "policy_roll", "plots")


def get_rollout_samples_dir():
    return os.path.join(get_config_dir(), "policy_roll", "samples")


# Instruction Data
# --------------------------------------------------------------------------------------------
def get_instruction_annotations_path():
    return os.path.join(get_config_dir(), "annotation_results.json")


# Data and Models
# --------------------------------------------------------------------------------------------

def get_config_base_dir():
    base_dir = get_current_parameters()["Environment"]["config_dir"]
    return base_dir


def get_pretrained_weight_dir():
    return os.path.join(get_model_dir(), "pretrained_modules")


def get_model_dir():
    return os.path.join(get_config_base_dir(), "models")


def get_dataset_dir():
    return os.path.join(get_config_base_dir(), "data")


def get_config_dir():
    return os.path.join(get_config_base_dir(), "configs")


def get_instruction_cache_dir():
    return os.path.join(get_config_base_dir(), "configs", "tmp")


def get_supervised_data_filename(env):
    filename = "supervised_train_data_env_" + str(env)
    return filename


def get_landmark_weights_path():
    filename = os.path.join(get_config_dir(), "landmark_counts.txt")
    return filename


def get_self_attention_path():
    filename = get_config_dir()+ "/self_attention/"
    return filename


def get_noisy_pose_path():
    path = os.path.join(get_dataset_dir(), "noisy_poses")
    return path


######## Load 1 config file or list of config files ##############
def load_config_file(env_id):
    filename = get_env_config_path(env_id)
    with open(filename, 'r') as fp:
        config_dict = json.load(fp)
    return config_dict

def load_config_files(env_ids):
    list_of_dict = []
    for env_id in env_ids:
        filename = get_env_config_path(env_id)
        with open(filename, 'r') as fp:
            config_dict = json.load(fp)
        list_of_dict.append(config_dict)
    return list_of_dict

# Results
# --------------------------------------------------------------------------------------------

def get_results_path(run_name, makedir=False):
    dir = os.path.join(get_config_base_dir(), "results")
    if makedir:
        os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, run_name + "_results.json")


def get_results_dir(run_name=None, makedir=False):
    if run_name is not None:
        dir = os.path.join(get_config_base_dir(), "results", run_name)
    else:
        dir = os.path.join(get_config_base_dir(), "results")
    if makedir:
        os.makedirs(dir, exist_ok=True)
    return dir


# Others
# --------------------------------------------------------------------------------------------

def get_landmark_images_dir(landmark_name, eval=False):
    which = "eval" if eval else "train"
    path = os.path.join(get_config_base_dir(), "landmark_images",  which, landmark_name)
    #print(path)
    return path


def get_env_image_path(env_id):
    config_path = get_config_dir()
    img_path = os.path.join(config_path, "env_img", str(env_id) + ".png")
    return img_path


def get_current_config_folder(i=None, instance_id=None):
    current_conf_folder = "current_config"
    if instance_id is not None:
        current_conf_folder += "/" + str(instance_id)
    folder = current_conf_folder if i is None else "configs"
    return folder


def get_english_vocab_path():
    path = os.path.join(get_config_base_dir(), "english_vocabulary.json")
    return path


def get_thesaurus_path():
    path = os.path.join(get_config_dir(), "thesaurus.json")
    return path


def get_similar_instruction_path():
    path = os.path.join(get_config_dir(), "similar_instructions.json")
    return path


def get_close_landmarks_path():
    path = os.path.join(get_config_dir(), "close_landmarks.json")
    return path


def get_semantic_maps_path():
    path = os.path.join(get_dataset_dir(), "prebuilt_maps")
    return path


def get_human_eval_envs_path():
    path = os.path.join(get_config_dir(), "human_eval_envs.json")
    return path


def get_human_eval_root_path():
    path = os.path.join(get_config_base_dir(), "human_eval")
    return path


def get_env_split_path():
    path = os.path.join(get_config_dir(), "train_env_split.json")
    return path


def get_semantic_map_path(map_id, env_id, set_idx, seg_idx):
    maps_path = get_semantic_maps_path()
    env_maps_path = os.path.join(maps_path, "maps_" + str(map_id) + "_" + str(env_id) + "_" + str(set_idx) + "_" + str(seg_idx) + ".npy")
    return env_maps_path



########### Landmark locations ################

def get_landmark_locations(conf_json):
    landmark_loc = []
    for i, x in enumerate(conf_json['xPos']):
        landmark_loc.append(np.array([x, conf_json['zPos'][i],0]))
    return landmark_loc
