Expected requirements

- TurboVNC (for viewing simulator)
- python 3

How to start the simulator:

  - must install turbovnc on own machine

  On csail machine:

    (in csail machine)
    $ cd /opt/TurboVNC/bin 
    $ ./vncserver :[number]

    ( Starts the vnc server on port [number])

  How to view the simulator on local machine:
    (in local machine)
    $ cd /opt/TurboVNC/bin/
    $ ./vncviewer
    ( enter the New TurboVNC Connection window, make sure the Vnc server is [csail_machine].csail.mit.edu:[port])
    $ type in the turbovnc password 
    ( a big window with circle and 3 small circles should pop up)

  On another terminal/ssh of csail machine:
    ## To run interactive top down experiment
    $ DISPLAY=:[TurboVNC port] vglrun -d :0.0 python interactive/interactive_top_down_pred.py interactive_top_down


CoRL 2018 training + experiments (from github)

  $ cd drif/mains
  $ python data_collect/collect_supervised_data.py corl_datacollect

  ## This saves data into "unreal_config_nl/data" (or whatever "config_dir" is set to in the corl_json settings - reference System Configuration in /drif readme) in the file format "supervised_train_data_env_#" for some number #

  $ DISPLAY=:[TurboVNC port] vglrun -d :0.0 python train/train_supervised.py corl_pvn_train_stage1

  ## This begins training the first stage of the model
  

How to use tensorboard on csail machine:
	
	- Note: Can't use native pytorch tensorboard, because this is only supported for pytorch 1.1+ (around there) but drif/ uses pytorch 0.4.1, so updating to 1.1 breaks everything

	Must ssh but forward localhost + port on csail machine to my own machine, which is what -L does (the following command forwards localhost:6006 on the csail machine to 4000 on local machine)

	(local) $ ssh -L 4000:127.0.0.1:6006 [username]@[csail machine name].csail.mit.edu
	(csail) $ cd drif
	(csail) $ tensorboard --logdir='./logs' --port=6006

  (this will put everything into the /logs directory)

  In order to send the logs to a new directory, go to drif/parameters/run_params/environments/corl_18.json and change the "tensorboard_dir" value to the new logs directory (and make sure the tensorboard command line flag for --logdir also corresponds to the same directory)

  (NOTE: 3 different files write to the same directory, namely /drif/learning/training/train_supervised.py logs the loss, and /drif/learning/models/model_sm_trajectory_ratio.py logs the images of the intermediate maps, then /learning/modules/goal_pred_criterion.py logs the images of the predicted trajectory and goal trajectory)

	Then, navigate to localhost:4000 inside your own browser on your own local machine


Training from a preloaded model
	- Look inside drif/mains/train/train_supervised.py
	- Add a load_pytorch_model with the name of the model inside the config directory/models (in this case, it's /storage/dxsun/unreal_config_nl/models/)
	- for example: load_pytorch_model(model, "supervised_pvn_stage1_train_corl_pvn_stage1")



Notes:

Intermediate models:
	- stored inside unreal_config_nl/models/tmp (the config directory)

Parameters:
	- All the parameters (for the training at least) come from /drif/parameters/run_params 
	- corl_pvn_train_stage1.json is where first stage of training comes from
	- If it doesn't exist, create attribute named "start_epoch" inside "Supervised" part of json to contain a start epoch if want to start training from a different epoch (if it doesn't exist, that means it starts at 0)

Top-down images (for searching: - top down, topdown)
  (go to config directory)
  - unreal_config_nl/configs/env_img
  - The number of the image file corresponds with /unreal_config_nl/configs/configs/random_config_[image_number]

Random notes:
	- on first loadup of the simulator (random_json_0), doesn't go into pomdp interface, while it does for every next environment
