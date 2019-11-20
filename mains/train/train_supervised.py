import sys
sys.path.insert(1, "/storage/dxsun/drif/")
from learning.training.train_supervised import Trainer
import data_io
from data_io.train_data import file_exists
from data_io.models import load_model
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.weights import restore_pretrained_weights, save_pretrained_weights
from data_io.instructions import get_all_env_id_lists
from data_io.env import load_env_split
from parameters.parameter_server import initialize_experiment, get_current_parameters


# Supervised learning parameters
def train_supervised():
    initialize_experiment()

    setup = get_current_parameters()["Setup"]
    supervised_params = get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]

    model, model_loaded = load_model()
    # import pdb; pdb.set_trace()
    # import pickle
    # with open('/storage/dxsun/model_input.pickle', 'rb') as f: data = pickle.load(f)
    # g = model(data['images'], data['states'], data['instructions'], data['instr_lengths'], data['has_obs'], data['plan'], data['save_maps_only'], data['pos_enc'], data['noisy_poses'], data['start_poses'], data['firstseg'])
    print("model:", model)
    print("model type:", type(model))
    print("Loading data")
    train_envs, dev_envs, test_envs = get_all_env_id_lists(max_envs=setup["max_envs"])
    if "split_train_data" in supervised_params and supervised_params["split_train_data"]:
        split_name = supervised_params["train_data_split"]
        split = load_env_split()[split_name]
        train_envs = [env_id for env_id in train_envs if env_id in split]
        print("Using " + str(len(train_envs)) + " envs from dataset split: " + split_name)

    filename = "supervised_" + setup["model"] + "_" + setup["run_name"]
    start_filename = "tmp/" + filename + "_epoch_" + str(supervised_params["start_epoch"])
    print("start_filename:", start_filename)
    if supervised_params["start_epoch"] > 0:
        if file_exists(start_filename):
            print("THE FILE EXISTS code1")
            load_pytorch_model(model, start_filename)
        else:
            print("Couldn't continue training. Model file doesn't exist at:")
            print(start_filename)
            exit(-1)

    all_train_data, all_test_data = data_io.train_data.load_supervised_data(max_envs=2000)
    if setup["restore_weights_name"]:
        restore_pretrained_weights(model, setup["restore_weights_name"], setup["fix_restored_weights"])

    trainer = Trainer(model, epoch=supervised_params["start_epoch"], name=setup["model"], run_name=setup["run_name"])
    # import pdb;pdb.set_trace()
    print("Beginning training...")
    best_test_loss = 1000
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_data=all_train_data, train_envs=train_envs, eval=False)

        trainer.model.correct_goals = 0
        trainer.model.total_goals = 0

        test_loss = trainer.train_epoch(train_data=None, train_envs=dev_envs, eval=True)

        print("GOALS: ", trainer.model.correct_goals, trainer.model.total_goals)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_pytorch_model(trainer.model, filename)
            print("Saved model in:", filename)
        print ("Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
        save_pytorch_model(trainer.model, "tmp/" + filename + "_epoch_" + str(epoch))
        if hasattr(trainer.model, "save"):
            trainer.model.save(epoch)
        save_pretrained_weights(trainer.model, setup["run_name"])


if __name__ == "__main__":
    train_supervised()
