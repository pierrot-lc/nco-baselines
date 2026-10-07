##########################################################################################
# Machine Environment Config

from pathlib import Path
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

import numpy as np
from TSP.Test_All.TSPTester import TSPTester as Tester

##########################################################################################
# parameters


env_params = {
    # 'problem_size': 100,
    "pomo_size": 1,
    "k_nearest": 1,
    "beam_width": 16,
    "decode_method": "greedy",
    "mode": "test",
    "test_in_tsplib": False,
    "tsplib_path": None,
    "data_path": None,
    "load_way": "allin",
    "sub_path": False,
    "budget": 10,
    "PRC": True,
    "repair_max_sub_length": 1000,
    "random_insertion": False,
}

model_params = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
    "use_k_nearest": True,
    "k_nearest_num": 1000,
}

tester_params = {
    "use_cuda": USE_CUDA,
    "cuda_device_num": CUDA_DEVICE_NUM,
    "model_load": {
        "path": None,  # directory path of pre-trained model and log files saved.
    },
    "test_episodes": 16,  # 65
    "test_batch_size": 4,
}

logger_params = {"log_file": {"desc": "test__uniform1k_greedy", "filename": "log.txt"}}


##########################################################################################
# main


def main_test(
    path,
    PRC,
    repair_max_sub_length,
    budget,
    random_insertion,
    data_path,
    batch_size,
):
    create_logger(**logger_params)
    tester_params["model_load"] = {
        "path": path,
    }

    data = np.load(data_path)
    tester_params["test_episodes"] = len(data["nodes"])
    tester_params["test_batch_size"] = batch_size
    env_params["data_path"] = data_path
    env_params["PRC"] = PRC
    env_params["repair_max_sub_length"] = repair_max_sub_length
    env_params["budget"] = budget
    env_params["random_insertion"] = random_insertion

    # if prob_size == -1:
    #     env_params["test_in_tsplib"] = True
    #     env_params["tsplib_path"] = data_path
    #     env_params["data_path"] = data_path

    if not random_insertion and budget == 0:  # (purely greedy search)
        model_params["use_k_nearest"] = False

    _print_config()
    tester = Tester(
        env_params=env_params, model_params=model_params, tester_params=tester_params
    )

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


def _print_config():
    logger = logging.getLogger("root")
    logger.info("DEBUG_MODE: {}".format(DEBUG_MODE))
    logger.info("USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(USE_CUDA, CUDA_DEVICE_NUM))
    [
        logger.info(g_key + "{}".format(globals()[g_key]))
        for g_key in globals().keys()
        if g_key.endswith("params")
    ]


##########################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--data", "-d", type=Path, required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args()

    PRC = True
    repair_max_sub_length = 1000

    # purely greedy search: random_insertion=False, budget=0
    # Initilize the solution by random insertion and refine it by PRC: random_insertion=True, budget>0
    budget = 0
    random_insertion = False

    main_test(
        args.checkpoint,
        PRC,
        repair_max_sub_length,
        args.budget,
        args.budget != 0,
        args.data,
        args.batch_size,
    )
