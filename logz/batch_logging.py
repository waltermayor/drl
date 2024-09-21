import time

import jax.numpy as jnp
import numpy as np
import wandb
from utils.ppo_metrics import (
    compute_ranks_from_features,
    compute_feature_norms
)

batch_logs = {}
log_times = []


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
        "mean_norm_input_features": info["mean_norm_input_features"],
        "mean_norm_actor_preoutput_features": info["mean_norm_actor_preoutput_features"],
        "mean_norm_critic_preoutput_features": info["mean_norm_critic_preoutput_features"],
        "mean_norm_Dense_0_kernel": info["mean_norm_Dense_0_kernel"],
        "mean_norm_Dense_0_bias": info["mean_norm_Dense_0_bias"],
        "mean_norm_Dense_6_kernel": info["mean_norm_Dense_6_kernel"],
        "mean_norm_Dense_6_bias": info["mean_norm_Dense_6_bias"],
    }

    if(config['ANALYZE_STRUCTURE']):
        to_log["feature_actor"] = info["feature_actor"]
        to_log["feature_pre_actor"] = info["feature_pre_actor"]
        to_log["feature_critic"] = info["feature_critic"]
        to_log["feature_pre_critic"] = info["feature_pre_critic"]    

    sum_achievements = 0
    for k, v in info.items():
        if "achievements" in k.lower():
            to_log[k] = v
            sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements

    if config.get("TRAIN_ICM") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
            to_log["icm_forward_loss"] = info["icm_forward_loss"]
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info["rnd_loss"]

    return to_log


def batch_log(update_step, log, config):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)
    print("len batch_logs: ", len(batch_logs[update_step]))
    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0][key]]
            else:
                for i in range(config["NUM_REPEATS"]):
                    val = batch_logs[update_step][i][key]
                    
                    #print("val: ",type(val))
                    if val.ndim==0:
                        if not jnp.isnan(val):
                            #print("val: ", val)
                            agg.append(val)
                    else:
                        if not jnp.isnan(val).any():
                            agg.append(val)

                    
            features_model = []
            if len(agg) > 0:
                if key in [
                    "episode_length",
                    "episode_return",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "rnd_loss",
                    "mean_norm_input_features",
                    "mean_norm_actor_preoutput_features",
                    "mean_norm_critic_preoutput_features",
                    "mean_norm_Dense_0_kernel",
                    "mean_norm_Dense_0_bias",
                    "mean_norm_Dense_6_kernel",
                    "mean_norm_Dense_6_bias",
                ]:
                    agg_logs[key] = np.mean(agg)
                else:
                     agg_logs[key] = np.array(agg)

                if key in [
                    "feature_actor",
                    "feature_pre_actor",
                    "feature_critic",
                    "feature_pre_critic"
                ]:
                    print("all: ",jnp.array(agg).shape)
                    mean_agg = np.mean(agg,axis=0)
                    print("after mean: ", mean_agg.shape)
                    features_model.append(mean_agg)
                    # ranks = compute_ranks_from_features(mean_agg)
                    # print(ranks)


        log_times.append(time.time())

        if config["DEBUG"]:
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        wandb.log(agg_logs)
