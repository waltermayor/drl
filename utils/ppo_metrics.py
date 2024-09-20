import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import wandb
import numpy as np
import optax



def dict_with_prefix(d, prefix): 
    """Adds a prefix to all keys in a dict."""
    return {prefix + k: v for k, v in d.items()}

def compute_ranks_from_features(feature_matrices):
    """Computes different approximations of the rank of the feature matrices."""

    cutoff = 0.01
    threshold = 1 - cutoff

    if feature_matrices.ndim >= 3 and feature_matrices.shape[1] < feature_matrices.shape[2]:
        return {}  # N < D.

    #svals = jscipy.linalg.svdvals(feature_matrices)
    svals = jnp.linalg.svdvals(feature_matrices)

    # (1) Effective rank
    sval_sum = jnp.sum(svals, axis=1)
    sval_dist = svals / sval_sum[:, None]
    sval_dist_fixed = jnp.where(sval_dist == 0, jnp.ones_like(sval_dist), sval_dist)
    effective_ranks = jnp.exp(-jnp.sum(sval_dist_fixed * jnp.log(sval_dist_fixed), axis=1))

    # (2) Approximate rank
    sval_squares = svals**2
    sval_squares_sum = jnp.sum(sval_squares, axis=1)
    cumsum_squares = jnp.cumsum(sval_squares, axis=1)
    threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum[:, None])
    approximate_ranks = (~threshold_crossed).sum(axis=-1) + 1

    # (3) srank
    cumsum = jnp.cumsum(svals, axis=1)
    threshold_crossed = cumsum >= threshold * sval_sum[:, None]
    sranks = (~threshold_crossed).sum(axis=-1) + 1

    # (4) Feature rank
    n_obs = feature_matrices.shape[1]
    svals_of_normalized = svals / jnp.sqrt(n_obs)
    over_cutoff = svals_of_normalized > cutoff
    feature_ranks = jnp.sum(over_cutoff, axis=-1)

    # (5) JAX/NumPy rank
    jnp_ranks = jnp.linalg.matrix_rank(feature_matrices)

    # Some singular values
    singular_values = {
        'lambda_1': svals_of_normalized[:, 0],
        'lambda_N': svals_of_normalized[:, -1],
    }
    if svals_of_normalized.shape[1] > 1:
        singular_values.update(lambda_2=svals_of_normalized[:, 1])

    ranks = {
        'effective_rank_vetterli': effective_ranks,
        'approximate_rank_pca': approximate_ranks,
        'srank_kumar': sranks,
        'feature_rank_lyle': feature_ranks,
        'pytorch_rank': jnp_ranks,
    }

    out = {**singular_values, **ranks}

    return out

def compute_eval_stats(batches):
    """Compute performance stats from a batch of rollouts."""
    out = {
        "perf/max_timestep": jnp.max(batches["next", "step_count"]),
        "perf/min_timestep": jnp.min(batches["step_count"]),
        "perf/max_reward": jnp.max(batches["next", "reward"]),
        "perf/min_reward": jnp.min(batches["next", "reward"]),
        "perf/avg_reward": jnp.mean(batches["next", "reward"]),
    }
    if jnp.any(batches["next", "done"]):
        n_done = jnp.sum(batches["next", "done"])
        mask = 1 / batches["next", "done"] - 1
        returns_masked = batches["next", "return"] - mask
        max_return = jnp.max(returns_masked)
        avg_return = jnp.sum(batches["next", "return"] * batches["next", "done"]) / n_done
        max_end_timestep = jnp.max(batches["next", "step_count"] * batches["next", "done"])
        avg_end_timestep = jnp.sum(batches["next", "step_count"] * batches["next", "done"]) / n_done
        out.update({
            "perf/max_return": max_return,
            "perf/avg_return": avg_return,
            "perf/max_episode_timestep": max_end_timestep,
            "perf/avg_episode_timestep": avg_end_timestep,
        })

        if ("next", "return_raw") in batches.keys(include_nested=True):
            returns_raw_masked = batches["next", "return_raw"] - mask
            max_return_raw = jnp.max(returns_raw_masked)
            avg_return_raw = jnp.sum(batches["next", "return_raw"] * batches["next", "done"]) / n_done
            out.update({"perf/max_return_raw": max_return_raw, "perf/avg_return_raw": avg_return_raw})
    return out

def compute_effective_ranks(data_list, data_groups, data_features):
    stack = [data[data_feature] for i, data in enumerate(data_list) for data_feature in data_features[i]]
    groups = [f"{data_feature}_{group}" for i, group in enumerate(data_groups) for data_feature in data_features[i]]
    features = jnp.stack(stack, axis=0)
    try:
        ranks = compute_ranks_from_features(features)
    except Exception:
        return {}
    out = {}
    for rank_group, ranks_values in ranks.items():
        for i, data_feature_group in enumerate(groups):
            out[f"SVD/{rank_group}/{data_feature_group}"] = ranks_values[i]
    return out

def compute_feature_norms(data_list, data_groups, data_features):
    stack = [data[data_feature] for i, data in enumerate(data_list) for data_feature in data_features[i]]
    groups = [f"{data_feature}_{group}" for i, group in enumerate(data_groups) for data_feature in data_features[i]]
    features = jnp.stack(stack, axis=0)
    norms = jnp.linalg.norm(features, axis=-1).mean(axis=-1)
    means = features.mean(axis=-1).mean(axis=-1)
    stds = features.std(axis=-1).mean(axis=-1)
    out = {}
    for i, data_feature_group in enumerate(groups):
        out[f"feature_stats/norm_{data_feature_group}"] = norms[i]
        out[f"feature_stats/avg_{data_feature_group}"] = means[i]
        out[f"feature_stats/std_{data_feature_group}"] = stds[i]
    return out

def compute_dead_neurons(data_list, data_groups, data_feature, activation):
    stack = [data[data_feature] for data in data_list]
    groups = [f"{data_feature}_{group}" for group in data_groups]
    features = jnp.stack(stack, axis=0)
    dead_neurons = compute_dead_neurons_from_features(features, activation)
    out = {}
    for i, data_feature_group in enumerate(groups):
        out[f"dead_neurons/{data_feature_group}"] = dead_neurons[i]
    return out

def compute_dead_neurons_from_features(features, activation):
    TANH_STD_THRESHOLD = 0.001
    if activation in ["ReLU", "GELU"]:
        return jnp.sum(jnp.all(features == 0, axis=1), axis=-1)
    elif activation == "Tanh":
        return jnp.sum(jnp.std(features, axis=1) < TANH_STD_THRESHOLD, axis=-1)
    elif activation == "LeakyReLU":
        return jnp.sum(jnp.all(features < 0, axis=1), axis=-1)
    else:
        raise NotImplementedError(f"Activation {activation} not implemented.")

def compute_value_diversity(data):
    state_values = data["state_value"].squeeze(-1)
    return dict_with_prefix(
        {
            "avg_state_value": jnp.mean(state_values),
            "std_state_value": jnp.std(state_values),
        },
        "state_value_diversity/"
    )

def compute_policy_diversity(data, policy_module, compute_histograms=False):
    dist = policy_module.build_dist_from_params(data)
    if dist.__class__.__name__ == "OneHotCategorical":
        policies = dist.probs
    elif dist.__class__.__name__ in ["Normal", "TanhNormal"]:
        policies = jnp.stack([dist.loc, dist.scale], axis=1)
    else:
        raise NotImplementedError(f"Policy diversity not implemented for {dist.__class__.__name__}.")

    mean_policy = jnp.mean(policies, axis=0)
    policy_vars = jnp.var(policies, axis=0)
    policy_var = jnp.mean(policy_vars)
    logs = {"policy_variance": policy_var}
    if compute_histograms:
        logs.update({
            "means_hist_wandb": wandb.Histogram(
                np_histogram=(mean_policy.cpu().numpy(), np.arange(len(mean_policy) + 1))
            ),
            "vars_hist_wandb": wandb.Histogram(
                np_histogram=(policy_vars.cpu().numpy(), np.arange(len(policy_vars) + 1))
            ),
        })
    return dict_with_prefix(logs, "action_diversity/")

def compute_applied_grad_norm(params, prev_params):
    """Computes the norm of the difference between the current and previous policy parameters."""
    # Compute the difference between the current and previous parameters
    grads = jax.tree_map(lambda p, p_prev: p - p_prev, params, prev_params)
    
    # Compute the global gradient norm using Optax's `global_norm` utility
    grad_norm = optax.global_norm(grads)
    return grad_norm

def compute_weights_norm(params):
    """Computes the norm of the weights."""
    # Compute the global weight norm
    weight_norm = optax.global_norm(params)
    return weight_norm

def compute_kl_divergence_from_dists(old_dist, new_dist):
    """Computes the KL divergence between two distributions."""
    kl_divergence = old_dist.kl_divergence(new_dist)
    return {"kl_dist": kl_divergence.mean()}

def compute_kl_divergence_from_samples(old_log_probs, new_log_probs):
    """Computes the empirical KL divergence between two sample probabilities."""
    log_ratio = new_log_probs - old_log_probs  # log(q/p)
    ratio = jnp.exp(log_ratio)
    
    kl_naive = -jnp.mean(log_ratio)  # KL(p, q)
    kl_schulman = jnp.mean((ratio - 1) - log_ratio)  # Schulman's approximation
    
    return {"kl_sample_naive": kl_naive, "kl_sample_schulman": kl_schulman}

def compute_models_diff(data_outputs, data_targets, policy_module, reverse_kl=False, ratio_epsilon=0.0):
    """Computes the difference between models in terms of KL divergence and probability ratios."""
    dist_outputs = policy_module.build_dist_from_params(data_outputs)
    dist_targets = policy_module.build_dist_from_params(data_targets)

    # Reverse KL if needed
    p, q = (dist_outputs, dist_targets) if reverse_kl else (dist_targets, dist_outputs)

    # Compute KL divergence between the distributions
    kl_dist_batch = jnp.mean(p.kl_divergence(q))

    # Compute probability ratios
    actions = data_targets["action"]
    output_logprobs = dist_outputs.log_prob(actions)
    target_logprobs = dist_targets.log_prob(actions)
    ratio = jnp.exp(output_logprobs - target_logprobs)

    out = {
        "kl_dist_batch": kl_dist_batch,
        "max_prob_ratio": jnp.max(ratio),
        "min_prob_ratio": jnp.min(ratio),
        "avg_prob_ratio": jnp.mean(ratio),
        "std_prob_ratio": jnp.std(ratio),
    }

    # Compute additional metrics for clipped elements if `ratio_epsilon` is provided
    if ratio_epsilon > 0:
        above_epsilon = ratio > 1 + ratio_epsilon
        below_epsilon = ratio < 1 - ratio_epsilon
        
        for prefix, mask in zip(["above", "below"], [above_epsilon, below_epsilon]):
            if jnp.any(mask):
                ratio_clipped = jnp.where(mask, ratio, 0)
                out.update({
                    f"max_prob_ratio_{prefix}_epsilon": jnp.max(ratio_clipped),
                    f"min_prob_ratio_{prefix}_epsilon": jnp.min(ratio_clipped),
                    f"avg_prob_ratio_{prefix}_epsilon": jnp.mean(ratio_clipped),
                    f"std_prob_ratio_{prefix}_epsilon": jnp.std(ratio_clipped),
                })

    # L2 and cosine similarity metrics
    for prefix in ["features", "features_preactivation", "all_preactivations"]:
        for model in ["policy", "value"]:
            key = f"{prefix}_{model}"
            out[f"{key}_l2_batch"] = jnp.mean(jnp.square(data_outputs[key] - data_targets[key]))
            out[f"{key}_cosine"] = jnp.mean(optax.cosine_similarity(data_outputs[key], data_targets[key]))

    return out