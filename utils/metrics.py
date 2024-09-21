import jax
import jax.numpy as jnp


def compute_bias_and_kernel_norms_from_params(params):
    #import pdb; pdb.set_trace()
    layers=['Dense_0','Dense_6']
    norms_from_params={}
    for layer in layers:
        kernel = params[layer]['kernel']
        bias = params[layer]['bias'][jnp.newaxis, :]    
        norms_kernel =jnp.linalg.norm(kernel, axis=0, keepdims=True)
        mean_norms_kernel = jnp.mean(norms_kernel, axis=1, keepdims=True) 
        norm_bias = jnp.linalg.norm(bias, axis=1, keepdims=True)
        norms_from_params.update(
            {
                "mean_norm_"+layer+"_kernel": mean_norms_kernel,
                "mean_norm_"+layer+"_bias": norm_bias,
            }
        )
    #import pdb; pdb.set_trace()
    print(norms_from_params)
    return norms_from_params
    

def compute_norms_from_features(features):
    norms = jnp.linalg.norm(features, axis=1, keepdims=True)
    mean_norm = jnp.mean(norms, axis=0, keepdims=True) 
    return mean_norm
    
    