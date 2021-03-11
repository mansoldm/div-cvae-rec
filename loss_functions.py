import torch
import torch.nn.functional as F


def kl_divergence_posterior_and_prior(Q_mean, Q_log_var, P_mean, P_log_var):
    P_var_inverse = 1 / torch.exp(P_log_var)
    var_ratio_term = torch.exp(Q_log_var) * P_var_inverse

    N_mean = Q_mean - P_mean
    mean_term = N_mean.pow(2) * P_var_inverse

    kl = 0.5 * torch.sum(P_log_var - 1 - Q_log_var + var_ratio_term + mean_term)

    return kl


def reconstruction_loss(k_head_unnormalized, x):
    # make reconstruction (batch_size, num_items, slate_size)
    k_head_unnormalized = k_head_unnormalized.permute(0, 2, 1)
    loss = F.cross_entropy(input=k_head_unnormalized, target=x)

    return loss


def loss_function_posterior_and_prior(x, K_head_unnormalized, Q_mean, Q_log_var, P_mean, P_log_var):
    kl = kl_divergence_posterior_and_prior(Q_mean, Q_log_var, P_mean, P_log_var)
    reconstruction = reconstruction_loss(K_head_unnormalized, x)

    return {'loss': reconstruction + kl, 'KL': kl, 'recon': reconstruction}

