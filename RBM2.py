import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def vis_to_hid_prob(rbm_w, vis_data):
    return sigmoid(np.dot(rbm_w, vis_data))


def sample_bernoulli(probs):
    return probs > np.random.rand(probs.shape)


def hid_to_vis_prob(rbm_w, states):
    return sigmoid(np.dot(rbm_w.T, states))


def conf_godness(vis_state, hid_state):
    num_cases = vis_state.shape[0]
    d_G_by_rbm_w = (1/num_cases) * hid_state*vis_state.T


def cd1(rbm_w, vis_data):
    hid_prob = vis_to_hid_prob(rbm_w, vis_data)
    hid_states = sample_bernoulli(hid_prob)
    recon_data = hid_to_vis_prob(rbm_w, hid_states)
    neg_hid_prob = vis_to_hid_prob(rbm_w, recon_data)

    energy_gap = conf_godness(vis_data, hid_states) - conf_godness(recon_data, neg_hid_prob)

    return energy_gap

