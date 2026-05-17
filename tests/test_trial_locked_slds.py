import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
import ssm.stats as stats
from ssm.messages import hmm_expected_states
from ssm.observations import TrialResetAutoRegressiveObservations
from ssm.transitions import TrialLockedTransitions


def test_trial_locked_transition_matrices():
    K, D = 3, 2
    trial_lengths = np.array([3, 2, 4])
    T = int(np.sum(trial_lengths))
    tag = {"trial_lengths": trial_lengths}

    transitions = TrialLockedTransitions(K, D)
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
    ])
    transitions.log_Ps = np.log(P)

    Ps = transitions.transition_matrices(
        np.zeros((T, D)), np.zeros((T, 0)), np.ones((T, D), dtype=bool), tag)

    boundary_idxs = {2, 4}
    for t in range(T - 1):
        expected = P if t in boundary_idxs else np.eye(K)
        assert np.allclose(Ps[t], expected)


def test_trial_locked_transition_m_step_uses_boundaries_only():
    K, D = 3, 2
    trial_lengths = np.array([3, 2, 4])
    T = int(np.sum(trial_lengths))
    tag = {"trial_lengths": trial_lengths}
    transitions = TrialLockedTransitions(K, D)

    Ezzp1 = np.ones((T - 1, K, K)) * 100.0
    boundary_counts = np.array([
        [3.0, 1.0, 0.0],
        [0.0, 2.0, 2.0],
        [1.0, 1.0, 2.0],
    ])
    Ezzp1[2] = boundary_counts
    Ezzp1[4] = 2.0 * boundary_counts
    expectations = [(np.ones((T, K)) / K, Ezzp1, 0.0)]

    transitions.m_step(
        expectations,
        [np.zeros((T, D))],
        [np.zeros((T, 0))],
        [np.ones((T, D), dtype=bool)],
        [tag],
    )

    expected_P = 3.0 * boundary_counts
    expected_P = expected_P / expected_P.sum(axis=1, keepdims=True)
    assert np.allclose(transitions.transition_matrix, expected_P)


def test_trial_locked_forward_backward_keeps_states_constant_within_trials():
    K, D = 4, 2
    trial_lengths = np.array([3, 4])
    T = int(np.sum(trial_lengths))
    tag = {"trial_lengths": trial_lengths}
    transitions = TrialLockedTransitions(K, D)
    Ps = transitions.transition_matrices(
        np.zeros((T, D)), np.zeros((T, 0)), np.ones((T, D), dtype=bool), tag)

    pi0 = np.ones(K) / K
    log_likes = npr.randn(T, K)
    Ez, _, _ = hmm_expected_states(pi0, Ps, log_likes)

    start = 0
    for length in trial_lengths:
        stop = start + int(length)
        assert np.allclose(Ez[start:stop], Ez[start])
        start = stop


def test_trial_locked_slds_discrete_update_uses_trial_chain():
    N, K, D = 5, 3, 2
    trial_lengths = np.array([4, 3])
    T = int(np.sum(trial_lengths))
    tag = {"trial_lengths": trial_lengths}
    model = ssm.SLDS(
        N,
        K,
        D,
        transitions="trial_locked",
        dynamics="trial_gaussian",
        emissions="gaussian",
        single_subspace=False,
    )

    pi0 = np.array([0.2, 0.3, 0.5])
    Ps = model.transitions.transition_matrices(
        np.zeros((T, D)), np.zeros((T, 0)), np.ones((T, D), dtype=bool), tag)
    log_likes = -1000.0 * np.ones((T, K))
    log_likes[0, 0] = 0.0
    log_likes[1, 1] = 0.0
    log_likes[2, 2] = 0.0
    log_likes[3, 0] = 0.0
    log_likes[4:, 1] = 0.0

    prms = model._trial_locked_discrete_state_params(pi0, Ps, log_likes, tag)
    Ez, Ezzp1, normalizer = prms["expectations"]
    assert np.isfinite(normalizer)
    assert np.all(np.isfinite(Ez))
    assert np.all(np.isfinite(Ezzp1))

    trial_log_likes = np.array([
        np.sum(log_likes[:4], axis=0),
        np.sum(log_likes[4:], axis=0),
    ])
    E_trials, E_trial_joints, manual_normalizer = \
        hmm_expected_states(pi0, Ps[[3]], trial_log_likes)

    assert np.allclose(normalizer, manual_normalizer)
    assert np.allclose(Ez[:4], E_trials[0])
    assert np.allclose(Ez[4:], E_trials[1])
    assert np.allclose(Ezzp1[:3], np.diag(E_trials[0]))
    assert np.allclose(Ezzp1[3], E_trial_joints[0])
    assert np.allclose(Ezzp1[4:], np.diag(E_trials[1]))


def test_trial_reset_dynamics_likelihood_matches_manual_trial_sum():
    K, D = 2, 2
    trial_lengths = np.array([2, 3])
    T = int(np.sum(trial_lengths))
    tag = {"trial_lengths": trial_lengths}
    dynamics = TrialResetAutoRegressiveObservations(K, D)
    data = npr.randn(T, D)
    inputs = np.zeros((T, 0))
    mask = np.ones_like(data, dtype=bool)

    lls = dynamics.log_likelihoods(data, inputs, mask, tag)

    manual = np.zeros((T, K))
    starts = np.array([0, 2])
    start_mask = np.zeros(T, dtype=bool)
    start_mask[starts] = True
    targets = np.where(~start_mask)[0]
    for k in range(K):
        manual[starts, k] = stats.multivariate_normal_logpdf(
            data[starts], dynamics.mu_init[k], dynamics.Sigmas_init[k])
        mus = data[targets - 1].dot(dynamics.As[k].T) + dynamics.bs[k]
        manual[targets, k] = stats.multivariate_normal_logpdf(
            data[targets], mus, dynamics.Sigmas[k])

    assert np.allclose(lls, manual)


def test_trial_reset_dynamics_statistics_and_hessian_skip_boundaries():
    K, D = 1, 1
    trial_lengths = np.array([3, 2])
    T = int(np.sum(trial_lengths))
    tag = {"trial_lengths": trial_lengths}
    dynamics = TrialResetAutoRegressiveObservations(K, D)
    data = np.array([[0.0], [1.0], [2.0], [100.0], [101.0]])
    inputs = np.zeros((T, 0))
    Ez = np.ones((T, K))
    expectations = [(Ez, None, None)]

    _, _, _, Ens, init_Exs, _, init_Ens = dynamics._get_sufficient_statistics(
        expectations, [data], [inputs], [tag])
    assert np.allclose(Ens, np.array([3.0]))
    assert np.allclose(init_Ens, np.array([2.0]))
    assert np.allclose(init_Exs, np.array([[100.0]]))

    dynamics.m_step(
        expectations,
        [data],
        [inputs],
        [np.ones_like(data, dtype=bool)],
        [tag],
    )
    assert np.allclose(dynamics.mu_init[0], np.array([50.0]))

    J_ini, J11, J21, J22 = dynamics.neg_hessian_expected_log_dynamics_prob(
        Ez, data, inputs, np.ones_like(data, dtype=bool), tag)
    assert J_ini.shape == (D, D)
    assert np.allclose(J11[2], 0.0)
    assert np.allclose(J21[2], 0.0)
    assert np.all(J22[2] > 0.0)


def test_trial_locked_slds_laplace_em_smoke():
    N, K, D = 10, 16, 2
    num_trials, trial_len = 3, 4
    T = num_trials * trial_len
    tag = {"trial_lengths": np.full(num_trials, trial_len)}
    data = 0.1 * npr.randn(T, N)

    model = ssm.SLDS(
        N,
        K,
        D,
        transitions="trial_locked",
        dynamics="trial_gaussian",
        emissions="gaussian",
        single_subspace=False,
    )
    elbos, posterior = model.fit(
        [data],
        tags=[tag],
        initialize=False,
        num_iters=1,
        continuous_maxiter=1,
        emission_optimizer_maxiter=1,
        verbose=0,
    )

    assert np.all(np.isfinite(elbos))
    xhat = posterior.mean_continuous_states[0]
    zhat = model.most_likely_trial_states(xhat, data, tag=tag)
    assert zhat.shape == (T,)
    for n in range(num_trials):
        z_trial = zhat[n * trial_len:(n + 1) * trial_len]
        assert np.all(z_trial == z_trial[0])
