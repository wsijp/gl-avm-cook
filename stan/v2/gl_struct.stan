// gl_avm_cook: the block graph-Laplacian (GL) model with well-defined location effects.
//
//   S (has_nbhd = 0):  y_i = intercept + x_i' beta + mu_t + u_s + eps_i
//   P (has_nbhd = 1):  y_i = intercept + x_i' beta + mu_t + zeta_k + psi_{k,t} + u_s + eps_i
//   eps ~ Student-t(nu, 0, sigEps)
//
// Every effect is the paper's whitened construction: effect = A z * sigma,
// z ~ N(0, 1), A = U Lambda^{-1/2} over the eigenvectors of a graph Laplacian with non-zero eigenvalue, so the null
// space is dropped and the effect sums to zero exactly over each whitened graph; the overall level is the intercept.
//   mu    township monthly effect       path graph over the 108 training months (RW1)
//   zeta  neighbourhood effect          neighbourhood graph, pieces joined by MST bridges (P only)
//   psi   neighbourhood x month         psi = A_comp psi_raw A_t' (the paper's Kronecker product; P only)
//   u     block-unit field              S: one graph over the township's units (all rook edges, pieces joined by
//                                       MST bridges): sums to zero over the township.
//                                       P: one graph per neighbourhood (rook edges within it, fragments joined by
//                                       MST bridges), block-diagonal A: sums to zero within every neighbourhood.
// The block whitener is scaled per whitened graph to unit geometric-mean marginal variance.
// Only the rows of A_u for units holding a training sale are passed (A_obs); u at every other unit is A_u z,
// computed from the draws afterwards (fit_struct.py), which is the exact joint prior. A_obs is passed dense
// (use_csr = 0, S) or as CSR (use_csr = 1, P's block-diagonal whitener).
// Priors (the earlier GL model's scales), Student-t, sigEps floor and nu bounds; regressors through QR.

data {
    int<lower=1> N;                        // training sales
    int<lower=2> Nt;                       // training months
    int<lower=1> K;                        // neighbourhoods (indexing only when has_nbhd = 0)
    int<lower=1> Ns_obs;                   // units holding a training sale
    int<lower=1> Nr;                       // regressors
    int<lower=0, upper=1> has_nbhd;

    array[N] int<lower=1, upper=Nt> sell;
    array[N] int<lower=1, upper=K> nbhd;
    array[N] int<lower=1, upper=Ns_obs> obs;
    vector[N] yVar;
    matrix[N, Nr] Q;                       // thin QR of the centred design: Q * sqrt(N - 1)
    matrix[Nr, Nr] R_inv;                  // beta = R_inv * theta

    int<lower=1> N_time_latent;
    matrix[Nt, N_time_latent] A_t;
    int<lower=0> N_comp_latent;            // 0 when has_nbhd = 0
    matrix[K, N_comp_latent] A_comp;

    int<lower=1> Nz;                       // latent dimension of the block field
    int<lower=0, upper=1> use_csr;
    matrix[use_csr ? 0 : Ns_obs, use_csr ? 0 : Nz] A_dense;
    int<lower=0> nnz;
    vector[nnz] csr_w;
    array[nnz] int csr_v;
    array[use_csr ? Ns_obs + 1 : 0] int csr_u;

    real y_init;
}

transformed data {
    array[N] int psi_idx;                  // position of psi[nbhd[i], sell[i]] in the column-major vector
    for (i in 1:N) {
        psi_idx[i] = (sell[i] - 1) * K + nbhd[i];
    }
}

parameters {
    real intercept;
    vector[Nr] theta;

    vector[N_time_latent] mu_raw;
    vector[N_comp_latent] zeta_raw;
    matrix[N_comp_latent, N_time_latent] psi_raw;
    vector[Nz] z;

    real<lower=0> sigma_mu;
    real<lower=0> sigma_u;
    array[has_nbhd] real<lower=0> sigma_zeta;
    array[has_nbhd] real<lower=0> sigma_psi;
    real<lower=0.01> sigEps;
    real<lower=2, upper=40> nu;
}

model {
    vector[Nt] mu = A_t * mu_raw * sigma_mu;
    vector[Ns_obs] u_obs;
    vector[N] eta;
    if (use_csr) {
        u_obs = csr_matrix_times_vector(Ns_obs, Nz, csr_w, csr_v, csr_u, z) * sigma_u;
    } else {
        u_obs = A_dense * z * sigma_u;
    }
    eta = intercept + Q * theta + mu[sell] + u_obs[obs];
    if (has_nbhd) {
        vector[K] zeta = A_comp * zeta_raw * sigma_zeta[1];
        vector[K * Nt] psi = to_vector(A_comp * psi_raw * A_t') * sigma_psi[1];   // column-major: (t - 1) * K + k
        eta += zeta[nbhd] + psi[psi_idx];
    }

    intercept ~ normal(y_init, 10);
    R_inv * theta ~ normal(0, 1);          // the earlier GL model's prior on beta; linear map, constant Jacobian

    mu_raw ~ normal(0, 1);
    zeta_raw ~ normal(0, 1);
    to_vector(psi_raw) ~ normal(0, 1);
    z ~ normal(0, 1);

    sigma_mu ~ normal(0, 0.1);
    sigma_u ~ normal(0, 0.5);
    sigma_zeta ~ normal(0, 0.5);
    sigma_psi ~ normal(0, 0.1);
    sigEps ~ normal(0, 0.25);
    nu ~ gamma(2, 0.1);

    yVar ~ student_t(nu, eta, sigEps);
}
