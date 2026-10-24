// gl_avm_cook, model development: model P (gl_struct.stan, has_nbhd = 1) with optional
// segment terms. With every option off it is model P term for term (same parameters in the same order).
//
//   y_i = intercept + x_i' beta + mu_t + zeta_k + psi_{k,t} + u_s                        (P)
//         + dmu_{g(i), t}            segment deviation of the township path       (n_dseg > 0)
//         + zf_k        [flats]      flat neighbourhood level                     (has_zf)
//         + psif_{k,t}  [flats]      flat neighbourhood x month deviation         (has_psif)
//         + g_{b(i)}    [resales]    premium curve over gap bins since the last sale (n_gbin > 0)
//         + a w(g_i) exp(-g_i / tau) [resales]  decaying premium in the years g_i since the last sale
//                                        (has_gexp); w = 1 - exp(-g / delta) with delta fixed (a ramp), or w = 1 (delta 0)
//         + sum_j x_ij v_{k(i), j}         neighbourhood deviations of the coefficients of chosen covariates
//                                        (n_nvc > 0): x_ij the centred covariate, v_.j whitened on the
//                                        neighbourhood graph (as zeta), so it sums to zero over neighbourhoods and the
//                                        township coefficient stays in beta
//         + eps_i,  eps ~ Student-t(nu, 0, sigEps)
//
// Every added effect is whitened like the others (effect = A z * sigma, z ~ N(0, 1), null space dropped), so it sums
// to zero exactly over its graph: dmu over months (the segment's level stays in the class indicators among the
// regressors), zf over neighbourhoods (the flats' overall level stays in the class indicators), psif over months for
// each neighbourhood and over neighbourhoods for each month. Each has its own scale with the prior of its P counterpart.
// g: a random walk over the n_gbin gap bins (path graph, whitened like the month path), summing
// to zero over the bins; the resale level stays in hist_sold among the regressors; sales with no earlier sale (gbin 0)
// get none.
// a exp(-g / tau): the resale premium fades to zero as the gap g (years) grows, so a home last sold
// long ago is valued like one with no earlier sale since 1999; a ~ N(0, gexp_a_sd), tau ~ lognormal(gexp_tau_mu,
// gexp_tau_sd) (an informative prior of a few years); a home with no earlier sale (gexp_sold 0) gets none. With
// gexp_delta > 0 the premium also rises from 0 over the first months after a sale (a renovation takes time).
// Forecasting time model (has_fc): the township path becomes mu_t = trend_t + s_{c(t)}. The trend is
// an ARIMA(1,1,0) with drift (a damped trend): its monthly increments d_t = delta + phi (d_{t-1} - delta) + sigma_mu e_t
// (stationary AR(1) around the drift; d_1 from the stationary distribution), e = mu_raw ~ N(0, 1) (the same Nt - 1
// innovations and the same sigma_mu as the random walk, which is the case phi = delta = 0), cumulated from 0. s = A_c
// s_raw sigma_s is a calendar-month effect over the 12 months, A_c the whitened cyclic-graph Laplacian (Dec joined to
// Jan): it sums to zero over the 12 months and is smoothed across neighbouring months; c(t) = the calendar month of month
// t. The whole path is centred over the training months (b = its training mean; the level stays in the intercept), and
// the forecast subtracts the same b. psi keeps its random-walk time structure.
// Generated quantities forecast both beyond the last training month, h = 1..H months ahead, from their own dynamics:
// the trend's increments continue the AR(1) with new innovations; psi continues its random walk,
// psi_{.,T+h} = psi_{.,T} + sigma_psi A_comp (xi_1 + ... + xi_h), xi ~ N(0, I) (the conditional of the next month under
// the Kronecker prior); and the latent value of each evaluation sale at its own month (lat_dev) is formed in Stan, as
// posterior draws, beside the same value with every time term held at the last training month (lat_dev_apr; the
// previous rule, a diagnostic). Evaluation rows carry no prices.

functions {
    // increments of the damped trend (has_fc): stationary AR(1) around delta with innovation SD s
    vector fc_incr(vector e, real s, real delta, real phi) {
        int n = rows(e);
        vector[n] d;
        d[1] = delta + s / sqrt(1 - square(phi)) * e[1];
        for (j in 2:n) d[j] = delta + phi * (d[j - 1] - delta) + s * e[j];
        return d;
    }
    // the trend before centring: increments cumulated from 0
    vector fc_cum(vector d) {
        int n = rows(d) + 1;
        vector[n] tr;
        tr[1] = 0;
        for (j in 1:(n - 1)) tr[j + 1] = tr[j] + d[j];
        return tr;
    }
}

data {
    int<lower=1> N;
    int<lower=2> Nt;
    int<lower=1> K;
    int<lower=1> Ns_obs;
    int<lower=1> Nr;

    array[N] int<lower=1, upper=Nt> sell;
    array[N] int<lower=1, upper=K> nbhd;
    array[N] int<lower=1, upper=Ns_obs> obs;
    vector[N] yVar;
    matrix[N, Nr] Q;
    matrix[Nr, Nr] R_inv;

    int<lower=1> N_time_latent;
    matrix[Nt, N_time_latent] A_t;
    int<lower=1> N_comp_latent;
    matrix[K, N_comp_latent] A_comp;

    int<lower=1> Nz;
    int<lower=0> nnz;
    vector[nnz] csr_w;
    array[nnz] int csr_v;
    array[Ns_obs + 1] int csr_u;

    real y_init;

    // segment options
    int<lower=0> n_dseg;                   // segments with their own township path deviation (0 = off)
    array[N] int<lower=0, upper=n_dseg> dseg;   // 0 = reference segment (single-family), else the deviation used
    int<lower=0, upper=1> has_zf;
    int<lower=0, upper=1> has_psif;
    array[N] int<lower=0, upper=1> is_flat;

    // gap-curve option
    int<lower=0> n_gbin;                   // gap bins (0 = off)
    array[N] int<lower=0, upper=n_gbin> gbin;   // 0 = no earlier sale
    matrix[n_gbin, n_gbin > 0 ? n_gbin - 1 : 0] A_g;

    // decaying-premium option
    int<lower=0, upper=1> has_gexp;
    array[N] int<lower=0, upper=1> gexp_sold;   // 1 = an earlier sale since 1999
    vector<lower=0>[N] gexp_years;              // years since the latest earlier sale (0 where none)
    real gexp_tau_mu;                           // prior of tau: lognormal(gexp_tau_mu, gexp_tau_sd), years
    real<lower=0> gexp_tau_sd;
    real<lower=0> gexp_a_sd;                    // prior of a: normal(0, gexp_a_sd)
    real<lower=0> gexp_delta;                   // ramp time in years (0 = no ramp)

    // neighbourhood-varying coefficients option
    int<lower=0> n_nvc;                         // covariates with neighbourhood deviations (0 = off)
    matrix[N, n_nvc] X_nvc;                     // their centred values

    // forecasting time model
    int<lower=0, upper=1> has_fc;
    array[has_fc ? Nt : 0] int<lower=1, upper=12> cal;    // calendar month of each training month
    matrix[has_fc ? 12 : 0, has_fc ? 11 : 0] A_c;         // whitened cyclic-graph Laplacian over the calendar months
    real<lower=0> fc_delta_sd;                            // prior of the drift: normal(0, fc_delta_sd), per month
    real<lower=0> fc_sigma_s_sd;                          // prior of sigma_s: half-normal(0, fc_sigma_s_sd)
    int<lower=0> H;                                       // months forecast beyond the last training month
    array[has_fc ? H : 0] int<lower=1, upper=12> cal_fc;
    int<lower=0> N_dev;                                   // evaluation sales predicted in Stan (0 unless has_fc)
    array[N_dev] int<lower=1, upper=H> h_dev;             // months after the last training month
    array[N_dev] int<lower=1, upper=K> nbhd_dev;
    matrix[N_dev, Nr] Q_dev;                              // (x - xbar) R^-1, so Q_dev theta = (x - xbar)' beta
    int<lower=0> nnz_dev;                                 // CSR rows of the block whitener at the evaluation sales' nodes
    vector[nnz_dev] csr_w_dev;
    array[nnz_dev] int csr_v_dev;
    array[N_dev + 1] int csr_u_dev;
    array[N_dev] int<lower=0, upper=1> gexp_sold_dev;
    vector<lower=0>[N_dev] gexp_years_dev;
}

transformed data {
    array[N] int psi_idx;
    int Nf = sum(is_flat);
    array[Nf] int flat_rows;
    int Ng = has_gexp ? sum(gexp_sold) : 0;
    array[Ng] int gexp_rows;
    vector[Ng] gexp_g;
    vector[Ng] gexp_w;
    if (has_fc && N_time_latent != Nt - 1) reject("has_fc: mu_raw must hold the Nt - 1 innovations");
    {
        int j = 1;
        for (i in 1:N) {
            psi_idx[i] = (sell[i] - 1) * K + nbhd[i];
            if (is_flat[i]) {
                flat_rows[j] = i;
                j += 1;
            }
        }
        if (has_gexp) {
            int k = 1;
            for (i in 1:N) {
                if (gexp_sold[i]) {
                    gexp_rows[k] = i;
                    gexp_g[k] = gexp_years[i];
                    gexp_w[k] = gexp_delta > 0 ? 1 - exp(-gexp_years[i] / gexp_delta) : 1;
                    k += 1;
                }
            }
        }
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
    array[1] real<lower=0> sigma_zeta;
    array[1] real<lower=0> sigma_psi;
    real<lower=0.01> sigEps;
    real<lower=2, upper=40> nu;

    matrix[N_time_latent, n_dseg] dmu_raw;
    vector<lower=0>[n_dseg] sigma_dmu;
    vector[has_zf ? N_comp_latent : 0] zf_raw;
    array[has_zf] real<lower=0> sigma_zf;
    matrix[has_psif ? N_comp_latent : 0, has_psif ? N_time_latent : 0] psif_raw;
    array[has_psif] real<lower=0> sigma_psif;
    vector[n_gbin > 0 ? n_gbin - 1 : 0] g_raw;
    array[n_gbin > 0 ? 1 : 0] real<lower=0> sigma_g;
    array[has_gexp] real gexp_a;
    array[has_gexp] real<lower=0> gexp_tau;
    matrix[N_comp_latent, n_nvc] nvc_raw;
    vector<lower=0>[n_nvc] sigma_nvc;
    array[has_fc] real fc_delta;
    array[has_fc] real<lower=-1, upper=1> fc_phi;
    vector[has_fc ? 11 : 0] s_raw;
    array[has_fc] real<lower=0> sigma_s;
}

model {
    vector[Nt] mu;
    if (has_fc) {
        vector[12] sea = A_c * s_raw * sigma_s[1];
        mu = fc_cum(fc_incr(mu_raw, sigma_mu, fc_delta[1], fc_phi[1])) + sea[cal];
        mu -= mean(mu);
    } else {
        mu = A_t * mu_raw * sigma_mu;
    }
    vector[Ns_obs] u_obs = csr_matrix_times_vector(Ns_obs, Nz, csr_w, csr_v, csr_u, z) * sigma_u;
    vector[K] zeta = A_comp * zeta_raw * sigma_zeta[1];
    vector[K * Nt] psi = to_vector(A_comp * psi_raw * A_t') * sigma_psi[1];
    vector[N] eta = intercept + Q * theta + mu[sell] + u_obs[obs] + zeta[nbhd] + psi[psi_idx];
    if (n_dseg > 0) {
        matrix[Nt, n_dseg] dmu = A_t * dmu_raw * diag_matrix(sigma_dmu);
        for (i in 1:N) {
            if (dseg[i] > 0) eta[i] += dmu[sell[i], dseg[i]];
        }
    }
    if (has_zf) {
        vector[K] zf = A_comp * zf_raw * sigma_zf[1];
        eta[flat_rows] += zf[nbhd[flat_rows]];
    }
    if (has_psif) {
        vector[K * Nt] psif = to_vector(A_comp * psif_raw * A_t') * sigma_psif[1];
        eta[flat_rows] += psif[psi_idx[flat_rows]];
    }
    if (n_gbin > 0) {
        vector[n_gbin] g = A_g * g_raw * sigma_g[1];
        for (i in 1:N) {
            if (gbin[i] > 0) eta[i] += g[gbin[i]];
        }
    }
    if (has_gexp) {
        eta[gexp_rows] += gexp_a[1] * (gexp_w .* exp(-gexp_g / gexp_tau[1]));
    }
    if (n_nvc > 0) {
        matrix[K, n_nvc] nvc = A_comp * nvc_raw * diag_matrix(sigma_nvc);
        for (j in 1:n_nvc) eta += X_nvc[:, j] .* nvc[nbhd, j];
    }

    intercept ~ normal(y_init, 10);
    R_inv * theta ~ normal(0, 1);

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

    to_vector(dmu_raw) ~ normal(0, 1);
    sigma_dmu ~ normal(0, 0.1);
    zf_raw ~ normal(0, 1);
    sigma_zf ~ normal(0, 0.5);
    to_vector(psif_raw) ~ normal(0, 1);
    sigma_psif ~ normal(0, 0.1);
    g_raw ~ normal(0, 1);
    sigma_g ~ normal(0, 0.1);
    gexp_a ~ normal(0, gexp_a_sd);
    gexp_tau ~ lognormal(gexp_tau_mu, gexp_tau_sd);
    to_vector(nvc_raw) ~ normal(0, 1);
    sigma_nvc ~ normal(0, 0.1);
    fc_delta ~ normal(0, fc_delta_sd);
    if (has_fc) target += beta_lpdf((fc_phi[1] + 1) / 2 | 2, 2);
    s_raw ~ normal(0, 1);
    sigma_s ~ normal(0, fc_sigma_s_sd);

    yVar ~ student_t(nu, eta, sigEps);
}

generated quantities {
    vector[has_fc ? Nt : 0] mu_fit;                       // the fitted township path, training months
    vector[has_fc ? 12 : 0] season;                       // s, by calendar month
    vector[has_fc ? H : 0] mu_fc;                         // the forecast township path, months T + 1 .. T + H
    matrix[has_fc ? K : 0, has_fc ? H : 0] psi_fc;        // the forecast neighbourhood x month deviations
    vector[N_dev] lat_dev;                                // evaluation sales at their own month
    vector[N_dev] lat_dev_apr;                            // the same with every time term held at month T (diagnostic)
    if (has_fc) {
        vector[Nt - 1] d = fc_incr(mu_raw, sigma_mu, fc_delta[1], fc_phi[1]);
        vector[K] psi_T = A_comp * (psi_raw * A_t[Nt]') * sigma_psi[1];
        vector[K] cum = psi_T;
        real dl = d[Nt - 1];
        real trl;
        real b;
        season = A_c * s_raw * sigma_s[1];
        {
            vector[Nt] tr = fc_cum(d);
            mu_fit = tr + season[cal];
            b = mean(mu_fit);
            mu_fit -= b;
            trl = tr[Nt];
        }
        for (h in 1:H) {
            dl = fc_delta[1] + fc_phi[1] * (dl - fc_delta[1]) + sigma_mu * normal_rng(0, 1);
            trl += dl;
            mu_fc[h] = trl + season[cal_fc[h]] - b;
            cum += A_comp * to_vector(normal_rng(rep_vector(0, N_comp_latent), 1)) * sigma_psi[1];
            psi_fc[:, h] = cum;
        }
        if (N_dev > 0) {
            vector[K] zeta = A_comp * zeta_raw * sigma_zeta[1];
            vector[N_dev] base = intercept + Q_dev * theta + zeta[nbhd_dev]
                                 + csr_matrix_times_vector(N_dev, Nz, csr_w_dev, csr_v_dev, csr_u_dev, z) * sigma_u;
            if (has_gexp) {
                for (i in 1:N_dev) {
                    if (gexp_sold_dev[i]) {
                        real w = gexp_delta > 0 ? 1 - exp(-gexp_years_dev[i] / gexp_delta) : 1;
                        base[i] += gexp_a[1] * w * exp(-gexp_years_dev[i] / gexp_tau[1]);
                    }
                }
            }
            for (i in 1:N_dev) {
                lat_dev[i] = base[i] + mu_fc[h_dev[i]] + psi_fc[nbhd_dev[i], h_dev[i]];
                lat_dev_apr[i] = base[i] + mu_fit[Nt] + psi_T[nbhd_dev[i]];
            }
        }
    }
}
