# ==============================================================================
# MCMC_Samplers_prox_valMH.jl
#
# Validation-Driven Metropolis-Hastings SGLD Framework
# Features:
# - Strict Train / Validation / Test partitioning.
# - Out-of-sample generalization via Validation Pseudo-RSS.
# - Posterior predictive distribution generation on Test data.
# ==============================================================================

@everywhere begin
    using LinearAlgebra
    using Random
    using Distributions
    using Statistics
    using SpecialFunctions 

    # ==============================================================================
    # 0. GLOBAL PARAMETERS
    # ==============================================================================
    
    # Target acceptance rate for the hyperparameter MH step
    TARGET_ACCEPTANCE_RATE = 0.35
    # Number of iterations before tuning the random walk proposal scales
    ADAPTATION_WINDOW = 100
    
    # Parameters for the Inverse-Gamma prior placed on the variance (σ²)
    A0_INV_GAMMA = 0.01
    B0_INV_GAMMA = 0.01
    
    # Fallback delta for lambda if not defined in the global scope
    if !isdefined(Main, :DELTA_LAMBDA)
        const DELTA_LAMBDA = 0.1 
    end
    
    # Settings for the custom Newton-Raphson root finder used in Lq proximal operators
    NEWTON_MAX_ITER, NEWTON_TOL = 10, 1e-8 

    # ==============================================================================
    # 1. PROXIMAL OPERATORS (In-Place)
    # ==============================================================================
    # Proximal operators apply the shrinkage/sparsity penalties (like Lasso or Lq).
    # They are solved iteratively for arbitrary q-norms (where 0 < q < 1).

    function newton_root_finder_lq(v_abs::Float64, κλ::Float64, q::Float64)
        # Initialize the guess for the root
        u_k = v_abs / 2.0
        if u_k < 1e-10; u_k = 1e-6; end
        
        # Iterative Newton-Raphson method to find the threshold root
        for _ in 1:NEWTON_MAX_ITER
            f_u = u_k - v_abs + (κλ * q) * u_k^(q - 1.0)
            fp_u = 1.0 + (κλ * q * (q - 1.0)) * u_k^(q - 2.0)
            
            # Break early if the derivative is too flat (to avoid division by zero)
            if abs(fp_u) < 1e-12; break; end
            
            u_next = max(1e-10, u_k - f_u / fp_u)
            
            # Convergence check
            if abs(u_next - u_k) < NEWTON_TOL; return u_next; end
            u_k = u_next
        end
        return u_k
    end

    # Proximal operator for arbitrary Lq penalties (e.g., L_0.5, L_0.75)
    function prox_lq_newton!(out::Vector{Float64}, v::Vector{Float64}, κλ::Float64, q::Float64)
        # Calculate the theoretical threshold (τ_q) where the penalty overcomes the signal
        λ_q = (2.0 * κλ * (1.0 - q)) ^ (1.0 / (2.0 - q))
        τ_q = λ_q + κλ * q * λ_q ^ (q - 1.0)
        
        @inbounds for i in eachindex(v)
            abs_v_i = abs(v[i])
            # Hard thresholding: if signal is below threshold, set to absolute zero
            if abs_v_i < τ_q
                out[i] = 0.0
            else
                # Soft/Non-convex thresholding: find the optimal shrunken value
                u_k = abs_v_i 
                for _ in 1:NEWTON_MAX_ITER
                    f_u = u_k - abs_v_i + (κλ * q) * u_k^(q - 1.0)
                    fp_u = 1.0 + (κλ * q * (q - 1.0)) * u_k^(q - 2.0)
                    
                    if abs(fp_u) < 1e-12; break; end
                    
                    u_next = max(1e-10, u_k - f_u / fp_u)
                    if abs(u_next - u_k) < NEWTON_TOL
                        u_k = u_next
                        break
                    end
                    u_k = u_next
                end
                
                # Compare the cost of the shrunken value vs setting it to exactly zero
                cost_s = 0.5*(u_k - abs_v_i)^2 + κλ * u_k^q
                cost_0 = 0.5*abs_v_i^2
                # Assign the value that minimizes the overall cost
                out[i] = cost_s <= cost_0 ? sign(v[i])*u_k : 0.0
            end
        end
    end

    # Standard L1 (Lasso) Proximal Operator using Soft Thresholding
    function prox_l1!(out::Vector{Float64}, v::Vector{Float64}, κ::Float64)
        @inbounds for i in eachindex(v)
            out[i] = sign(v[i]) * max(0.0, abs(v[i]) - κ)
        end
    end
    
    # Analytical root for the L_1/2 penalty (faster than Newton method)
    half_thresh_root(v_abs, κ_λ) = (2/3)*v_abs * (1 + cos((2*π/3) - (2/3)*acos(min(1.0,max(-1.0,(3*sqrt(3)/4)*κ_λ/(v_abs^(3/2)))))))
    
    # Proximal operator specifically optimized for L_1/2
    function prox_l1_half!(out::Vector{Float64}, v::Vector{Float64}, κ_λ::Float64)
        thresh = (54/4)^(1/3) * κ_λ^(2/3)
        @inbounds for i in eachindex(v)
            abs_v_i = abs(v[i])
            # Apply analytical threshold
            out[i] = abs_v_i > thresh ? sign(v[i])*half_thresh_root(abs_v_i, κ_λ) : 0.0
        end
    end

    # ==============================================================================
    # 2. SCALAR LOG-POSTERIOR (Dimension-Scaled Exponential Prior)
    # ==============================================================================
    
    # Evaluates the posterior likelihood to decide whether to accept/reject hyperparameter changes.
    function log_posterior_scalar(RSS::Float64, penalty_norm::Float64, 
                                  n::Int, p::Int, 
                                  λ::Float64, σ²::Float64, q::Float64)
        
        # 1. Gaussian Log-Likelihood based on Residual Sum of Squares (RSS)
        log_lik = -0.5 * n * log(2π * σ²) - RSS / (2.0 * σ²)
        
        # 2. Log-Normalization Constant (LNC) for the generalized normal distribution prior
        log_C_part1 = log(q / 2.0) - loggamma(1.0 / q) 
        log_C_part2 = (1.0 / q) * (log(λ) - log(q) - log(σ²))
        log_norm_const = p * (log_C_part1 + log_C_part2)
        
        # 3. Penalty applied to the coefficients (β)
        log_prior_penalty = -(λ / (q * σ²)) * penalty_norm
        log_prior_beta = log_norm_const + log_prior_penalty
        
        # 4. Inverse-Gamma Prior on the variance (σ²)
        log_prior_sigma = -(A0_INV_GAMMA + 1.0)*log(σ²) - B0_INV_GAMMA/σ²
        
        # 5. DYNAMIC EXPONENTIAL PRIOR ON LAMBDA (λ)
        # Prevents λ from growing unbounded by capping it relative to dimension (p) and geometry (q).
        lambda_max = 100.0
        rate_lambda = p / (q * lambda_max)
        log_prior_lambda = log(rate_lambda) - rate_lambda * λ

        # Return the unnormalized log-posterior sum
        return log_lik + log_prior_beta + log_prior_sigma + log_prior_lambda
    end

    # ==============================================================================
    # 3. MAIN SAMPLER (Validation-Driven MH)
    # ==============================================================================
    
    function generic_sampler_valMH(y_train, X_train, y_val, X_val, num_iter, η₀, batch_size, model_config, penalty_name,
                                   burn_in::Int; thinning=1,
                                   γ_step=0.55, prop_scale_init=0.05)                        
        n_train, p = size(X_train)
        n_val = length(y_val)
        q_val = model_config.q
        
        # --- PRE-ALLOCATION ---
        # Pre-allocating memory prevents garbage collection overhead inside the MCMC loop.
        β = zeros(p)            # Model coefficients
        v = zeros(p)            # Gradient descent intermediate vector
        grad_β = zeros(p)       # Gradients
        
        resid_batch = zeros(batch_size)         # Training residuals
        resid_val = Vector{Float64}(undef, n_val) # Validation residuals
        
        # --- PRE-CALCULATION ---
        # Generate all random numbers upfront for speed
        indices_matrix = Matrix{Int}(undef, batch_size, num_iter)
        rand!(indices_matrix, 1:n_train)
        
        noise_matrix = Matrix{Float64}(undef, p, num_iter)
        randn!(noise_matrix)
        
        # --- INITIALIZE HYPERPARAMETERS ---
        hyper_params = Dict{Symbol, Float64}(:σ² => 1.0) 
        for hp in model_config.hyperparams
            hyper_params[hp] = model_config.init_vals[hp]
        end
        
        mh_params = [:σ²; model_config.hyperparams]
        
        # Tracking dictionaries for Metropolis-Hastings proposals and adaptation
        prop_scales = Dict(k => prop_scale_init for k in mh_params)
        acc_count = Dict(k => 0 for k in mh_params)
        cumulative_accepted = Dict(k => 0 for k in mh_params)
        adaptation_runs = 0

        # --- SAVING STRUCTURES ---
        num_saved = floor(Int, (num_iter - burn_in) / thinning)
        samples_β = Matrix{Float64}(undef, p, num_saved)
        samples_hyper = Dict(k => Vector{Float64}(undef, num_saved) for k in keys(hyper_params))
        save_idx = 1
        
        # Alias the proximal function specific to the chosen model
        prox_func! = model_config.prox_func!

        # ==========================================================================
        # CORE MCMC LOOP
        # ==========================================================================
        for t in 1:num_iter
            # Anneal the step size using the Robbins-Monro schedule
            η_t = max(η₀ * (1.0 + t)^(-γ_step), 1e-6)
            
            # --- 1. SGLD Step for β (Uses Training Data) ---
            # Extract mini-batch
            batch_indices = @view indices_matrix[:, t]
            X_B = @view X_train[batch_indices, :] 
            y_B = @view y_train[batch_indices]

            # Calculate gradients: X^T * (Xβ - y)
            mul!(resid_batch, X_B, β) 
            resid_batch .-= y_B
            mul!(grad_β, X_B', resid_batch)
            
            # Gradient Scaling: Adjust for mini-batch size and variance
            scaler = (n_train / batch_size / hyper_params[:σ²])
            grad_β .*= scaler
            
            noise_t = @view noise_matrix[:, t]
            
            # Tempered SGLD drift step (Cold Posterior)
            # T_temp shrinks the noise, allowing the threshold to perfectly trap useless variables to zero
            T_temp = 0.005 
            @. v = β - η_t * grad_β + sqrt(2.0 * η_t * T_temp) * noise_t
            
            # Proximal Step: Enforce sparsity (in-place modification of β)
            prox_func!(β, v, η_t, hyper_params)

            # --- 2. VALIDATION-DRIVEN MH Step for Hyperparameters ---
            # Evaluate current β strictly on out-of-sample validation data
            mul!(resid_val, X_val, β)
            resid_val .-= y_val
            
            # Pseudo-RSS trick: Scales validation error back to training volume 
            # to properly balance against the Prior Log-Normalization Constant
            RSS_val_raw = dot(resid_val, resid_val) 
            RSS_pseudo = RSS_val_raw * (n_train / n_val)
            
            # Calculate the current penalty norm (e.g., L1 sum)
            penalty_norm_curr = sum(abs(b)^q_val for b in β)

            # Propose and accept/reject hyperparameter updates
            for param_name in mh_params
                current_val = hyper_params[param_name]   
                
                # Propose a new value via Linear Random Walk
                prop_val = current_val + prop_scales[param_name] * randn()
                
                # Enforce physical constraints (e.g., variance must be strictly > 0)
                is_valid = true
                if haskey(model_config.constraints, param_name)
                    for constraint in model_config.constraints[param_name]
                        if !constraint(prop_val)
                            is_valid = false
                            break
                        end
                    end
                end

                if is_valid
                    # Temporarily update the parameter to evaluate the proposal
                    hyper_params[param_name] = prop_val
                    lp_prop = log_posterior_scalar(RSS_pseudo, penalty_norm_curr, n_train, p,
                                                   hyper_params[:λ], hyper_params[:σ²], q_val)

                    # Revert to current parameter to evaluate the baseline
                    hyper_params[param_name] = current_val
                    lp_curr = log_posterior_scalar(RSS_pseudo, penalty_norm_curr, n_train, p,
                                                   hyper_params[:λ], hyper_params[:σ²], q_val)

                    # Standard MH Acceptance criteria (Log-Space)
                    if log(rand()) < (lp_prop - lp_curr)
                        # Accept the proposal
                        hyper_params[param_name] = prop_val
                        acc_count[param_name] += 1
                        cumulative_accepted[param_name] += 1
                    end
                end
            end

            # --- 3. Adaptation ---
            # Periodically adjust the MH proposal scaling to hit the target acceptance rate (35%)
            if t <= burn_in && t % ADAPTATION_WINDOW == 0
                adaptation_runs += 1
                gain = 1.0 / (adaptation_runs^0.6) # Decaying adaptation gain
                for param in mh_params
                    acc_rate = cumulative_accepted[param] / ADAPTATION_WINDOW
                    prop_scales[param] *= exp(gain * (acc_rate - TARGET_ACCEPTANCE_RATE))
                    cumulative_accepted[param] = 0
                end
            end

            # --- 4. Saving ---
            # Save the states after the burn-in period, according to the thinning interval
            if t > burn_in && (t - burn_in) % thinning == 0
                if save_idx <= num_saved
                    samples_β[:, save_idx] = β
                    for k in keys(hyper_params)
                        samples_hyper[k][save_idx] = hyper_params[k]
                    end
                    save_idx += 1
                end
            end
        end
        
        # Compile acceptance rates for diagnostics
        acc_rates = Dict(string(k) => v/num_iter for (k,v) in acc_count)
        return merge((β=samples_β,), NamedTuple(samples_hyper), (acc_rates=acc_rates,))
    end

    # ==============================================================================
    # 4. POSTERIOR PREDICTIVE EVALUATION ON TEST SET
    # ==============================================================================

    """
    Evaluates the posterior predictive distribution of y_hat on the Test set,
    returning a distribution of predictive Mean Squared Errors (pMSEs) across all saved MCMC samples.
    """
    function evaluate_posterior_predictive(β_samples::Matrix{Float64}, X_test::Matrix{Float64}, y_test::Vector{Float64})
        n_samples = size(β_samples, 2)
        n_test = length(y_test)
        
        pMSE_dist = zeros(n_samples)
        y_hat_dist = zeros(n_test, n_samples)
        
        # Calculate predictions and errors for every single posterior sample
        for i in 1:n_samples
            β_i = @view β_samples[:, i]
            y_hat = X_test * β_i
            y_hat_dist[:, i] = y_hat
            pMSE_dist[i] = mean((y_hat .- y_test).^2)
        end
        
        # Extract summary statistics of the predictive distribution
        mean_pMSE = mean(pMSE_dist)
        median_pMSE = median(pMSE_dist)
        ci_lower = quantile(pMSE_dist, 0.025)
        ci_upper = quantile(pMSE_dist, 0.975)
        
        # Calculate ensemble average prediction (averaging predictions before scoring)
        y_hat_ensemble = mean(y_hat_dist, dims=2)[:]
        ensemble_pMSE = mean((y_hat_ensemble .- y_test).^2)
        
        return (
            pMSE_dist = pMSE_dist,
            y_hat_dist = y_hat_dist,
            mean_pMSE = mean_pMSE,
            median_pMSE = median_pMSE,
            ci_lower = ci_lower,
            ci_upper = ci_upper,
            ensemble_pMSE = ensemble_pMSE
        )
    end

    # ==============================================================================
    # 5. HIGH-LEVEL WORKER FUNCTION (TRAIN/VAL/TEST PIPELINE)
    # ==============================================================================

    # Orchestrates data splitting, step-size calculation, and invokes the sampler.
    function run_full_pipeline_valMH(chain_id, y, X, target_model, gamma_val, num_iter, burn_in, thinning;
                                     train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, 
                                     base_seed::Union{Int,Nothing}=nothing)
        # 1. Enforce Reproducibility
        if !isnothing(base_seed)
            Random.seed!(base_seed + chain_id * 100)
        else
            Random.seed!(42 + chain_id * 100)
        end
        
        n_total = length(y)
        n_train = floor(Int, train_ratio * n_total)
        n_val = floor(Int, val_ratio * n_total)
        # Test size is automatically whatever remains
        
        # 2. Strict Partitioning (Random Shuffling)
        perm = randperm(n_total)
        train_idx = perm[1:n_train]
        val_idx = perm[n_train+1 : n_train+n_val]
        test_idx = perm[n_train+n_val+1 : end]
        
        X_train, y_train = X[train_idx, :], y[train_idx]
        X_val, y_val = X[val_idx, :], y[val_idx]
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        # 3. Base Step Size (η₀) Calculation
        # Calculated via the Lipschitz constant, derived strictly from the Training data.
        L = try eigmax(Symmetric(X_train * X_train')) catch; opnorm(X_train, Inf) * opnorm(X_train, 1) end
        η₀ = 1.0 / L
        
        # 4. Run Validation-Driven SGLD Sampler
        batch_size = floor(Int, n_train / 3) # Using ~33% of the data per mini-batch
        results = generic_sampler_valMH(y_train, X_train, y_val, X_val, num_iter, η₀, batch_size, 
                                        target_model, target_model.name, burn_in; 
                                        thinning=thinning, γ_step=gamma_val)
        
        # 5. Posterior Predictive Evaluation on strictly held-out Test Data
        predictive_results = evaluate_posterior_predictive(results.β, X_test, y_test)
        
        # Assemble and Return the final Dictionary
        final_results = Dict(
            "β" => results.β, 
            "acc_rates" => results.acc_rates,
            "predictive" => predictive_results,
            "test_indices" => test_idx # Helpful for mapping back predictions to raw data
        )
        
        for k in keys(results)
            if k != :β && k != :acc_rates
                final_results[String(k)] = results[k]
            end
        end
        
        return final_results
    end

    # ==============================================================================
    # 5.5. LOWER-LEVEL WORKER FUNCTION (PRE-SPLIT DATA)
    # ==============================================================================
    
    # A leaner wrapper for when data is already split externally (e.g., in K-Fold CV)
    function run_single_chain(chain_id, y_train, X_train, y_val, X_val, num_iter, η0, batch_size, thinning, 
                              model_config, penalty_name, burn_in::Int, 
                              gamma_step::Float64=0.55; base_seed::Union{Int,Nothing}=nothing)
        if !isnothing(base_seed)
            Random.seed!(base_seed + chain_id * 100)
        else
            Random.seed!(42 + chain_id * 100)
        end
        
        results = generic_sampler_valMH(y_train, X_train, y_val, X_val, num_iter, η0, batch_size, model_config, penalty_name,
                                        burn_in; thinning=thinning,
                                        γ_step=gamma_step, prop_scale_init=0.05)
        
        final_results = Dict("β" => results.β, "acc_rates" => results.acc_rates)
        for k in keys(results)
            if k != :β && k != :acc_rates
                final_results[String(k)] = results[k]
            end
        end
        return final_results
    end

    # ==============================================================================
    # 6. MODEL CONFIGURATIONS
    # ==============================================================================
    # Centralized dictionary mapping model names to their specific q-norms, 
    # proximal functions, and constraints.
    MODELS = Dict(
        "Lasso" => (name="Lasso", q=1.0, 
                    prox_func! = (out,v,ηt,p) -> prox_l1!(out, v, ηt*p[:λ]/p[:σ²]), 
                    hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0])),
                    
        "L7_eighth" => (name="L7_eighth", q=7.0/8.0, 
                        prox_func! = (out,v,ηt,p) -> prox_lq_newton!(out, v, ηt*p[:λ]/(p[:σ²]*(7/8)), 7/8), 
                        hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0])),
                        
        "L3_fourth" => (name="L3_fourth", q=3.0/4.0, 
                        prox_func! = (out,v,ηt,p) -> prox_lq_newton!(out, v, ηt*p[:λ]/(p[:σ²]*(3/4)), 3/4), 
                        hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0])),
                        
        "L2_third" => (name="L2_third", q=2.0/3.0, 
                       prox_func! = (out,v,ηt,p) -> prox_lq_newton!(out, v, ηt*p[:λ]/(p[:σ²]*(2/3)), 2/3), 
                       hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0])),
                       
        "L1_half" => (name="L1_half", q=0.5, 
                      prox_func! = (out,v,ηt,p) -> prox_l1_half!(out, v, ηt*p[:λ]/(p[:σ²]*0.5)), 
                      hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0])),
                      
        "L1_third" => (name="L1_third", q=1.0/3.0, 
                       prox_func! = (out,v,ηt,p) -> prox_lq_newton!(out, v, ηt*p[:λ]/(p[:σ²]*(1/3)), 1/3), 
                       hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0])),
                       
        "L1_fourth" => (name="L1_fourth", q=1.0/4.0, 
                        prox_func! = (out,v,ηt,p) -> prox_lq_newton!(out, v, ηt*p[:λ]/(p[:σ²]*(1/4)), 1/4), 
                        hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0])),
                        
        "L1_eighth" => (name="L1_eighth", q=1.0/8.0, 
                        prox_func! = (out,v,ηt,p) -> prox_lq_newton!(out, v, ηt*p[:λ]/(p[:σ²]*(1/8)), 1/8), 
                        hyperparams=[:λ], init_vals=Dict(:λ=>1.0), constraints=Dict(:σ²=>[x->x>0],:λ=>[x->x>0]))
    )
end