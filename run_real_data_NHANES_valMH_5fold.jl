# ==============================================================================
# run_real_data_NHANES_valMH_5Fold.jl
#
# ANALYSIS: NHANES 2017-2018 (Predicting Systolic Blood Pressure)
# REGIME: n > p (n=3050, p=774)
# GEOMETRY: Fully Standardized X and y (Mandatory for mixed clinical data).
# ==============================================================================

using Distributed
using DataFrames, CSV
using LinearAlgebra, Random, Statistics, Distributions
using MCMCDiagnosticTools 

# --- CONFIG ---
const N_CHAINS = 4
if nprocs() < N_CHAINS
    addprocs(N_CHAINS - nprocs() + 1)
end

const DATA_FILE = "nhanes_sbp.csv"
const TARGET_COL = "SBP"
const SEED_SPLIT = 2024
const NUM_FOLDS = 5

const NUM_ITER = 250000
const THINNING = 25
const BURN_IN  = 100000

const PENALTY_NAMES = [
    "Lasso", "L7_eighth", "L3_fourth", "L2_third", 
    "L1_half", "L1_third", "L1_fourth", "L1_eighth"
]

@everywhere include("MCMC_Samplers_prox_valMH.jl")

@everywhere begin
    using Random
    function run_prox_real_wrapper(chain_id, y_t, X_t, y_v, X_v, n_iter, eta0, bs, thin, cfg, pname, burn, fold_k)
        rep_seed = 1234 + chain_id * 100 + fold_k
        Random.seed!(rep_seed)
        
        res = run_single_chain(
            chain_id, y_t, X_t, y_v, X_v, 
            n_iter, eta0, bs, thin, cfg, pname, 
            burn, 0.55; base_seed=rep_seed
        )
        return Dict("β" => res["β"], "σ²" => res["σ²"], "λ" => res["λ"])
    end
end

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

function load_data_and_create_chunks(file_path, target_col, seed, k_folds)
    if !isfile(file_path); error("File not found: $file_path"); end
    df = CSV.read(file_path, DataFrame)
    
    if !(target_col in names(df))
        error("Target column '$target_col' not found in dataset!")
    end
    
    y_raw = Float64.(df[!, target_col])
    X_raw = Matrix{Float64}(select(df, Not(target_col)))
    
    n = length(y_raw)
    Random.seed!(seed)
    shuffled_idx = randperm(n)
    
    fold_size = floor(Int, n / k_folds)
    chunks = [shuffled_idx[(i-1)*fold_size + 1 : (i == k_folds ? n : i*fold_size)] for i in 1:k_folds]
    
    return y_raw, X_raw, chunks
end

function get_fold_data(y_raw, X_raw, chunks, fold_k, k_folds)
    test_idx = chunks[fold_k]
    val_idx_k = mod1(fold_k + 1, k_folds)
    val_idx = chunks[val_idx_k]
    
    train_idx = vcat([chunks[j] for j in 1:k_folds if j != fold_k && j != val_idx_k]...)
    
    y_train = y_raw[train_idx]
    X_train = X_raw[train_idx, :]
    y_val   = y_raw[val_idx]
    X_val   = X_raw[val_idx, :]
    y_test  = y_raw[test_idx]
    X_test  = X_raw[test_idx, :]
    
    # --- FULL STANDARDIZATION (Required for mixed clinical units) ---
    μ_X = mean(X_train, dims=1)
    σ_X = std(X_train, dims=1)
    σ_X[σ_X .< 1e-8] .= 1.0 
    
    X_train_std = (X_train .- μ_X) ./ σ_X
    X_val_std   = (X_val .- μ_X) ./ σ_X
    X_test_std  = (X_test .- μ_X) ./ σ_X
    
    μ_y = mean(y_train)
    σ_y = std(y_train)
    if σ_y < 1e-8; σ_y = 1.0; end
    
    y_train_std = (y_train .- μ_y) ./ σ_y
    y_val_std   = (y_val .- μ_y) ./ σ_y
    y_test_std  = (y_test .- μ_y) ./ σ_y
    
    return y_train_std, X_train_std, y_val_std, X_val_std, y_test_std, X_test_std
end

function calculate_stats(vec)
    clean_vec = filter(!isnan, vec)
    if isempty(clean_vec)
        return (med = NaN, lo = NaN, hi = NaN)
    end
    return (med = median(clean_vec), lo = quantile(clean_vec, 0.025), hi = quantile(clean_vec, 0.975))
end

function calculate_predictive_metrics(X_test, y_test, β_samples)
    n_samples = size(β_samples, 2)
    n_test = length(y_test)
    
    B = [abs(y_test[i] - y_test[j]) for i in 1:n_test, j in 1:n_test]
    B_centered = B .- mean(B, dims=1) .- mean(B, dims=2) .+ mean(B)
    dVarYY = mean(B_centered.^2)
    
    mse_vec = zeros(n_samples); pears_vec = zeros(n_samples); dcor_vec = zeros(n_samples)
    
    for s in 1:n_samples
        y_pred = X_test * β_samples[:, s]
        mse_vec[s] = mean((y_test .- y_pred).^2)
        
        if var(y_pred) > 1e-12
            pears_vec[s] = cor(y_test, y_pred)
        else
            pears_vec[s] = 0.0
        end
        
        A = [abs(y_pred[i] - y_pred[j]) for i in 1:n_test, j in 1:n_test]
        A_centered = A .- mean(A, dims=1) .- mean(A, dims=2) .+ mean(A)
        dCov2 = mean(A_centered .* B_centered)
        dVarXX = mean(A_centered.^2)
        
        if dVarXX > 1e-12 && dVarYY > 1e-12
            dcor_vec[s] = sqrt(max(0.0, dCov2)) / sqrt(sqrt(dVarXX) * sqrt(dVarYY))
        else
            dcor_vec[s] = 0.0
        end
    end
    
    return (
        mse_med = median(filter(!isnan, mse_vec)), mse_lo = quantile(filter(!isnan, mse_vec), 0.025), mse_hi = quantile(filter(!isnan, mse_vec), 0.975),
        pears_med = median(filter(!isnan, pears_vec)), pears_lo = quantile(filter(!isnan, pears_vec), 0.025), pears_hi = quantile(filter(!isnan, pears_vec), 0.975),
        dcor_med = median(filter(!isnan, dcor_vec)), dcor_lo = quantile(filter(!isnan, dcor_vec), 0.025), dcor_hi = quantile(filter(!isnan, dcor_vec), 0.975)
    )
end

function calculate_mcmc_diagnostics(chain_results)
    n_s = length(chain_results[1]["σ²"])
    σ²_c = reshape(vcat([r["σ²"] for r in chain_results]...), n_s, 1, N_CHAINS)
    λ_c = reshape(vcat([r["λ"] for r in chain_results]...), n_s, 1, N_CHAINS)
    
    return (
        r_hat_s = rhat(σ²_c)[1], ess_s = ess(σ²_c)[1],
        r_hat_l = rhat(λ_c)[1], ess_l = ess(λ_c)[1]
    )
end

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

println("--- Starting NHANES Analysis: Proximal SGLD [n > p Regime] ---")

y_raw, X_raw, chunks = load_data_and_create_chunks(DATA_FILE, TARGET_COL, SEED_SPLIT, NUM_FOLDS)
p = size(X_raw, 2)
println("Loaded Data: n = $(length(y_raw)), p = $p")

fold_metrics = DataFrame(
    Fold = Int[], Penalty = String[],
    MSE_Med = Float64[], MSE_Low = Float64[], MSE_High = Float64[],
    Cor_Med = Float64[], Cor_Low = Float64[], Cor_High = Float64[],
    dCor_Med = Float64[], dCor_Low = Float64[], dCor_High = Float64[],
    Model_Size_CI = Float64[],  # 95% Credible Interval Selection
    Model_Size_PIP = Float64[], # PIP > 0.5 Selection
    Rhat_Lambda = Float64[], ESS_Lambda = Float64[],
    Sig_Med = Float64[], Sig_Low = Float64[], Sig_High = Float64[],
    Rhat_Sigma = Float64[], ESS_Sigma = Float64[],
    Time_Sec = Float64[]
)

coef_details_all_folds = Dict{String, Matrix{Float64}}()
for p_name in PENALTY_NAMES
    coef_details_all_folds[p_name] = zeros(p, NUM_FOLDS)
end

for fold_k in 1:NUM_FOLDS
    println("\n" * "="^60)
    println(">>> Running CV Fold $fold_k / $NUM_FOLDS <<<")
    
    y_train_std, X_train_std, y_val_std, X_val_std, y_test_std, X_test_std = get_fold_data(y_raw, X_raw, chunks, fold_k, NUM_FOLDS)
    
    n_train = size(X_train_std, 1)
    batch_size_n = max(1, floor(Int, n_train / 3))
    
    # Calculate Lipschitz
    if n_train < p
        L_val = eigmax(Symmetric(X_train_std * X_train_std'))
    else
        L_val = eigmax(Symmetric(X_train_std' * X_train_std))
    end
    eta0_dynamic = 1.0 / L_val
    println("  -> Data configured: n_train=$n_train. batch_size=$batch_size_n, eta0=$(round(eta0_dynamic, digits=6))")
    
    for penalty_name in PENALTY_NAMES
        println("  -> Evaluating Penalty: $penalty_name")
        
        @everywhere model_config = MODELS[$penalty_name]
        
        elapsed = @elapsed begin
            chain_results = pmap(id -> run_prox_real_wrapper(
                id, y_train_std, X_train_std, y_val_std, X_val_std, 
                NUM_ITER, eta0_dynamic, batch_size_n, THINNING, 
                model_config, penalty_name, BURN_IN, fold_k
            ), 1:N_CHAINS)
        end
        
        all_β = hcat([r["β"] for r in chain_results]...)
        all_σ² = vcat([r["σ²"] for r in chain_results]...)
        
        pred = calculate_predictive_metrics(X_test_std, y_test_std, all_β)
        diag = calculate_mcmc_diagnostics(chain_results)
        sig_stats = calculate_stats(all_σ²)
        
        # Safe 95% CI & PIP calculations
        lower = zeros(p); upper = zeros(p); pip_vec = zeros(p)
        for j in 1:p
            clean_trace = filter(!isnan, all_β[j, :])
            if length(clean_trace) > 0
                q = quantile(clean_trace, (0.025, 0.975))
                lower[j], upper[j] = q[1], q[2]
                pip_vec[j] = mean(abs.(clean_trace) .> 1e-12)
            else
                lower[j], upper[j], pip_vec[j] = NaN, NaN, 0.0
            end
        end
        
        is_selected_ci = [(!isnan(l) && (l > 0.0 || u < 0.0)) for (l,u) in zip(lower, upper)]
        model_size_ci = sum(is_selected_ci)
        
        is_selected_pip = pip_vec .> 0.5 
        model_size_pip = sum(is_selected_pip)
        
        coef_details_all_folds[penalty_name][:, fold_k] = vec(mean(all_β, dims=2))
        
        push!(fold_metrics, (
            fold_k, penalty_name,
            pred.mse_med, pred.mse_lo, pred.mse_hi,
            pred.pears_med, pred.pears_lo, pred.pears_hi,
            pred.dcor_med, pred.dcor_lo, pred.dcor_hi,
            Float64(model_size_ci), Float64(model_size_pip), 
            diag.r_hat_l, diag.ess_l,
            sig_stats.med, sig_stats.lo, sig_stats.hi,
            diag.r_hat_s, diag.ess_s,
            elapsed
        ))
        println("     [Done] Time: $(round(elapsed, digits=1))s | CI Size: $model_size_ci | Std-MSE: $(round(pred.mse_med, digits=3))")
        
        # 1. Nullify the massive arrays to remove references
        all_β = nothing
        all_σ² = nothing
        chain_results = nothing
        
        # 2. Force Garbage Collection on the Master process
        GC.gc()
        
        # 3. Force Garbage Collection on ALL Worker processes
        @everywhere GC.gc()
    end
end

println("\n" * "="^60)
println("Calculating 5-Fold Averages per Penalty...")

mean_metrics = combine(groupby(fold_metrics, :Penalty),
    :MSE_Med => mean => :MSE_Mean, :MSE_Med => std => :MSE_Std,
    :Cor_Med => mean => :Cor_Mean, :Cor_Med => std => :Cor_Std,
    :dCor_Med => mean => :dCor_Mean, :dCor_Med => std => :dCor_Std,
    :Model_Size_CI => mean => :Model_Size_CI_Mean, :Model_Size_CI => std => :Model_Size_CI_Std,
    :Model_Size_PIP => mean => :Model_Size_PIP_Mean, :Model_Size_PIP => std => :Model_Size_PIP_Std,
    :Sig_Med => mean => :Sig_Mean, :Sig_Med => std => :Sig_Std,
    :Rhat_Lambda => mean => :Rhat_Lambda_Mean,
    :Time_Sec => mean => :Time_Mean, :Time_Sec => std => :Time_Std
)

println(mean_metrics)

CSV.write("NHANES_metrics_valMH_5Fold_Std.csv", fold_metrics)
CSV.write("NHANES_summary_valMH_5Fold_Std.csv", mean_metrics)

coef_df = DataFrame(Index = 1:p)
for p_name in PENALTY_NAMES
    coef_df[!, Symbol(p_name * "_Mean_CV")] = vec(mean(coef_details_all_folds[p_name], dims=2))
end
CSV.write("NHANES_coefs_valMH_5Fold_Std.csv", coef_df)

println("NHANES Prox-SGLD Results saved successfully.")