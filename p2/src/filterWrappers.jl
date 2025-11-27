module FilteringReduction

using MLJ
using MLJBase
using StatsBase
using HypothesisTests
using GLM            
using DataFrames
using LinearAlgebra
using MLJModelInterface
using CategoricalArrays
using Tables 

import MLJLinearModels
import MLJModelInterface: fit, transform, predict, input_scitype, target_scitype, output_scitype

@load MultinomialClassifier pkg=MLJLinearModels

# ==============================================================================
# 1. PEARSON (Top K)
# ==============================================================================

export PearsonSelector

mutable struct PearsonSelector <: MLJModelInterface.Supervised
    k::Int
end

PearsonSelector(; k::Int=5) = PearsonSelector(k)

function MLJBase.fit(model::PearsonSelector, verbosity::Int, X, y)
    # 1. Limpieza de Tipos
    X_mat = Float64.(Matrix(MLJBase.matrix(X)))
    n_features = size(X_mat, 2)
    
    # Conversión segura del target
    y_numeric = y isa CategoricalVector ? Int.(levelcode.(y)) : Int.(y)
    
    # 2. Correlación contra target (Sin chequeo de redundancia entre features)
    # Usamos valor absoluto porque interesa la magnitud de la relación
    scores = vec(abs.(cor(X_mat, y_numeric)))
    replace!(scores, NaN => 0.0)
    
    # 3. Selección Top K
    # Ordenamos índices basándonos en el score descendente
    sorted_indices = sortperm(scores, rev=true)
    
    # Aseguramos no pedir más features de las que existen
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    # Ordenamos los índices finales para mantener el orden original de las columnas
    final_features = sort(selected_indices)
    
    # --- GESTIÓN DE NOMBRES ---
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    fitresult = (final_features, selected_names)
    
    report = (
        n_original = n_features,
        n_final = length(final_features),
        scores = scores
    )
    
    if verbosity > 0
        println("PearsonSelector: Top $(report.n_final) features seleccionadas.")
    end

    return fitresult, nothing, report
end

function MLJBase.transform(model::PearsonSelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X))
    X_selected = X_mat[:, idxs]
    return MLJBase.table(X_selected, names=names)
end

input_scitype(::Type{<:PearsonSelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:PearsonSelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:PearsonSelector}) = Table(Continuous)


# ==============================================================================
# 2. SPEARMAN (Top K)
# ==============================================================================

export SpearmanSelector

mutable struct SpearmanSelector <: MLJModelInterface.Supervised
    k::Int 
end

SpearmanSelector(; k::Int=5) = SpearmanSelector(k)

function MLJBase.fit(model::SpearmanSelector, verbosity::Int, X, y)
    raw_data = MLJBase.matrix(X)
    X_mat = Matrix{Float64}(undef, size(raw_data))
    X_mat .= raw_data

    n_features = size(X_mat, 2)
    y_temp = (typeof(y) <: CategoricalArray) ? levelcode.(y) : y
    y_vec = Vector{Float64}(undef, length(y_temp))
    y_vec .= y_temp
    
    # Cálculo
    scores = vec(abs.(corspearman(X_mat, y_vec)))
    replace!(scores, NaN => 0.0)
    
    # Selección Top K
    sorted_indices = sortperm(scores, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    final_features = sort(selected_indices)
    
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    fitresult = (final_features, selected_names)
    
    report = (
        n_original = n_features,
        n_final = length(final_features),
        scores = scores
    )
    
    if verbosity > 0
        println("SpearmanSelector: Top $(report.n_final) features seleccionadas.")
    end

    return fitresult, nothing, report
end

function MLJBase.transform(model::SpearmanSelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X)) 
    X_selected = X_mat[:, idxs]
    return MLJBase.table(X_selected, names=names)
end

input_scitype(::Type{<:SpearmanSelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:SpearmanSelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:SpearmanSelector}) = Table(Continuous)


# ==============================================================================
# 3. KENDALL (Top K)
# ==============================================================================

export KendallSelector

mutable struct KendallSelector <: MLJModelInterface.Supervised
    k::Int
end

KendallSelector(; k::Int=5) = KendallSelector(k)

function MLJBase.fit(model::KendallSelector, verbosity::Int, X, y)
    raw_data = MLJBase.matrix(X)
    X_mat = Matrix{Float64}(undef, size(raw_data))
    X_mat .= raw_data
    
    n_features = size(X_mat, 2)
    y_temp = (typeof(y) <: CategoricalArray) ? levelcode.(y) : y
    y_vec = Vector{Float64}(undef, length(y_temp))
    y_vec .= y_temp
    
    scores = vec(abs.(corkendall(X_mat, y_vec)))
    replace!(scores, NaN => 0.0)
    
    # Selección Top K
    sorted_indices = sortperm(scores, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    final_features = sort(selected_indices)
    
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    fitresult = (final_features, selected_names)
    
    report = (
        n_original = n_features,
        n_final = length(final_features),
        scores = scores
    )
    
    if verbosity > 0
        println("KendallSelector: Top $(report.n_final) features seleccionadas.")
    end

    return fitresult, nothing, report
end

function MLJBase.transform(model::KendallSelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X)) 
    X_selected = X_mat[:, idxs]
    return MLJBase.table(X_selected, names=names)
end

input_scitype(::Type{<:KendallSelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:KendallSelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:KendallSelector}) = Table(Continuous)


# ==============================================================================
# 4. ANOVA (Top K - Usando F-Statistic)
# ==============================================================================

export ANOVASelector

mutable struct ANOVASelector <: MLJModelInterface.Supervised
    k::Int 
end

ANOVASelector(; k::Int=5) = ANOVASelector(k)

function MLJBase.fit(model::ANOVASelector, verbosity::Int, X, y)
    Xmat = Float64.(MLJBase.matrix(X))
    y_numeric = y isa CategoricalVector ? Int.(levelcode.(y)) : Int.(y)

    n_features = size(Xmat, 2)
    f_stats = zeros(Float64, n_features) # Guardamos el estadístico F

    for j in 1:n_features
        feature = Xmat[:, j]
        classes = unique(y_numeric)
        groups = [feature[y_numeric .== c] for c in classes]
        groups = filter(g -> length(g) > 0, groups)

        if length(groups) < 2 
            f_stats[j] = 0.0 
            continue 
        end

        try
            test = OneWayANOVATest(groups...)
            # En HypothesisTests, el campo 'F' contiene el estadístico F
            f_stats[j] = test.F 
        catch e
            f_stats[j] = 0.0
        end
    end
    
    replace!(f_stats, NaN => 0.0)

    # Selección Top K (Mayor F-Statistic es mejor)
    sorted_indices = sortperm(f_stats, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]

    final_features = sort(selected_indices)

    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    fitresult = (final_features, selected_names)

    report = (
        n_original = n_features,
        n_final = length(final_features),
        scores = f_stats # Reportamos los F-statistics
    )
    
    if verbosity > 0
        println("ANOVASelector: Top $(report.n_final) features seleccionadas (basado en F-stat).")
    end

    return fitresult, nothing, report
end

function MLJBase.transform(model::ANOVASelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X)) 
    X_selected = X_mat[:, idxs]
    return MLJBase.table(X_selected, names=names)
end

input_scitype(::Type{<:ANOVASelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:ANOVASelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:ANOVASelector}) = Table(Continuous)


# ==============================================================================
# 5. MUTUAL INFORMATION (Top K)
# ==============================================================================
export MutualInfoSelector

mutable struct MutualInfoSelector <: MLJModelInterface.Supervised
    k::Int
    n_bins::Int
end

MutualInfoSelector(; k::Int=5, n_bins::Int=10) = MutualInfoSelector(k, n_bins)

function MLJBase.fit(model::MutualInfoSelector, verbosity::Int, X, y)
    
    X_mat = Float64.(Matrix(MLJBase.matrix(X)))
    n_features = size(X_mat, 2)
    n_samples = size(X_mat, 1)
    
    y_int = try
        Int.(levelcode.(y))
    catch
        Int.(y)
    end
    if minimum(y_int) < 1 y_int .+= (1 - minimum(y_int)) end
    n_classes = maximum(y_int)
    
    mi_scores = zeros(Float64, n_features)
    
    for i in 1:n_features
        feature = view(X_mat, :, i)
        h = StatsBase.fit(Histogram, feature, nbins=model.n_bins)
        edges = h.edges[1]
        n_actual_bins = length(edges) - 1
        
        counts = zeros(Int, n_actual_bins, n_classes)
        
        for k in 1:n_samples
            val = feature[k]
            bin_idx = searchsortedlast(edges, val)
            if bin_idx > n_actual_bins bin_idx = n_actual_bins end
            if bin_idx < 1 bin_idx = 1 end
            
            class_idx = y_int[k]
            if class_idx <= n_classes
                counts[bin_idx, class_idx] += 1
            end
        end
        
        count_x = sum(counts, dims=2)
        count_y = sum(counts, dims=1)
        mi = 0.0
        for bx in 1:n_actual_bins
            for by in 1:n_classes
                c_xy = counts[bx, by]
                if c_xy > 0
                    term = log2( (c_xy * n_samples) / (count_x[bx] * count_y[by]) )
                    mi += (c_xy / n_samples) * term
                end
            end
        end
        mi_scores[i] = mi
    end
    
    replace!(mi_scores, NaN => 0.0)
    
    # Selección Top K
    sorted_indices = sortperm(mi_scores, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    final_features = sort(selected_indices)
    
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    fitresult = (final_features, selected_names)
    
    report = (
        n_original = n_features,
        n_final = length(final_features),
        scores = mi_scores
    )
    
    if verbosity > 0
        println("MutualInfoSelector: Top $(report.n_final) features seleccionadas.")
    end

    return fitresult, nothing, report
end

function MLJBase.transform(model::MutualInfoSelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X)) 
    X_selected = X_mat[:, idxs]
    return MLJBase.table(X_selected, names=names)
end

input_scitype(::Type{<:MutualInfoSelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:MutualInfoSelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:MutualInfoSelector}) = Table(Continuous)



# ==============================================================================
# Logistic Regression Wrapper para MLJ
# ==============================================================================
mutable struct LogisticRFE <: MLJModelInterface.Probabilistic  # ⬅️ CAMBIO A Deterministic
    lambda::Float64
    penalty::Symbol
    fit_intercept::Bool
end

LogisticRFE(; lambda::Float64=1.0, penalty::Symbol=:l2, fit_intercept::Bool=true) =
    LogisticRFE(lambda, penalty, fit_intercept)

MLJModelInterface.input_scitype(::Type{<:LogisticRFE}) = Table(Continuous)
MLJModelInterface.target_scitype(::Type{<:LogisticRFE}) = AbstractVector{<:Finite}
MLJModelInterface.reports_feature_importances(::Type{<:LogisticRFE}) = true

# ⬅️ Devolver el MISMO TIPO que el modelo base
function MLJModelInterface.fit(model::LogisticRFE, verbosity::Int, X, y)

    # NO convertir y aquí - dejar que MultinomialClassifier lo haga
    real_model = MLJLinearModels.MultinomialClassifier(
        lambda = model.lambda,
        penalty = model.penalty,
        fit_intercept = model.fit_intercept
    )
    
    # Crear máquina interna
    mach = machine(real_model, X, y)
    fit!(mach, verbosity=verbosity)
    
    # Guardar features
    features = collect(Tables.columnnames(X))
    
    # ⬅️ CLAVE: Devolver estructura correcta para MLJ
    # fitresult debe ser algo que predict pueda usar
    fitresult = (mach=mach, features=features)
    cache = nothing
    report_content = report(mach)
    
    return fitresult, cache, report_content
end

# ⬅️ Predict debe devolver clases (Deterministic)
function MLJModelInterface.predict(model::LogisticRFE, fitresult, Xnew)
    mach = fitresult.mach
    return predict(mach, Xnew)  # Clases directas
end


function MLJModelInterface.feature_importances(model::LogisticRFE, fitresult, report)
    mach = fitresult.mach
    features = fitresult.features
    
    try
        fp = fitted_params(mach)
        coefs = fp.coefs
        n_features = length(features)
        
        # Manejar matriz (multiclase) o vector (binario)
        if coefs isa AbstractMatrix
            # Promedio absoluto sobre clases
            # Si tiene intercept, es (n_classes, n_features+1)
            if model.fit_intercept && size(coefs, 2) == n_features + 1
                coefs = coefs[:, 2:end]  # Quitar intercept
            end
            importances = vec(mean(abs.(coefs), dims=1))
        else
            # Vector: binario
            if model.fit_intercept && length(coefs) == n_features + 1
                coefs = coefs[2:end]
            end
            importances = abs.(coefs)
        end
        
        # Limpiar NaN/Inf
        importances = replace(importances, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
        
        return [features[i] => importances[i] for i in 1:length(importances)]
    catch e
        @warn "Error en feature_importances: $e"
        return [feat => 0.0 for feat in features]
    end
end


function RFELogistic(; k::Int=5, step::Union{Int,Float64}=1, lambda::Float64=1.0)
    base_model = LogisticRFE(lambda=lambda, penalty=:l2)
    return RecursiveFeatureElimination(base_model, n_features=k, step=step)
end

end # module