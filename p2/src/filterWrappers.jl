module FilteringReduction

"""
Módulo que implementa varios selectores de características basados en filtrado.

Incluye selectores basados en:
    - Correlación de Pearson
    - Correlación de Spearman
    - Correlación de Kendall
    - ANOVA (F-Statistic)
    - Mutual Information

A mayores, se incluye un wrapper de clasificación multinomial adaptado para poder ser usando
con RecursiveFeatureElimination de MLJ.

"""

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
using FeatureSelection

import MLJLinearModels
import MLJModelInterface: fit, transform, predict, input_scitype, target_scitype, output_scitype

@load MultinomialClassifier pkg=MLJLinearModels

# ==============================================================================
# Pearson
# ==============================================================================

"""
    Wrapper de filtrado basado en correlación de Pearson, siguiendo el estándard de MLJ.
    Selecciona las k características con mayor correlación con el target
"""

export PearsonSelector

mutable struct PearsonSelector <: MLJModelInterface.Supervised
    k::Int
end

PearsonSelector(; k::Int=5) = PearsonSelector(k)

"""
    Función fit que calcula las features seleccionadas a partir de un
    conjunto de datos.
"""
function MLJBase.fit(model::PearsonSelector, verbosity::Int, X, y)

    # Limpieza de tipos
    X_mat = Float64.(Matrix(MLJBase.matrix(X)))
    n_features = size(X_mat, 2)
    
    # Conversión del target a numérico
    y_numeric = y isa CategoricalVector ? Int.(levelcode.(y)) : Int.(y)
    
    # Correlación contra target 
    # Usamos valor absoluto porque interesa la magnitud de la relación
    scores = vec(abs.(cor(X_mat, y_numeric)))
    replace!(scores, NaN => 0.0)
    
    # Selección Top K
    # Ordenamos índices basándonos en el score descendente
    sorted_indices = sortperm(scores, rev=true)
    
    # Seleccionar los primeros k
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    # Ordenamos los índices finales para mantener el orden original de las columnas
    final_features = sort(selected_indices)
    
    # Gestión de variables seleccionadas
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    # Generar fitresult
    fitresult = (final_features, selected_names)
    
    # Generar report 
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

"""
    Función Transform que genera el dataset filtrado.
"""
function MLJBase.transform(model::PearsonSelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X))
    X_selected = X_mat[:, idxs] # Seleccionar el dataset final
    return MLJBase.table(X_selected, names=names)
end

# Acalaración de scitypes para coherencia de tipos
input_scitype(::Type{<:PearsonSelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:PearsonSelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:PearsonSelector}) = Table(Continuous)


# ==============================================================================
# SPEARMAN 
# ==============================================================================

export SpearmanSelector

"""
    Wrapper de filtrado empleando la correlación de Spearman.
    Devuelve un dataset con las k feature con mayor correlación con el target.
"""

mutable struct SpearmanSelector <: MLJModelInterface.Supervised
    k::Int 
end

SpearmanSelector(; k::Int=5) = SpearmanSelector(k)

"""
    Función fit que calcula las features seleccionadas a partir de un
    conjunto de datos.
"""

function MLJBase.fit(model::SpearmanSelector, verbosity::Int, X, y)

    # Conversión de datos
    raw_data = MLJBase.matrix(X)
    X_mat = Matrix{Float64}(undef, size(raw_data))
    X_mat .= raw_data

    n_features = size(X_mat, 2)
    y_temp = (typeof(y) <: CategoricalArray) ? levelcode.(y) : y
    y_vec = Vector{Float64}(undef, length(y_temp))
    y_vec .= y_temp
    
    # Cálculo de la correlación de Spearman
    scores = vec(abs.(corspearman(X_mat, y_vec)))
    replace!(scores, NaN => 0.0)
    
    # Selección Top K features
    sorted_indices = sortperm(scores, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    # Ordenamos los índices finales para mantener el orden original de las columnas
    final_features = sort(selected_indices)
    
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    # Generar fitresult
    fitresult = (final_features, selected_names)
    
    # Generar report
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

"""
    Función Transform que genera el dataset filtrado.
"""

function MLJBase.transform(model::SpearmanSelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X)) 
    X_selected = X_mat[:, idxs]
    return MLJBase.table(X_selected, names=names)
end

# Acalaración de scitypes para coherencia de tipos
input_scitype(::Type{<:SpearmanSelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:SpearmanSelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:SpearmanSelector}) = Table(Continuous)


# ==============================================================================
# KENDALL 
# ==============================================================================

export KendallSelector

"""
    Wrapper de filtrado empleando la tau de Kendall.
    Devuelve un dataset con las k feature con correlación con el target.
"""

mutable struct KendallSelector <: MLJModelInterface.Supervised
    k::Int
end

KendallSelector(; k::Int=5) = KendallSelector(k)

"""
    Función fit que calcula las features seleccionadas a partir de un
    conjunto de datos.
"""

function MLJBase.fit(model::KendallSelector, verbosity::Int, X, y)

    # Adaptación de tipos
    raw_data = MLJBase.matrix(X)
    X_mat = Matrix{Float64}(undef, size(raw_data))
    X_mat .= raw_data
    
    n_features = size(X_mat, 2)
    y_temp = (typeof(y) <: CategoricalArray) ? levelcode.(y) : y
    y_vec = Vector{Float64}(undef, length(y_temp))
    y_vec .= y_temp
    
    # Cálculo de correlación de kendall
    scores = vec(abs.(corkendall(X_mat, y_vec)))
    replace!(scores, NaN => 0.0)
    
    # Selección Top K
    sorted_indices = sortperm(scores, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    # Mantener orden de entrada para las features seleccionadas
    final_features = sort(selected_indices)
    
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    # Generar fitresult
    fitresult = (final_features, selected_names)
    
    # Generar report
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

"""
    Función Transform que genera el dataset filtrado.
"""

function MLJBase.transform(model::KendallSelector, fitresult, X)
    idxs, names = fitresult
    X_mat = Float64.(MLJBase.matrix(X)) 
    X_selected = X_mat[:, idxs]
    return MLJBase.table(X_selected, names=names)
end

# Definir scitypes para coherencia de tipos.
input_scitype(::Type{<:KendallSelector}) = Table(Continuous, Missing)
target_scitype(::Type{<:KendallSelector}) = AbstractVector{<:Finite}
output_scitype(::Type{<:KendallSelector}) = Table(Continuous)


# ==============================================================================
# ANOVA 
# ==============================================================================

export ANOVASelector

"""
    Wrapper de filtrado empleando el test de ANOVA.
    Devuelve un dataset con las k features con mayor explicabilidad 
    de la clase objetivo.
"""

mutable struct ANOVASelector <: MLJModelInterface.Supervised
    k::Int 
end

ANOVASelector(; k::Int=5) = ANOVASelector(k)

"""
    Función fit que calcula las features seleccionadas a partir de un
    conjunto de datos.
"""

function MLJBase.fit(model::ANOVASelector, verbosity::Int, X, y)

    # Adaptación de tipos
    Xmat = Float64.(MLJBase.matrix(X))
    y_numeric = y isa CategoricalVector ? Int.(levelcode.(y)) : Int.(y)

    n_features = size(Xmat, 2)
    f_stats = zeros(Float64, n_features) # Vector para guardar los valores de f

    # Calcular f-statistic para cada feature
    for j in 1:n_features
        feature = Xmat[:, j]
        classes = unique(y_numeric)
        groups = [feature[y_numeric .== c] for c in classes] # obtener grupos de clases
        groups = filter(g -> length(g) > 0, groups)

        # Asegurar que hay más de dos grupos
        if length(groups) < 2 
            f_stats[j] = 0.0 
            continue 
        end

        try
            # Calcular f
            test = OneWayANOVATest(groups...)
            f_stats[j] = test.F 
        catch e # En caso de error, asignamos 0 a f.
            f_stats[j] = 0.0
        end
    end
    
    # Reemplazar NaN por seguridad
    replace!(f_stats, NaN => 0.0)

    # Selección Top K 
    sorted_indices = sortperm(f_stats, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]

    # Mantener orden en features seleccionadas
    final_features = sort(selected_indices)

    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    # Generar fitresult
    fitresult = (final_features, selected_names)

    # Generar report
    report = (
        n_original = n_features,
        n_final = length(final_features),
        scores = f_stats # Reportamos los F-statistics
    )
    
    if verbosity > 0
        println("ANOVASelector: Top $(report.n_final) features seleccionadas.")
    end

    return fitresult, nothing, report
end

"""
    Función Transform que genera el dataset filtrado.
"""
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
#  MUTUAL INFORMATION
# ==============================================================================
export MutualInfoSelector

"""
    Wrapper de filtrado empleando Mutual Information.
    Devuelve las top k features que mejor explican el target de acurdo al Cálculo
    de información mutua.
"""

mutable struct MutualInfoSelector <: MLJModelInterface.Supervised
    k::Int
    n_bins::Int
end

MutualInfoSelector(; k::Int=5, n_bins::Int=10) = MutualInfoSelector(k, n_bins)

"""
    Función fit que calcula las features seleccionadas a partir de un
    conjunto de datos.
"""

function MLJBase.fit(model::MutualInfoSelector, verbosity::Int, X, y)
    
    # Preparar datos
    X_mat = Float64.(Matrix(MLJBase.matrix(X)))
    n_features = size(X_mat, 2)
    n_samples = size(X_mat, 1)
    
    # Convertir y a índices numéricos (1-indexed)
    y_int = try
        Int.(levelcode.(y))
    catch
        Int.(y)
    end
    
    # Asegurar índices desde 1
    if minimum(y_int) < 1
        y_int .+= (1 - minimum(y_int))
    end
    
    n_classes = maximum(y_int)
    mi_scores = zeros(Float64, n_features)
    
    # Calcular MI para cada feature
    for i in 1:n_features
        feature = @view X_mat[:, i]
        
        # Discretizar feature en bins
        bins = StatsBase.fit(Histogram, feature, nbins=model.n_bins).edges[1]
        discretized = searchsortedlast.(Ref(bins), feature)
        
        # Corregir límites
        n_bins = length(bins) - 1
        clamp!(discretized, 1, n_bins)
        
        # Tabla de contingencia: contar co-ocurrencias
        counts = zeros(Int, n_bins, n_classes)
        for j in 1:n_samples
            counts[discretized[j], y_int[j]] += 1
        end
        
        # Distribuciones marginales
        p_x = vec(sum(counts, dims=2)) ./ n_samples  # P(X)
        p_y = vec(sum(counts, dims=1)) ./ n_samples  # P(Y)
        p_xy = counts ./ n_samples                   # P(X,Y)
        
        # Calcular mutual information
        mi = 0.0
        for bx in 1:n_bins, by in 1:n_classes
            if p_xy[bx, by] > 0
                mi += p_xy[bx, by] * log2(p_xy[bx, by] / (p_x[bx] * p_y[by]))
            end
        end
        
        mi_scores[i] = mi
    end
    

    replace!(mi_scores, NaN => 0.0)
    
    # Selección Top K
    sorted_indices = sortperm(mi_scores, rev=true)
    n_keep = min(model.k, n_features)
    selected_indices = sorted_indices[1:n_keep]
    
    # Mantener features ordenadas
    final_features = sort(selected_indices)
    
    all_names = collect(Tables.columnnames(X))
    selected_names = all_names[final_features]
    
    # Generar features ordenadas
    fitresult = (final_features, selected_names)
    
    # Generar report
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

"""
    Función Transform que genera el dataset filtrado.
"""

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

"""
    Se implementa un wrapper de MultinomialClassifier para incorporar 
    feature importance y poder emplear RFE.
"""

mutable struct LogisticRFE <: MLJModelInterface.Probabilistic  
    lambda::Float64
    penalty::Symbol
    fit_intercept::Bool
end

LogisticRFE(; lambda::Float64=1.0, penalty::Symbol=:l2, fit_intercept::Bool=true) =
    LogisticRFE(lambda, penalty, fit_intercept)

MLJModelInterface.input_scitype(::Type{<:LogisticRFE}) = Table(Continuous)
MLJModelInterface.target_scitype(::Type{<:LogisticRFE}) = AbstractVector{<:Finite}
MLJModelInterface.reports_feature_importances(::Type{<:LogisticRFE}) = true

"""
    Función fit para el ajuste del modelo
"""
function MLJModelInterface.fit(model::LogisticRFE, verbosity::Int, X, y)

    # NO convertir y aquí - dejar que MultinomialClassifier lo haga
    multinomial_model = MLJLinearModels.MultinomialClassifier(
        lambda = model.lambda,
        penalty = model.penalty,
        fit_intercept = model.fit_intercept
    )
    
    # Crear máquina interna
    mach = machine(multinomial_model, X, y)
    fit!(mach, verbosity=verbosity)
    
    # Guardar features
    features = collect(Tables.columnnames(X))
    
    # Crear fitresult y report
    fitresult = (mach=mach, features=features)
    cache = nothing
    report_content = report(mach)
    
    return fitresult, cache, report_content
end

"""
    Función predict para que MLJ pueda hacer predicciones, s
    implemente se devuelve la predicción del modelo.
"""
function MLJModelInterface.predict(model::LogisticRFE, fitresult, Xnew)
    mach = fitresult.mach
    return predict(mach, Xnew)  # Clases directas
end

"""
    Este método implementa el cálculo de la importancia de cada feature
    que no viene implementada de manera nativa en el modelo para que pueda
    ser usada por RFE.

    La importancia se obtiene del modelo subyacente.
"""
function MLJModelInterface.feature_importances(model::LogisticRFE, fitresult, report)
    mach = fitresult.mach
    features = fitresult.features
    n_features = length(features)
    
    try
        fp = fitted_params(mach)
        
        # Asegurar que los coeficientes existen
        if !haskey(fp, :coefs)
            @warn "No se encontraron coeficientes"
            return [feat => 0.0 for feat in features]
        end
        
        # Obtener los coesficientes del modelo que nos servirán para determinar importancia
        coefs_pairs = fp.coefs
        
        # Calcular importancia para cada feature
        # Se calculará como el promedio del coeficiente para cada clase
        importances = Float64[]
        
        for pair in coefs_pairs
            feature_name = pair.first
            coefs_across_classes = pair.second  # Vector de coeficientes de esta feature en todas las clases
            
            # Usaremos el valor absoluto para garantizar funcionamiento correcto
            importance = mean(abs.(coefs_across_classes))
            push!(importances, importance)
        end
        
        # Devolver como Vector de Pairs importancia-feature
        return [features[i] => importances[i] for i in 1:n_features]
        
    catch e
        @warn "Error calculando importances: $e"
    end
end

"""
    Implementación de un método factory de RFE para abstraer en funcionamiento
    del filtrado y facilitar la implementación clara en el notebook.
"""
function RFELogistic(; k::Int=5, step::Union{Int,Float64}=1, lambda::Float64=1.0)
    base_model = LogisticRFE(lambda=lambda, penalty=:l2)
    return RecursiveFeatureElimination(base_model, n_features=k, step=step)
end

end # module