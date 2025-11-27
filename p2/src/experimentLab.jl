module ExperimentLab

using MLJ
using MLJBase
using DataFrames
using CSV
using Dates
using Statistics
using Tables
using LIBSVM   
using CategoricalArrays

# Importamos utilidades para definir la red
import MLJBase: source, machine, node

export run_experiment_csv, History

# ==============================================================================
# LEARNING NETWORK
# ==============================================================================
"""
FlexiblePipeline: Una red de aprendizaje que permite conectar:
Scaler (Opcional) -> Filtro Supervisado (Opcional) -> Reductor (Híbrido) -> Clasificador
Maneja automáticamente si el reductor necesita 'y' (LDA) o no (PCA).
"""
mutable struct FlexiblePipeline <: MLJ.ProbabilisticNetworkComposite
    scaler::Union{MLJ.Model, Nothing}
    filter::Union{MLJ.Model, Nothing}
    projector::Union{MLJ.Model, Nothing}
    classifier::MLJ.Model
end

# Constructor con keywords para facilitar su uso
function FlexiblePipeline(; scaler=nothing, filter=nothing, projector=nothing, classifier)
    return FlexiblePipeline(scaler, filter, projector, classifier)
end

function MLJBase.prefit(model::FlexiblePipeline, verbosity, X, y)
    # 1. Nodos fuente (Entradas)
    Xs = source(X)
    ys = source(y)

    current_X = Xs
    
    # Diccionario para capturar reportes de cada paso
    reports = Dict{Symbol, Any}()

    # --- PASO 1: SCALER ---
    if model.scaler !== nothing
        mach_scaler = machine(:scaler, current_X)
        current_X = MLJ.transform(mach_scaler, current_X)
        # No solemos necesitar reporte del scaler, pero lo guardamos por consistencia
        reports[:scaler] = node(report, mach_scaler)
    end

    # --- PASO 2: FILTRO (SUPERVISADO) ---
    if model.filter !== nothing
        # El filtro necesita 'ys' para aprender
        mach_filter = machine(:filter, current_X, ys)
        current_X = MLJ.transform(mach_filter, current_X)
        reports[:filter] = node(report, mach_filter)
    end

    # --- PASO 3: PROYECTOR (HÍBRIDO) ---
    if model.projector !== nothing
        # Detectamos si el modelo es supervisado (como LDA) o no (como PCA)
        if is_supervised(model.projector)
            mach_proj = machine(:projector, current_X, ys)
        else
            mach_proj = machine(:projector, current_X)
        end
        
        current_X = MLJ.transform(mach_proj, current_X)
        reports[:projector] = node(report, mach_proj)
    end

    # --- PASO 4: CLASIFICADOR ---
    mach_clf = machine(:classifier, current_X, ys)
    yhat = MLJ.predict(mach_clf, current_X)

    # Exponemos la predicción y los reportes acumulados
    # Convertimos el dict a NamedTuple para cumplir con la interfaz de MLJ
    report_nt = (; (k => v for (k,v) in reports)...)

    return (predict = yhat, report = report_nt)
end

# ==============================================================================
# 2. HISTORY
# ==============================================================================

struct History
    name::String
    metrics::Dict{String, Any} 
    input_dim::Int
    filter_dim::Tuple{Float64, Float64}
    proj_dim::Tuple{Float64, Float64}
end

# ==============================================================================
# 3. EJECUTOR DE EXPERIMENTOS CROSSVALIDATION
# ==============================================================================
function run_experiment_crossvalidation(
    scaler, filter_model, reducer_model, model, X, y, folds; 
    tag="experiment", 
    metrics::AbstractDict,
    verbosity::Int=1
)
    # 1. Pipeline
    pipe = FlexiblePipeline(scaler, filter_model, reducer_model, model)
    
    # 2. Preparar Métricas
    m_names = collect(keys(metrics))
    m_objs  = collect(values(metrics))

    # 3. Evaluar (Silencioso)
    eval = evaluate(pipe, X, y, resampling=folds, measure=m_objs, verbosity=0)

    # 4. Construir Diccionario de Resultados
    results_dict = Dict{String, Vector{Float64}}()
    for (name, idx) in zip(m_names, 1:length(m_objs))
        # Guardamos EL VECTOR COMPLETO de resultados por fold
        results_dict[name] = eval.per_fold[idx]
    end

    # 5. Calcular Estadísticas de Dimensiones
    n_orig = size(MLJ.matrix(X), 2) 
    n_filt_list = Int[]
    n_proj_list = Int[]

    for r in eval.report_per_fold
        # Filtro
        nf = n_orig
        if filter_model !== nothing && hasproperty(r, :filter) && hasproperty(r.filter, :n_final)
            nf = r.filter.n_final
        end
        push!(n_filt_list, nf)

        # Proyección
        np = nf
        if reducer_model !== nothing
            np = r.projector.outdim
        end
        push!(n_proj_list, np)
    end

    # 6. Verbosity

    f_mean, f_std = mean(n_filt_list), std(n_filt_list)
    p_mean, p_std = mean(n_proj_list), std(n_proj_list)

    if verbosity == 1

        dim_str = "{$n_orig} -> {$(round(f_mean, digits=1)) ± $(round(f_std, digits=1))} -> {$(round(p_mean, digits=1)) ± $(round(p_std, digits=1))}"
        print("Exp: $tag   Topología: $dim_str")
        
        for (name, vals) in results_dict
            m = round(mean(vals), digits=4)
            s = round(std(vals), digits=4)
            print("     $name: $m ± $s")
        end
        println() 
    end

    # 7. Crear Objeto History
    history = History(
        tag,
        results_dict,
        n_orig,
        (mean(n_filt_list), std(n_filt_list)), # Tupla Filter
        (mean(n_proj_list), std(n_proj_list))  # Tupla Proj
    )

    return history
end

end # module