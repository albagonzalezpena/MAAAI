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
    metrics::AbstractDict
)
    # 1. Pipeline
    pipe = FlexiblePipeline(scaler, filter_model, reducer_model, model)
    
    # 2. Métricas
    m_names = collect(keys(metrics))
    m_objs  = collect(values(metrics))

    # 3. Evaluar (Importante: fitted_params_per_fold es clave para RFE)
    eval = evaluate(pipe, X, y, resampling=folds, measure=m_objs, verbosity=0)

    # 4. Resultados Numéricos
    results_dict = Dict{String, Vector{Float64}}()
    for (name, idx) in zip(m_names, 1:length(m_objs))
        results_dict[name] = eval.per_fold[idx]
    end

    # 5. Detección Inteligente de Dimensiones
    n_orig = length(MLJ.schema(X).names)
    n_filt_list = Int[]
    n_proj_list = Int[]

    # Iteramos Reportes (Para ANOVA/Pearson) y Fitted Params (Para RFE)
    for (r, fp) in zip(eval.report_per_fold, eval.fitted_params_per_fold)
        
        nf = n_orig
        
        # --- Lógica de Detección del Filtro ---
        if filter_model !== nothing
            # A. Filtros Nativos (ANOVA, Pearson...) -> Usan Reporte
            if hasproperty(r, :filter) && hasproperty(r.filter, :n_final)
                nf = r.filter.n_final
            
            # B. Wrapper RFE -> Usa Fitted Params (:features_left)
            elseif hasproperty(fp, :filter) && hasproperty(fp.filter, :features_left)
                nf = length(fp.filter.features_left)
            
            # C. Wrapper Standard (Fallback) -> Usa Fitted Params (:selected_features)
            elseif hasproperty(fp, :filter) && hasproperty(fp.filter, :selected_features)
                nf = length(fp.filter.selected_features)
            end
        end
        
        push!(n_filt_list, nf)

        # --- Lógica de Detección del Proyector ---
        np = nf
        if reducer_model !== nothing && hasproperty(r, :projector) && hasproperty(r.projector, :outdim)
             np = r.projector.outdim
        end
        push!(n_proj_list, np)
    end

    # Estadísticas
    f_mean, f_std = mean(n_filt_list), std(n_filt_list)
    p_mean, p_std = mean(n_proj_list), std(n_proj_list)

    # Verbosity
    dim_str = "{$n_orig} -> {$(round(f_mean, digits=1))} -> {$(round(p_mean, digits=1))}"
    print("Exp: $tag   Dims: $dim_str")
    for (name, vals) in results_dict
        print("   $name: $(round(mean(vals), digits=3))")
    end
    println()

    # 6. Objeto History
    history = History(
        tag,
        results_dict,
        n_orig,
        (f_mean, f_std),
        (p_mean, p_std)
    )

    return history
end

end # module