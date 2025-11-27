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

export run_experiment_csv, FlexiblePipeline

# ==============================================================================
# 1. DEFINICIÓN DE LA LEARNING NETWORK (FLEXIBLE PIPELINE)
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
# 2. LOGGER CSV
# ==============================================================================
function log_to_csv(path, row_dict)
    df_row = DataFrame([row_dict])
    select!(df_row, sort(names(df_row))) 
    if isfile(path)
        CSV.write(path, df_row, append=true)
    else
        CSV.write(path, df_row)
    end
end

# ==============================================================================
# 3. EJECUTOR DE EXPERIMENTOS
# ==============================================================================
function run_experiment_csv(
    scaler, filter_model, reducer_model, model, X, y, folds; 
    tag="experiment", metrics=[accuracy], csv_path="results.csv"
)

    # 1. Pipeline
    pipe = FlexiblePipeline(
        scaler=scaler, filter=filter_model, projector=reducer_model, classifier=model
    )
    
    println("⚙️ Ejecutando: $tag ...")

    # 2. Evaluación con múltiples métricas
    eval = evaluate(
        pipe, X, y,
        resampling = folds,
        measure = metrics,
        verbosity = 0
    )

    # =============================
    # 3. Métricas de rendimiento
    # =============================

    # Cada métrica tendrá:
    # - media
    # - std
    # - valores por fold
    # Todo se añadirá al diccionario row

    metric_stats = Dict{Symbol, Any}()

    for (m_idx, metric) in enumerate(metrics)
        metric_name = first(string(metric), 3)

        mean_val = eval.measurement[m_idx]
        std_val  = std(eval.per_fold[m_idx])
        fold_vals = eval.per_fold[m_idx]

        # Guardamos
        metric_stats[Symbol(metric_name * "_Mean")] = round(mean_val, digits=4)
        metric_stats[Symbol(metric_name * "_Std")]  = round(std_val, digits=4)

        # Valores por fold
        for (i, s) in enumerate(fold_vals)
            metric_stats[Symbol(metric_name * "_Fold_$i")] = round(s, digits=4)
        end
    end

    # =============================
    # 4. INSPECCIÓN PROFUNDA
    # =============================
    
    n_feats_per_fold = Int[]
    n_dims_per_fold  = Int[]
    
    for r in eval.report_per_fold
        # -- A. Features tras Filtro --
        n_f = ncol(X) 
        if filter_model !== nothing && hasproperty(r, :filter)
            if hasproperty(r.filter, :n_final)
                n_f = r.filter.n_final
            end
        end
        push!(n_feats_per_fold, n_f)
        
        # -- B. Dimensiones tras Proyección --
        n_d = n_f
        if reducer_model !== nothing && hasproperty(r, :projector)
            rep_p = r.projector
            n_d = rep_p.outdim
        end
        push!(n_dims_per_fold, n_d)
    end

    feats_mean = mean(n_feats_per_fold)
    feats_std  = std(n_feats_per_fold)
    dims_mean  = mean(n_dims_per_fold)
    dims_std   = std(n_dims_per_fold)

    # =============================
    # 5. Construcción de fila CSV
    # =============================

    row = Dict(
        :Timestamp => string(Dates.now()),
        :Experiment => tag,
        :Scaler => isnothing(scaler) ? "None" : string(typeof(scaler)),
        :Filter => isnothing(filter_model) ? "None" : string(typeof(filter_model)),
        :Projector => isnothing(reducer_model) ? "None" : string(typeof(reducer_model)),
        :Model => string(typeof(model)),

        # Topología
        :Feats_Original => ncol(X),
        :Feats_Mean => round(feats_mean, digits=1),
        :Feats_Std  => round(feats_std, digits=2),
        :Dims_Mean  => round(dims_mean, digits=1),
        :Dims_Std   => round(dims_std, digits=2)
    )

    # Mezclar las métricas dentro del diccionario final
    merge!(row, metric_stats)

    # ----------------------------
    # GUARDAR EN CSV
    # ----------------------------
    log_to_csv(csv_path, row)

    # ----------------------------
    # Feedback visual
    # ----------------------------
    println("   ✅ Completado: $tag")
    return row
end

end # module