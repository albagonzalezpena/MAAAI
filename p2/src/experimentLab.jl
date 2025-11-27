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
function run_experiment_crossvalidation(
    scaler, 
    filter_model, 
    reducer_model, 
    model, 
    X, y, folds; 
    tag="experiment", 
    metrics::AbstractDict, # <--- TIPO CORREGIDO
    csv_path="results.csv"
)

    # 1. Pipeline
    pipe = FlexiblePipeline(
        scaler=scaler, filter=filter_model, projector=reducer_model, classifier=model
    )

    # Extraemos medidas
    measure_objs = collect(values(metrics))
    measure_names = collect(keys(metrics))

    # 2. Evaluación
    eval = evaluate(
        pipe, X, y,
        resampling = folds,
        measure = measure_objs,
        verbosity = 0
    )

    # 3. Construcción de Resultados
    row = Dict{Symbol, Any}()
    
    # --- A. Metadatos ---
    row[:Timestamp] = string(Dates.now())
    row[:Experiment] = tag
    row[:Model_Type] = string(typeof(model))
    row[:Scaler] = isnothing(scaler) ? "None" : string(typeof(scaler))
    row[:Filter] = isnothing(filter_model) ? "None" : string(typeof(filter_model))
    row[:Projector] = isnothing(reducer_model) ? "None" : string(typeof(reducer_model))

    # --- B. Dimensiones (Cálculo) ---
    n_feats_per_fold = Int[]
    n_dims_per_fold  = Int[]
    
    for r in eval.report_per_fold
        n_f = ncol(X) 
        if filter_model !== nothing
            n_f = r.filter.n_final
        end
        push!(n_feats_per_fold, n_f)
        
        n_d = n_f
        if reducer_model !== nothing
            n_d = r.projector.outdim
        end
        push!(n_dims_per_fold, n_d)
    end

    f_mean, f_std = mean(n_feats_per_fold), std(n_feats_per_fold)
    d_mean, d_std = mean(n_dims_per_fold), std(n_dims_per_fold)
    
    # Guardamos Dimensiones
    row[:Feats_Original] = ncol(X)
    row[:Feats_PostFilter] = f_mean
    row[:Feats_PostProj] = d_mean

    # --- C. Métricas ---
    for (name, m_idx) in zip(measure_names, 1:length(measure_objs))
        mean_val = eval.measurement[m_idx]
        std_val  = std(eval.per_fold[m_idx])
        fold_vals = eval.per_fold[m_idx]

        row[Symbol(name * "_Mean")] = round(mean_val, digits=4)
        row[Symbol(name * "_Std")]  = round(std_val, digits=4)
        
        for (i, v) in enumerate(fold_vals)
            row[Symbol(name * "_Fold_$i")] = v
        end
    end

    # 4. Verbosity Personalizado
    dim_str = "{$(ncol(X))} -> {$(round(f_mean, digits=1)) ± $(round(f_std, digits=1))} -> {$(round(d_mean, digits=1)) ± $(round(d_std, digits=1))}"
    
    println("   Exp: $tag")
    println("   Topología: $dim_str")
    for name in measure_names
        m_mean = row[Symbol(name * "_Mean")]
        m_std = row[Symbol(name * "_Std")]
        println("     $name: $m_mean ± $m_std")
    end
    println("-"^60)

    # 5. Guardar CSV (CON ORDEN ESPECÍFICO)
    df_row = DataFrame([row])
    
    # Definimos el orden de las columnas
    # 1. Metadatos
    meta_cols = [:Timestamp, :Experiment, :Model_Type, :Scaler, :Filter, :Projector]
    
    # 3. Dimensiones
    dim_cols = [:Feats_Original, :Feats_PostFilter, :Feats_PostProj]
    
    # 2. Métricas (Todo lo que no sea meta ni dim)
    all_cols = Symbol.(names(df_row))
    metric_cols = setdiff(all_cols, [meta_cols; dim_cols])
    sort!(metric_cols) # Ordenamos alfabéticamente para que queden juntas las de mismo tipo
    
    # ORDEN FINAL: METADATOS -> MÉTRICAS -> DIMENSIONES
    final_order = [metric_cols; dim_cols; meta_cols]
    
    # Filtramos para asegurarnos de que solo pedimos columnas que existen
    final_order = intersect(final_order, all_cols)
    
    # Reordenamos el DataFrame
    select!(df_row, final_order)

    # Escribimos
    if isfile(csv_path)
        CSV.write(csv_path, df_row, append=true)
    else
        CSV.write(csv_path, df_row)
    end
end

end # module