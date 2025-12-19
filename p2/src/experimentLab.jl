module ExperimentLab

"""
    Este módulo nos sirve para abstraer la lógica de los experimentos y poder 
    ejecutar experimentos y gestionar resultados de manera sencilla.

    Aquí se implementa el pipeline que seguirán nuestros experimentos, aparte de
    las estructuras de datos pertinentes para gestionar resultados y la lógica de los
    experimentos, recogiendo resultados intermedios, calculando métricas, etc. Con el fin
    de depurar de manera adecuada los experimentos y aumentar la interpretabilidad 
    de nuestros resultados.
"""

using MLJ
using MLJBase
using DataFrames
using CSV
using Dates
using Statistics
using Tables
using LIBSVM   
using CategoricalArrays
import MLJBase: source, machine, node

export run_experiment_crossvalidation, History, run_experiment_holdout

# ==============================================================================
# HELPER PARA FEATURE IMPORTANCE
# ==============================================================================
"""
    Este método nos permitirá obtener la importancia de las features para 
    los modelos en etapas intermedias del pipeline.

    Params:
        - mach: máquina se la que obtener feature importance.
    Returns:
        - datos de feature importance si ha sido posible obtenerlos

"""
function safe_get_importances(mach)
    try
        return feature_importances(mach)
    catch
        println("Error obteniendo feature importance")
        return nothing
    end
end

# ==============================================================================
# LEARNING NETWORK
# ==============================================================================
"""
    En este objeto se implementa el pipeline para ejecutar todos y cada unos
    de los experimentos de la práctica.

    Args:
        - scaler: modelo normalizador de los datos.
        - filter: modelo de filtrado de features.
        - projector: modelo reductor de dimensionalidad por proyección.
        - classifier: clasificador.
"""
mutable struct FlexiblePipeline <: MLJ.ProbabilisticNetworkComposite
    scaler::Union{MLJ.Model, Nothing}
    filter::Union{MLJ.Model, Nothing}
    projector::Union{MLJ.Model, Nothing}
    classifier::MLJ.Model
end

# Constructor
function FlexiblePipeline(; scaler=nothing, filter=nothing, projector=nothing, classifier)
    return FlexiblePipeline(scaler, filter, projector, classifier)
end

"""
    Método prefit que implementa el flujo de datos en la learning network.

    Sirve para indicar cómo aplicar cada uno de los pasos, además de obtener
    reports sobre cada etapa intermedia del pipeline.
"""
function MLJBase.prefit(model::FlexiblePipeline, verbosity, X, y)
    # Nodos fuente 
    Xs = source(X)
    ys = source(y)

    current_X = Xs
    
    # Diccionario para capturar reportes de cada paso
    reports = Dict{Symbol, Any}()

    # Paso 1: scaler (opcional), se obtiene su report
    if model.scaler !== nothing
        mach_scaler = machine(:scaler, current_X)
        current_X = MLJ.transform(mach_scaler, current_X)
        reports[:scaler] = node(report, mach_scaler)
    end

    # Paso 2: filtrado (opcional), se obtiene su report
    if model.filter !== nothing
        # El filtro necesita 'ys' para aprender
        mach_filter = machine(:filter, current_X, ys)
        current_X = MLJ.transform(mach_filter, current_X)
        reports[:filter] = node(report, mach_filter)
    end

    # Paso 3: proyector (opcional), se obtiene su report.
    # Se implementa soporte para modelo supervisado y no supervisado.
    if model.projector !== nothing
        # Detectamos si el modelo es supervisado
        if is_supervised(model.projector)
            mach_proj = machine(:projector, current_X, ys)
        else
            mach_proj = machine(:projector, current_X)
        end
        
        current_X = MLJ.transform(mach_proj, current_X)
        reports[:projector] = node(report, mach_proj)
    end

    # Paso 4: clasificador
    mach_clf = machine(:classifier, current_X, ys)
    yhat = MLJ.predict(mach_clf, current_X)
    reports[:classifier] = node(report, mach_clf)
    reports[:feature_importances] = node(safe_get_importances, mach_clf) # obtener feature_importance

    # Exponemos la predicción y los reportes acumulados
    # Convertimos el dict a NamedTuple para cumplir con la interfaz de MLJ
    report_nt = (; (k => v for (k,v) in reports)...)

    return (predict = yhat, report = report_nt)
end

# ==============================================================================
# HISTORY
# ==============================================================================

"""
    Estructura de datos inspirada en frameworks de deep learning que nos permite
    almacenar y gestionar resultados de experimentos de manera sencilla, diseñado 
    para las necesidades del análisis de resultados.

    Args:
        - name: tag del modelo.
        - metrics: diccionario (nombre, vector) que contiene los resultados de cada métrica.
        - input_dim: features de entrada del dataset.
        - filter_dim: tupla (mean, std) con las features del dataset tras el filtrado.
        - proj_dim: tupla (mean, std) con las features del dataset tras la proyección.
        - feature_importances: contiene las importancias de features usadas por los modelos.
        - confussion_matrix: usado en experimentos holdout, calcula la matriz de confusion calculada.
"""

struct History
    name::String
    metrics::Dict{String, Any} 
    input_dim::Int
    filter_dim::Tuple{Float64, Float64}
    proj_dim::Tuple{Float64, Float64}
    feature_importances::Any
    confussion_matrix::Any
end

# ==============================================================================
# EJECUTOR DE EXPERIMENTOS CROSSVALIDATION
# ==============================================================================

"""
    Esta función nos permite ejecutar y obtener los resultados del entrenamiento y evaluación
    de un modelo entrenado con crossvalidation.

    Params:
        - scaler: normalizador empleado.
        - filter_model: wrapper de filtrado usado.
        - reducer_model: modelo de reducción por proyección empleado.
        - model: clasificador empleado.
        - X: inputs del dataset.
        - y: labels del dataset.
        - folds: resampling para crossvalidation.
        - tag: nombre del modelo/experimento.
        - metrics: diccionario que contiene las métricas y su nombre.

    Returns:
        - History: con los resultados de la ejecución.
"""

function run_experiment_crossvalidation(
    scaler, filter_model, reducer_model, model, X, y, folds; 
    tag="experiment", 
    metrics::AbstractDict
)
    # Crear pipeline
    pipe = FlexiblePipeline(scaler, filter_model, reducer_model, model)
    
    # Obtener métricas
    m_names = collect(keys(metrics))
    m_objs  = collect(values(metrics))

    # Evaluar pipeline
    eval = evaluate(pipe, X, y, resampling=folds, measure=m_objs, verbosity=0)

    # Resultados de métricas, creamos diccionario de resultados
    results_dict = Dict{String, Vector{Float64}}()
    for (name, idx) in zip(m_names, 1:length(m_objs))
        results_dict[name] = eval.per_fold[idx]
    end

    # Detección de dimensionalidad
    n_orig = length(MLJ.schema(X).names)
    n_filt_list = Int[]
    n_proj_list = Int[]

    # Iterar reportes por folds
    for (r, fp) in zip(eval.report_per_fold, eval.fitted_params_per_fold)
        
        nf = n_orig # features originales del dataset
        
        # Detección de dimensiones tras filtrado
        if filter_model !== nothing
            # wrappers implementados que usan report
            if hasproperty(r, :filter) && hasproperty(r.filter, :n_final)
                nf = r.filter.n_final
            
            # RFE: obtenemos las dimensiones a través de features_left
            elseif hasproperty(fp, :filter) && hasproperty(fp.filter, :features_left)
                nf = length(fp.filter.features_left)
            
            # fallback: en caso de que alguno de los métodos anteriores falle
            elseif hasproperty(fp, :filter) && hasproperty(fp.filter, :selected_features)
                nf = length(fp.filter.selected_features)
            end
        end
        
        push!(n_filt_list, nf)

        # Detección de las dimensiones tras proyección
        np = nf
        if reducer_model !== nothing && hasproperty(r, :projector) && hasproperty(r.projector, :outdim)
             np = r.projector.outdim
        end
        push!(n_proj_list, np)
    end

    # Calcular mean y std tras la reducción de dimensionalidad
    f_mean, f_std = mean(n_filt_list), std(n_filt_list)
    p_mean, p_std = mean(n_proj_list), std(n_proj_list)

    # Verbosity
    dim_str = "{$n_orig} -> {$(round(f_mean, digits=1)) ± $(round(f_std, digits=1))} -> {$(round(p_mean, digits=1)) ± $(round(p_std, digits=1))}"
    print("Exp: $tag   Dims: $dim_str")
    for (name, vals) in results_dict
        print("   $name: $(round(mean(vals), digits=3))")
    end
    println()

    # Creación de history
    history = History(
        tag,
        results_dict,
        n_orig,
        (f_mean, f_std),
        (p_mean, p_std),
        nothing,
        nothing
    )

    return history
end

# ==============================================================================
# EJECUTOR DE EXPERIMENTOS HOLDOUT
# ==============================================================================

"""
    Esta función nos permite ejecutar y obtener los resultados del entrenamiento y evaluación
    de un modelo entrenado con holdout.

    Params:
        - scaler: normalizador empleado.
        - filter_model: wrapper de filtrado usado.
        - reducer_model: modelo de reducción por proyección empleado.
        - model: clasificador empleado.
        - X_train: inputs del set de entrenamiento.
        - y_train: labels del set de entrenamiento.
        - X_test: inputs del set de test
        - y_test: labels del set de test.
        - tag: nombre del modelo/experimento.
        - metrics: diccionario que contiene las métricas y su nombre.

    Returns:
        - History: con los resultados de la ejecución.
"""

function run_experiment_holdout(
    scaler, filter_model, reducer_model, model, 
    X_train, y_train, X_test, y_test; 
    tag="Final_Test", 
    metrics::AbstractDict
)

    # Construir Pipeline y Máquina
    pipe = FlexiblePipeline(scaler, filter_model, reducer_model, model)
    mach = machine(pipe, X_train, y_train)

    # Entrenar con todo el conjunto de Train
    MLJ.fit!(mach, verbosity=0)

    # Predecir sobre Test
    y_mode  = MLJ.predict_mode(mach, X_test)

    # Matriz de confusión
    cmat = MLJ.confusion_matrix(y_mode, y_test)
    raw_matrix = cmat.mat
    raw_matrix = raw_matrix' # transponer para que filas=verdaderos, columnas=predichos


    results_dict = Dict{String, Vector{Float64}}()
    
    # Calcular métricas
    for (name, measure_fn) in metrics
        try
            val = measure_fn(y_mode, y_test)
            results_dict[name] = [val]
        catch e
            @warn "Error calculando métrica $name" exception=e
            results_dict[name] = [0.0]
        end
    end


    # Informacion sobre filtrado
    r = report(mach)
    fp = fitted_params(mach)
    n_orig = length(MLJ.schema(X_train).names)
    nf = n_orig

    # Filtrado
    if filter_model !== nothing
        if hasproperty(r, :filter) && hasproperty(r.filter, :n_final)
            nf = r.filter.n_final # Wrappers implementados
        elseif hasproperty(fp, :filter) && hasproperty(fp.filter, :features_left)
            nf = length(fp.filter.features_left) # RFE
        elseif hasproperty(fp, :filter) && hasproperty(fp.filter, :selected_features)
            nf = length(fp.filter.selected_features) # fallback
        end
    end
    
    # Informacion sobre la por proyección
    np = nf
    if reducer_model !== nothing && hasproperty(r, :projector) && hasproperty(r.projector, :outdim)
         np = r.projector.outdim
    end

    feature_importance = nothing

    # Clasificador: extraer feature importance
    if hasproperty(r, :feature_importances) && r.feature_importances !== nothing
        feature_importance = r.feature_importances
    end

    # Verbosity
    dim_str = "{$n_orig} -> {$nf} -> {$np}"
    print("Exp: $tag   Dims: $dim_str")
    for (name, vals) in results_dict
        print("   $name: $(round((vals[1]), digits=4))")
    end
    println()

    # Devolver history
    return History(
        tag,
        results_dict,
        n_orig,
        (Float64(nf), 0.0),
        (Float64(np), 0.0),
        feature_importance,
        raw_matrix
    )
end

end # module