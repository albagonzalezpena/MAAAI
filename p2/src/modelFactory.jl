module ModelFactory

using MLJ
using MLJFlux      
using Flux         
using LIBSVM       
using NearestNeighborModels 
using Random
using Optimisers
using MLJEnsembles
using EvoTrees
using MLJDecisionTreeInterface
using MLJXGBoostInterface
using LightGBM
using CatBoost
using MLJBase
using MLJModelInterface
using CategoricalArrays


export get_knn_model, 
       get_svm_model, 
       get_mlp_model, 
       get_bagging_knn_model, 
       get_evotree_model, 
       get_adaboost_model, 
       get_rf_model, 
       get_xgboost_model, 
       get_lightgbm_model, 
       get_catboost_model,
       get_voting_classifier,
       get_stacking_model,
       get_individual_wise_CV,
       IndividualWiseCV


const MLJ_SVC = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
const MLJ_EvoTree = @load EvoTreeClassifier pkg=EvoTrees verbosity=0
const AdaBoostStump = @load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
const MLJ_RF = @load RandomForestClassifier pkg=DecisionTree verbosity=0
const MLJ_XGBoost   = @load XGBoostClassifier pkg=XGBoost verbosity=0
const MLJ_LGBM = LightGBM.MLJInterface.LGBMClassifier
const MLJ_CatBoost = @load CatBoostClassifier pkg=CatBoost verbosity=0


# ==============================================================================
# BUILDER DINÁMICO PARA MLP 
# ==============================================================================
"""
DynamicBuilder: Estructura auxiliar para construir redes neuronales
con topología variable basada en un vector de enteros.
"""
mutable struct DynamicBuilder <: MLJFlux.Builder
    hidden_layers::Vector{Int}
end

function MLJFlux.build(b::DynamicBuilder, rng, n_in, n_out)
    layers = []
    input_dim = n_in
    
    # Crear capas ocultas
    for h in b.hidden_layers
        push!(layers, Dense(input_dim, h, relu))
        input_dim = h
    end
    
    # Capa de salida
    push!(layers, Dense(input_dim, n_out))
    
    return Chain(layers...)
end

# ==============================================================================
# FACTORY: KNN
# ==============================================================================
"""
Devuelve un modelo KNN configurado.
Parámetros:
  - k: Número de vecinos (default: 5)
"""
function get_knn_model(k::Int)
    return KNNClassifier(K=k)
end

# ==============================================================================
# FACTORY: SVM
# ==============================================================================
"""
Devuelve un modelo SVM (Support Vector Machine) configurado.
Parámetros:
  - cost: Penalización por error (C) (default: 1.0)
  - kernel: Tipo de kernel (default: Radial)
"""
function get_svm_model(cost::Float64)
    return MLJ_SVC(cost=cost,  kernel=LIBSVM.Kernel.Linear)
end

# ==============================================================================
# FACTORY: MLP (RED NEURONAL)
# ==============================================================================
"""
Devuelve un modelo de Red Neuronal (MLP) configurado.
Parámetros:
  - hidden_layers: Vector con el tamaño de las capas ocultas (ej: [100, 50])
  - epochs: Número de épocas de entrenamiento
  - learning_rate: Tasa de aprendizaje para el optimizador ADAM
"""
function get_mlp_model(hidden_layers::Vector{Int}; epochs::Int=50, learning_rate::Float64=0.001)
    
    builder = DynamicBuilder(hidden_layers)
    
    return NeuralNetworkClassifier(
        builder = builder,
        epochs = epochs,
        batch_size = 16,
        optimiser = Optimisers.Adam(learning_rate)
    )
end

# ==============================================================================
# FACTORY: BAGGING (KNN BASE)
# ==============================================================================
"""
Devuelve un modelo de Bagging usando KNN como clasificador base.
Parámetros:
  - k: Número de vecinos para el KNN base (ej: 5).
  - n_estimators: Número de modelos en el ensemble (ej: 10, 50).
  - fraction: Porcentaje de datos a usar en cada bolsa (default: 0.8).
"""
function get_bagging_knn_model(k::Int, n_estimators::Int; fraction::Float64=0.8)
    
    # Definir modelo base
    base_model = KNNClassifier(K=k)
    
    # Crear ensemble
    return EnsembleModel(
        model = base_model,
        n = n_estimators,
        bagging_fraction = fraction
    )
end

# ==============================================================================
# FACTORY: ADABOOST (Decision Stumps)
# ==============================================================================
"""
Devuelve un modelo AdaBoost usando 'Decision Stumps' (árboles de profundidad 1).
Este es el algoritmo clásico de AdaBoost.
Parámetros:
  - n_estimators: Número de iteraciones (estimadores).
"""
function get_adaboost_model(n_estimators::Int)
    return AdaBoostStump(
        n_iter = n_estimators  # Mapeamos el argumento al parámetro interno
    )
end


# ==============================================================================
# FACTORY: EvoTree
# ==============================================================================

function get_evotree_model(n_estimators::Int; learning_rate::Float64=0.2)

    return MLJ_EvoTree(
        nrounds = n_estimators,
        eta = learning_rate,
        max_depth = 5,  
        nbins = 32      # Discretización para acelerar (estándar en boosting)
    )
end

# ==============================================================================
# FACTORY: Random Forest
# ==============================================================================


function get_rf_model(n_trees::Int, max_depth::Int=10)
    return MLJ_RF(
        n_trees = n_trees,
        max_depth = max_depth,         
        min_samples_split = 2,
        sampling_fraction = 1.0,       # Bagging de filas (80% datos por árbol)
        n_subfeatures = -1             # -1 significa sqrt(n_features) (Estándar en RF)
    )
end

# ==============================================================================
# FACTORY: XGBoost
# ==============================================================================


function get_xgboost_model(n_rounds::Int)

    return MLJ_XGBoost(
        num_round = n_rounds,
        nthread = 1 
    )
end

# ==============================================================================
# FACTORY: LightGBM
# ==============================================================================


function get_lightgbm_model(n_iters::Int)

    return MLJ_LGBM(
        num_iterations = n_iters,
        objective = "multiclass", # O "binary" si fuera el caso
        metric = ["multi_logloss"],        
        verbosity = -1
    )
end

# ==============================================================================
# FACTORY: CatBoost
# ==============================================================================

function get_catboost_model()
    return MLJ_CatBoost(
        thread_count = 1,   
        devices = nothing,        
        task_type = "CPU",          # Aseguramos que no busque GPUs raras
        allow_writing_files = false)
end



# ===================================================
# Feature Subspace Hard Voting Classifier
# ===================================================

"""
    PartitionedVotingClassifier <: Probabilistic

    Un clasificador ensemble que divide las características (columnas) en N particiones disjuntas.
    Entrena un modelo base en cada subconjunto de características y combina las predicciones
    mediante Votación Mayoritaria (Hard Voting).

    # Campos
    - `model::Probabilistic`: Modelo base (ej. SVM, KNN).
    - `n_partitions::Int`: En cuántos subconjuntos dividir las características.
    - `rng::Int`: Semilla para mezclar las características antes de dividir.
"""

mutable struct PartitionedVotingClassifier <: Probabilistic
    model::Probabilistic
    n_partitions::Int
    rng::Int
end

function PartitionedVotingClassifier(; model=nothing, n_partitions=3, rng=104)
    return PartitionedVotingClassifier(model, n_partitions, rng)
end

"""
    MLJModelInterface.fit(model, verbosity, X, y)

    1. Baraja las características (features).
    2. Las divide en `n_partitions` grupos.
    3. Entrena una copia del modelo base para cada grupo de características.
"""
function MLJModelInterface.fit(m::PartitionedVotingClassifier, verbosity::Int, X, y)
    
    # 1. Obtener nombres de las características
    schema_X = MLJBase.schema(X)
    all_features = collect(schema_X.names)
    n_features = length(all_features)
    
    # 2. Barajar y particionar características (Feature Map)
    rng = MersenneTwister(m.rng)
    shuffled_feats = shuffle(rng, all_features)
    
    # Calcular tamaño de cada partición
    chunk_size = cld(n_features, m.n_partitions) # Ceiling division
    
    machines_list = [] # Guardar los modelos entrenados
    feature_groups = [] # Guardar features usadas por cada modelo
    
    start_idx = 1
    for i in 1:m.n_partitions
        end_idx = min(start_idx + chunk_size - 1, n_features)
        if start_idx > n_features break end
        
        # Seleccionar las columnas para este modelo
        current_feats = shuffled_feats[start_idx:end_idx]
        push!(feature_groups, current_feats)
        
        # Subespacio de características para entrenar el modelo 
        X_sub = MLJBase.selectcols(X, current_feats)
        
        # Entrenar modelo base
        mach = machine(deepcopy(m.model), X_sub, y)
        MLJBase.fit!(mach, verbosity=0)
        push!(machines_list, mach)
        
        # Avanzar el índice para el siguiente grupo
        start_idx = end_idx + 1
    end

    # Guardar estado actual del entrenamiento
    fitresults = (
        machines = machines_list,
        feature_groups = feature_groups, 
        class_levels = collect(levels(y)),
        class_pool = CategoricalArrays.pool(y)
    )
    
    report = (n_models_trained=length(machines_list),)
    cache = nothing
    
    return fitresults, cache, report
end

"""
    MLJModelInterface.predict_mode(model, fitresult, Xnew)

    Implementa la **Votación Dura** (Mayoría).
"""
function MLJModelInterface.predict_mode(m::PartitionedVotingClassifier, fitresult, Xnew)

    machines       = fitresult.machines
    feature_groups = fitresult.feature_groups
    class_levels   = fitresult.class_levels
    
    n_samples = nrows(Xnew)
    n_models  = length(machines)

    # Matriz que almacena los votos
    votes = Matrix{eltype(class_levels)}(undef, n_samples, n_models)

    # Predicciones de cada modelo en su subespacio
    for (i, mach) in enumerate(machines)
        feats  = feature_groups[i]
        X_sub  = MLJBase.selectcols(Xnew, feats)
        votes[:, i] = MLJBase.predict_mode(mach, X_sub)
    end

    # vector de resultados
    final_predictions = similar(votes[:,1]) 

    # Contar votos y elegir ganador
    for i in 1:n_samples
        row_votes = @view votes[i, :] # generar vista de la fila de la matriz de votos

        # contar votos por clase
        counts = Dict{eltype(class_levels),Int}()
        for lbl in row_votes
            counts[lbl] = get(counts, lbl, 0) + 1
        end

        # Encontrar valor máximo de votos
        max_votes = maximum(values(counts))

        # Elegir ganador(es)
        winners = [lbl for (lbl,c) in counts if c == max_votes]

        # Caso 1: no hay empate
        if length(winners) == 1
            final_predictions[i] = winners[1]
            continue
        end

        # Caso 2: empate -> resolver por soft voting

        # Vector de probabilidades acumuladas
        total_probs = zeros(Float64, length(class_levels))

        # Para cada modelo, sumar probabilidades
        for (j, mach) in enumerate(machines)
            feats  = feature_groups[j]
            X_sub  = MLJBase.selectcols(Xnew, feats)

            # predicción probabilística del modelo base
            dist = MLJBase.predict(mach, X_sub)[i]

            # acumulamos
            probs = [pdf(dist, c) for c in class_levels]  # vector de probabilidades
            total_probs .+= probs
        end

        # se elige la clase con mayor porbabilidad
        max_idx = argmax(total_probs)
        final_predictions[i] = class_levels[max_idx]
    end

    return final_predictions
end

"""
    MLJModelInterface.predict(model, fitresult, Xnew)

    Envuelve la predicción dura en un formato Probabilístico para que el Pipeline no falle.
    Devuelve probabilidad 1.0 para la clase ganadora.
"""
function MLJModelInterface.predict(m::PartitionedVotingClassifier, fitresult, Xnew)

    # Obtener la clase ganadore
    yhat = MLJModelInterface.predict_mode(m, fitresult, Xnew)
    
    class_levels = fitresult.class_levels
    class_pool = fitresult.class_pool
    
    # Convertir a UnivariateFinite (probabilidad simulada)
    n_samples = length(yhat)
    
    dists = Vector{MLJBase.UnivariateFinite}(undef, n_samples)
    
    for i in 1:n_samples
        # Crear una distribución donde la probabilidad es 1.0 para la clase ganadora
        # y 0.0 para el resto.
        winner = yhat[i]
        probs = [c == winner ? 1.0 : 0.0 for c in class_levels]
        dists[i] = MLJBase.UnivariateFinite(class_levels, probs; pool=class_pool)
    end
    
    return dists
end

# Metadatos para que MLJ reconozca el modelo
MLJModelInterface.metadata_model(PartitionedVotingClassifier,
    input_scitype = Table(Continuous),
    target_scitype = AbstractVector{<:Finite},
    supports_weights = false,
    load_path = "PartitionedVotingClassifier"
)

# ==============================================================================
#  FACTORY: Voting Classifier
# ==============================================================================

function get_voting_classifier(cost::Float64, n_partitions::Int; rng::Int=104)

    base_model = MLJ_SVC(cost=cost,  kernel=LIBSVM.Kernel.Linear)

    return PartitionedVotingClassifier(
        model = base_model,
        n_partitions = n_partitions,
        rng = rng
    )

end

# ==============================================================================
#  FACTORY: Custom Resampling Strategy
# ==============================================================================

struct IndividualWiseCV <: MLJBase.ResamplingStrategy
    values::Vector{Tuple{Vector{Int}, Vector{Int}}}
end

function get_individual_wise_CV(folds::Vector{Tuple{Vector{Int}, Vector{Int}}})
    return IndividualWiseCV(folds)
end

function MLJBase.train_test_pairs(strategy::IndividualWiseCV, rows, X, y)
    return strategy.values
end

function MLJBase.train_test_pairs(strategy::IndividualWiseCV, rows)
    return strategy.values
end



# ==============================================================================
#  FACTORY: Stacking Ensemble
# ==============================================================================

function get_stacking_model(meta_model::Any, 
                    base_models::Dict{Symbol, <:Any}, 
                    resampling::Any)
    

    # Construir y devolver un ensemble Stacking
    return Stack(;metalearner = meta_model,
        resampling = resampling, 
        svm = base_models[:svm],
        knn = base_models[:knn],
        mlp = base_models[:mlp],
    )
end

end # module