module ModelFactory

using MLJ
using MLJFlux      
using Flux         
using LIBSVM       
using NearestNeighborModels 
using Random
using Optimisers

# Exportamos las funciones factoría
export get_knn_model, get_svm_model, get_mlp_model

const MLJ_SVC = @load ProbabilisticSVC pkg=LIBSVM verbosity=0

# ==============================================================================
# 1. BUILDER DINÁMICO PARA MLP 
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
# 2. FACTORY: KNN
# ==============================================================================
"""
Devuelve un modelo KNN configurado.
Parámetros:
  - k: Número de vecinos (default: 5)
"""
function get_knn_model(k::Int)
    # Nota: NearestNeighborModels usa 'K' mayúscula
    return KNNClassifier(K=k)
end

# ==============================================================================
# 3. FACTORY: SVM
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
# 4. FACTORY: MLP (RED NEURONAL)
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

end # module