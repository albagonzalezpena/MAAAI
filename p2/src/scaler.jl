module CustomScalers

"""
    Este módulo contiene la lógica del min-max wrapper que usaremos 
    para normalizar los datos. Sigue el esquema definido por MLJ para que
    sea insertable en el pipeline del experimento.
"""

using MLJ
using MLJBase
using Statistics
using Tables

export MinMaxScaler

mutable struct MinMaxScaler <: MLJ.Unsupervised
end

"""
    Función fit que calcula los factores de escalado.
"""
function MLJBase.fit(model::MinMaxScaler, verbosity::Int, X)
    # Convertimos a matriz para calcular rápido
    X_mat = MLJBase.matrix(X)
    
    # Calculamos min y max por feature 
    min_vals = minimum(X_mat, dims=1)
    max_vals = maximum(X_mat, dims=1)
    
    # Evitar división por cero si una columna es constante 
    denominators = max_vals .- min_vals
    replace!(denominators, 0.0 => 1.0)
    
    # Guardamos los valores para usar en transform
    fitresult = (min_vals, denominators, Tables.schema(X).names)
    cache = nothing
    report = NamedTuple()
    
    return fitresult, cache, report
end

"""
    Método transform que normaliza el dataset.
"""
function MLJBase.transform(model::MinMaxScaler, fitresult, X)
    min_vals, denominators, names = fitresult
    
    X_mat = MLJBase.matrix(X)
    
    # Aplicamos el escalado
    X_scaled = (X_mat .- min_vals) ./ denominators
    
    # Devolvemos una tabla limpia
    return MLJBase.table(X_scaled, names=names)
end

# Metadatos para mantener coherencia de datos en el pipeline
MLJBase.input_scitype(::Type{<:MinMaxScaler}) = Table(Continuous, Missing)
MLJBase.output_scitype(::Type{<:MinMaxScaler}) = Table(Continuous)

end # module