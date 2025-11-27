module CustomScalers

using MLJ
using MLJBase
using Statistics
using Tables

export MinMaxScaler

mutable struct MinMaxScaler <: MLJ.Unsupervised
    # No tiene hiperparámetros, escala siempre a [0, 1]
end

function MLJBase.fit(model::MinMaxScaler, verbosity::Int, X)
    # Convertimos a matriz para calcular rápido
    X_mat = MLJBase.matrix(X)
    
    # Calculamos min y max por columna (dims=1)
    min_vals = minimum(X_mat, dims=1)
    max_vals = maximum(X_mat, dims=1)
    
    # Evitar división por cero si una columna es constante (max == min)
    # Si son iguales, el denominador será 1.0 (no cambia nada)
    denominators = max_vals .- min_vals
    replace!(denominators, 0.0 => 1.0)
    
    # Guardamos los valores para usar en transform
    fitresult = (min_vals, denominators, Tables.schema(X).names)
    cache = nothing
    report = NamedTuple()
    
    return fitresult, cache, report
end

function MLJBase.transform(model::MinMaxScaler, fitresult, X)
    min_vals, denominators, names = fitresult
    
    X_mat = MLJBase.matrix(X)
    
    # Aplicamos la fórmula: (X - min) / (max - min)
    X_scaled = (X_mat .- min_vals) ./ denominators
    
    # Devolvemos una tabla limpia
    return MLJBase.table(X_scaled, names=names)
end

# Metadatos para que MLJ sepa qué entra y qué sale
MLJBase.input_scitype(::Type{<:MinMaxScaler}) = Table(Continuous, Missing)
MLJBase.output_scitype(::Type{<:MinMaxScaler}) = Table(Continuous)

end # module