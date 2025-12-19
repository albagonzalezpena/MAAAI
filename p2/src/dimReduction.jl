module ProyectionReduction

"""
    Este módulo contiene las fucniones factory necesarias
    para obtener métodos de reducción de dimensionalidad por proyección
    de manera sencilla.
"""

using MLJ
using MLJBase
using MultivariateStats
using LinearAlgebra
using CategoricalArrays

export get_lda_model, get_pca_model, get_ica_model

const MLJ_PCA = @load PCA pkg=MultivariateStats verbosity=0
const MLJ_ICA = @load ICA pkg=MultivariateStats verbosity=0
const MLJ_LDA = @load LDA pkg=MultivariateStats verbosity=0


# ==============
# 2. FACTORY  
# ==============

"""
    Devuelve un modelo PCA.

    Params:
        - pratio: ratio de explicabilidad.
        - outdim: features post-proyección deseadas.
    
        Se prioriza outdim a pratio.
"""
function get_pca_model(; pratio=0.99, outdim=nothing)

    if isnothing(outdim)
        return MLJ_PCA(variance_ratio=pratio)
    else
        return MLJ_PCA(maxoutdim=outdim)
    end
end

"""
    Devuelve un modelo LDA.

    Params:
        - outdim: número de dimensiones finales deseadas.
"""
function get_lda_model(; outdim=10)
    return MLJ_LDA(outdim=outdim)
end

"""
    Devuelve un modelo ICA implementado con Fast ICA.

    Params:
        - outdim: numero de dimensiones de salida deseadas.
        - maxiter: número máximo de iteraciones permitidas.
        - do_whiten: permite seleccinar si se hace blanqueamiento de datos.
        - tol: tolerancia que se otorga a ICA.
"""

function get_ica_model(; outdim=10, maxiter=500, do_whiten=true, tol=1e-4)
    return MLJ_ICA(outdim=outdim, maxiter=maxiter, tol=tol, do_whiten=do_whiten)
end

end # module