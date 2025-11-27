module ProyectionReduction

using MLJ
using MLJBase
using MultivariateStats
using LinearAlgebra
using CategoricalArrays

# Exportamos el Wrapper propio y funciones helper para los nativos
export get_lda_model, get_pca_model, get_ica_model

const MLJ_PCA = @load PCA pkg=MultivariateStats verbosity=0
const MLJ_ICA = @load ICA pkg=MultivariateStats verbosity=0
const MLJ_LDA = @load LDA pkg=MultivariateStats verbosity=0


# ==============
# 2. FACTORY  
# ==============

function get_pca_model(; pratio=0.99, outdim=nothing)

    if isnothing(outdim)
        return MLJ_PCA(variance_ratio=pratio)
    else
        return MLJ_PCA(maxoutdim=outdim)
    end
end

function get_lda_model(; outdim=10)
    return MLJ_LDA(outdim=outdim)
end

function get_ica_model(; outdim=10, maxiter=500, do_whiten=true, tol=1e-4)
    return MLJ_ICA(outdim=outdim, maxiter=maxiter, tol=tol, do_whiten=do_whiten)
end

end # module