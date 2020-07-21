using Distances, LinearAlgebra, Statistics, MultivariateStats, StatsBase, Random
"""
File info:
Provides various initializations options for the T-Sne algorithm.
Available options: (1) random, (2) PCA, (3) cMDS (cosine distance),
                   (4) cMDS (angular distance) and (5) user provided.
Variables:
X:= The dataset to be used
dim_out:= The number of dimensions in the T-Sne embedding (default: 2)

"""

#Generates a reproducable random initialization
function init_rnd(X::AbstractMatrix, dim_out::Int = 2, seed_val::Number = 456177)
    obs,dim = size(X)
    Random.seed!(seed_val)
    init = 0.001*randn(obs,dim_out)
    return init
end

#Generates a pca initialization
function init_pca(X::AbstractMatrix, dim_out::Int = 2)
    pca_model = fit(PCA, X', maxoutdim = dim_out)
    init = transform(pca_model, X')'
    init = 0.001*init/std(init[:,1])
    return init
end

#Generates a classical mds initialization based on cosine distance
function init_mds(X::AbstractMatrix, dim_out::Int = 2)
    D = similarity(X,true)
    mds_model = fit(MDS, D, maxoutdim = dim_out, distances = true)
    init = transform(mds_model)'
    init = 0.001*init/std(init[:,1])
    return init
end

#Generates a classical mds initialization based on angular distance
function init_mds_ang(X::AbstractMatrix, dim_out::Int = 2)
    D = angular(similarity(X,false))
    mds_model = fit(MDS, D, maxoutdim = dim_out, distances = true)
    init = transform(mds_model)'
    init = 0.001*init/std(init[:,1])
    return init
end

#Scales a user-provided custom initialization Y
#Defaults to init_pca of X if custom initialization is not usable
function init_cus(X::AbstractMatrix, Y::AbstractMatrix, dim_out::Int = 2)
    if size(Y)[2] < dim_out || size(Y)[1] != size(X)[1]
        println("Error: dimension mismatch, custom initialization cannot be used")
        return init = init_pca(X, dim_out)
    elseif size(Y)[2] > dim_out
        println("Warning: dimension mismatch, using PCA of custom initialization")
        return init = init_pca(Y, dim_out)
    else
        return init = 0.001*Y/std(X[:,1])
    end
end

#Not used
function init_sim(X::AbstractMatrix, dim_out::Int = 2)
end
