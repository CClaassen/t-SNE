using Distances, LinearAlgebra, Statistics, MultivariateStats, StatsBase, Random

"""
File info:
Inserts a new point in the t-SNE embedding the using the weighted geometric median of dataset 'X'.
Geometric median is calculated using the Weiszfeld algorithm with modifications.
Also includes functions for standardizing and trimming similarity vectors.

Variables:
X:= The dataset to be used
sim:= The similarity weight vector (default: [])
info:= Boolean indicating whether divergence from mean and iterations should be returned (default: false)
cap:= The maximum amount of iterations (default: 1000)
eps:= Precision used as termination condition (default: 1e-6)

"""

#Calculate the weighted geometric median by performing Weiszfeld updates with similarity weights
function geo_median(X::AbstractMatrix, sim::AbstractArray = [], info::Bool=false, cap::Int=1000, eps::Number=1e-6; iter::Int = 0)
    obs,dim = size(X)

    #Check the data input dimensionality
    if dim == 1
        return median(X)
    end

    #Check similarity input
    if isempty(sim)
        sim = ones(1, obs)
    end
    mu_sim = mean(sim)
    adj_sim = sim/mu_sim

    #Initialize some things
    old = mean(X, weights(adj_sim), dims=1)
    new = zeros(1, dim)

    #Run the algorithm until convergence
    while norm(old - new)>eps
        num = zeros(1, dim)
        denom = 0
        for i = 1:obs
            d = norm(new - transpose(X[i,:]))/adj_sim[i]
            num += transpose(X[i,:])/d
            denom += 1/d
        end
        old = new
        new = num/denom
        iter += 1

        if iter >= cap
            println("No geometric median found at $iter iterations, using (weighted) mean instead.")
            return res = mean(X,weights(adj_sim), dims=1)
            break
        end
    end

    if !info
        return res = new
    else
        return res = new, norm(new - mean(X,weights(adj_sim), dims=1)), iter
    end
end

#Standardizes the cosine similarities to [0,1]
function std_sim(sim::AbstractArray)
    if !all(-1 .<= sim.<= 1)
        println("Warning: some similarities are not in [-1,1], adjusting...")
        sim = sim/maximum(abs.(sim))
        return sim = 0.5*sim.+ 0.5

    elseif !all(0 .<= sim.<= 1)
        return sim = 0.5*sim.+ 0.5

    else
        return sim
    end
end

#Nullifies similairities that are below some trimming factor
function trim_sim(sim::AbstractArray, trim_factor:: Number = 0.05)
    dim = size(sim)[1]

    for i=1:dim
        if sim[i] <= trim_factor
            sim[i] = 0
        end
    end

    return sim
end

#Insert a new point in the t-SNE embedding Y
function insert_point(X::AbstractMatrix, Y::AbstractMatrix, label::AbstractArray, word::String, transform_type::String = "pwr"; p::Number = NaN, t::Number = NaN)
    X = [X; get_embedding(word)']
    label = [label; "New: "*word]
    index = size(X)[1]

    sim = similarity(X)[(1:index-1), index]

    sim = std_sim(sim)

    if transform_type == "pwr"
        sim = power_transform(sim, p, t)
    elseif transform_type == "exp"
        sim = exp_transform(sim, p, t)
    end

    sim = trim_sim(sim, mean(sim))
    new_point = geo_median(Y, sim)
    Y = [Y; new_point]
    return X, Y , label
end

#Insert a new point by performing t-SNE on the entire dataset
#Note: old t-SNE embeddings should have some initial value for the new word
function insert_point_tsne(X::AbstractMatrix, label::AbstractArray, word::String; iter::Number = 5000, perp = 35)
    X = [X; get_embedding(word)']
    label = [label; "New: "*word]
    Y = tsne_main(X, 2, iter, perp, init_type = "cus")
    return X, Y, label
end

#Deprecated function, weighted geometric median integrated in geo_median
function geow_median(X::AbstractMatrix,sim::AbstractArray = [], div::Bool=false, cap::Number=1000, eps::Number=1e-6; iter::Number = 0)
    obs = size(X, 1)
    dim = size(X, 2)

    if isempty(sim)
        sim = ones(1, obs)
    end
    mu_sim = mean(sim)
    adj_sim = sim/mu_sim

    #old = mean(X, dims=1)
    old = mean(X, weights(adj_sim), dims=1)
    new = zeros(1, dim)

    while norm(old - new)>eps
        num = zeros(1, dim)
        denom = 0
        for i = 1:obs
            d = norm(new - transpose(X[i,:]))/adj_sim[i]
            num += transpose(X[i,:])/d
            denom += 1/d
        end
        old = new
        new = num/denom
        iter += 1

        if iter >= cap
            println("Caution:no geometric median found at $iter iterations, using (weighted) mean instead.")
            return res = mean(X, dims=1)
            break
        end
    end

    if div == false
        return res = new
    else
        return res = new, norm(new - mean(X,weights(adj_sim), dims=1)), iter
        #return res = new, norm(new - mean(X, dims=1))
    end
end
