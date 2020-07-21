using Distances, LinearAlgebra, Statistics, MultivariateStats
using StatsBase, Random, Distributed, SharedArrays

"""
File info:
Provides various pairwise distance and similiarty (peudo-)metrics.
Contains cosine distance, cosine similarity, Euclidean distance and angular distance.

Variables:
X:= The dataset to be used
Bool:= The boolean indicating if distance or similarity should be calculated (default: false)

"""

#Shortcut for cosine similarity calculation
function cos_sim(x::AbstractArray, y::AbstractArray)
    return 1 - cosine_dist(x,y)
end

#Calculates the pairwise cosine similarity for a dataset
function similarity(X::AbstractMatrix, dist::Bool = false)
    obs,dim = size(X)
    sim = zeros(obs,obs)

    if dist == false
        for i=1:obs-1
            for j=i+1:obs
               sim[i,j] = cos_sim(X[i,:],X[j,:])
           end
        end
        sim = sim + sim'
        sim[diagind(sim)] .= 1
    else
        for i=1:obs-1
            for j=i+1:obs
               sim[i,j] = cosine_dist(X[i,:],X[j,:])
           end
        end
        sim = sim + sim'
        #sim[diagind(sim)] .= 0
    end

    return sim
end

#Calculates the pairwise Euclidean distance for a dataset
function euclid(X::AbstractMatrix)
    obs,dim = size(X)
    dist = zeros(obs,obs)
    for i=1:obs
        for j=i+1:obs
                dist[i,j] = euclidean(X[i,:],X[j,:])
        end
    end
    dist = dist + dist'
    #dist[diagind(dist)] .= 0
    return dist
end

#Calculates the pairwise angular distance implied by cosine similarities
function angular(X::AbstractMatrix, dist::Bool = true)
    A = 2*acos.(X)/pi

    if dist == false
        A = 1 .-A
    end

    return A
end

#Not used: function for hyperthreading similarity, not consistenly faster
function similarity_hyper(X::AbstractMatrix, dist::Bool = false)
    obs,dim = size(X)
    sim =  SharedArray(zeros(obs,obs))

    if dist == false
        Threads.@threads for i=1:obs
               for j=i+1:obs
               sim[i,j] = cos_sim(X[i,:],X[j,:])
           end
        end
        sim = sim + sim'
        sim[diagind(sim)] .= 1
    else
        Threads.@threads for i=1:obs
            Threads.@threads for j=i+1:obs
               sim[i,j] = cosine_dist(X[i,:],X[j,:])
           end
        end
        sim = sim + sim'
        #sim[diagind(sim)] .= 0
    end

    return sim
end
