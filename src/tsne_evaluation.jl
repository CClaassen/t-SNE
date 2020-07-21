using Distances, LinearAlgebra, Statistics, MultivariateStats, StatsBase, Random, NearestNeighbors

"""
File info:
Contains functions for evaluating the local, global and point stuctural quality

Variables:
X:= The original dataset to be used
Y:= The embedding dataset to be used
k:= The amount of effective neighbours in the kNN algorithm

"""

#Evaluates the global structure of an embedding
function eval_global(X:: AbstractMatrix, Y:: AbstractMatrix)
    dist_low = euclid(Y)
    dist_high = euclid(X)
    res = diag(corspearman(dist_low, dist_high))

    return mean(res), std(res)
end

#Evaluates the local structure of an embedding
function eval_local(X:: AbstractMatrix, Y:: AbstractMatrix, k::Integer = 10)
    res = zeros(size(X)[1], 1)
    tree_low = BruteTree(Y')
    btree_high = BruteTree(X')

    #check k most similair for each word
    for i=1:size(X)[1]
        calc = 0
        neighbour_high = knn(btree_high, X[i, :], k)[1]
        neighbour_low = knn(tree_low, Y[i, :], k)[1]

        for j=1:k
            calc += sum(x-> x==neighbour_high[j], neighbour_low)
        end

        res[i] = calc/k
    end
    return mean(res), std(res)
end

#Evaluates a newly inserted point
function eval_point(X:: AbstractMatrix, Y:: AbstractMatrix, k::Integer = 25; index = size(X)[1])
    tree_low = BruteTree(Y')
    btree_high = BruteTree(X')

    #check k closest points for each point
    neighbour_high = knn(btree_high, X[index, :], k)[1]
    neighbour_low = knn(tree_low, Y[index, :], k)[1]

    calc = 0
    for j=1:k
        calc += sum(x-> x==neighbour_high[j], neighbour_low)
    end
    res = calc/k

    dist_low = euclid(Y)
    dist_high = euclid(X)
    res2 = corspearman(dist_low[:, index], dist_high[:, index])

    result = [res res2]'
    return result
end
