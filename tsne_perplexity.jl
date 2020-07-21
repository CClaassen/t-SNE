using Distances, LinearAlgebra, Statistics, MultivariateStats, StatsBase, Random

"""
File info:
Calculates the local perplexity scores and the associated affinity matrix P

Variables:
sim:= The similarity matrix
treshold: = the minimial similarity treshold (default: 0.5)
k:= The number of standard deviations from the mean (default: 3)

"""

#Calculates the local perplexity score
function local_perp_score(sim::AbstractArray, treshold::Number = 0.5; k::Number = 3)
        dims = size(sim)[1]
        sim[diagind(sim)] .= 0 #Identical points do not count as neighbours
        local_perps, local_perps2 = zeros(Int, dims, 1), zeros(Int, dims, 1)

        mu, sig = mean(sim, dims = 1), std(sim, dims = 1)
        treshold2 = mu + k * sig

        for i=1:dims
            local_perps[i] = count(k->(k>=treshold), sim[:,i])
            local_perps2[i] = count(k->(k>=treshold2[i]), sim[:,i])
        end
        return local_perps, local_perps2
end

#Calculates the affinity matrix implied by the perplexity scores
function local_affinity(sim::AbstractArray, perp_score::AbstractArray)
        dims = size(sim)[1]
        sim[diagind(sim)] .= 0 #Identical points do not count as neighbours
        replace!(perp_score, 0=>1) #But all point need at least 1 neigbour
        res = zeros(dims, dims)

        for i=1:dims
            temp_index = collect(1:dims)
            temp_sim = sim[:, i]
            temp = Any[temp_sim temp_index]
            temp = temp[sortperm(temp[:, 1], rev = true), :]

            for j=1:perp_score[i]
                res[temp[j, 2], i] = temp[j, 1]
            end
        end

        return res ./ sum(res, dims =1)
end
