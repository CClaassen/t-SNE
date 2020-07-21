using Distances, LinearAlgebra, Statistics, MultivariateStats
using StatsBase, Random, Distributed, SharedArrays
using LinearAlgebra, ProgressMeter

using Printf: @sprintf

"""
File info:
Contains the main t-SNE function and associated functions. Tnse_main(ARGS) generates
a representative 2d embedding by extending the original t-SNE algorithm.

Additional features: (1) different initialization options, (2) late exaggeration,
                     (3) local perplexity scaling and (4) new point insertion.

Required paramters:
X:= The original dataset to be used
dims: = The embedding dimensionality integer (default: 2)
iter_cap: = The maximum number of iterations (default: 5000)
perp: = The global perplexity number (default: 30)

Optional parameters:
init_type:= A string determining the initialization type (default: "pca")
pre_proces:= The number of dimensions the original dataset should reduced to (default: 0)
local_p:= A boolean determining if local perplexities should be used (default: false)
verbose:= A boolean determing if additional info should be displayed (default: false)
info:= A boolean determining whether additional info should be returned (default: false)

Inspired by the original codes and ideas of:
Original t-SNE (see: https://lvdmaaten.github.io/tsne/)
Recommended Julia implementation of t-SNE (see: https://github.com/lejon/TSne.jl)
"""

#Runs the t-SNE algorithm (https://lvdmaaten.github.io/tsne/) with extensions
function tsne_main(X::Union{AbstractMatrix, AbstractVector}, dims::Integer = 2, iter_cap::Integer = 5000, perp::Number = 30.0;
              min_gain::Number = 0.01, eta::Number = 200.0, init_type::String = "pca", pre_proces::Integer = 0, local_p::Bool = false,
              initial_momentum::Number = 0.5, final_momentum::Number = 0.8, momentum_switch_iter::Integer = 250,
              stop_cheat_iter::Integer = 250, start_cheat::Integer = iter_cap - 250, cheat_scale::Number = 12.0,
              verbose::Bool = false, progress::Bool=true,
              info = false)


    #Initialize low-dimensional space
    if init_type == "rnd"
        Y = init_rnd(X, dims)
    elseif init_type == "pca"
        Y = init_pca(X, dims)
    elseif init_type == "mds"
        Y = init_mds(X, dims)
    elseif init_type == "mds_ang"
        Y = init_mds_ang(X, dims)
    elseif init_type == "cus"
        println("Enter [FILENAME] of a '[FILENAME].csv' here:")
        name = readline()
        Z = readdlm("$name.csv", ',', Float64)
        Y = init_cus(X, Z)
    end

    verbose && @info("Using the $init_type initialization")

    #Check if high-dimensional data should be reduced
    if pre_proces > 2 && pre_proces < size(X)[2]
        X = init_pca(X, pre_proces)
        verbose && @info("Reducing dimensionality to $pre_proces")
    elseif pre_proces > size(X)[2]
        verbose && @info("Warning: Cannot change dimensionality to $pre_proces")
    end


    #Compute distance and similarity
    S = similarity(X)
    D = euclid(X)

    #Check if local perplexities should be used
    if local_p == true
        score1, score2 = local_perp_score(S, k = 3)
        P, beta = perplexity(D, 1e-5,  perp)
        P2 = local_affinity(S, score2)
        P = 0.5(P+P2)
    else
        P, beta = perplexity(D, 1e-5, perp)
    end

    P .+= P'
    P .*= cheat_scale/sum(P) # normalize + early exaggeration
    sum_P = cheat_scale

    #Initial gradient
    dY, iY, gains  = fill!(similar(Y), 0), fill!(similar(Y), 0), fill!(similar(Y), 1)

    n = size(X, 1)

    # Run iterations of t-SNE
    progress && (pb = Progress(iter_cap, "Computing t-SNE"))
    Q = fill!(similar(P), 0)
    Ymean = similar(Y, 1, dims)
    sum_YY = similar(Y, n, 1)
    L = fill!(similar(P), 0)
    Lcolsums = similar(L, n, 1)

    last_kldiv = NaN
    for iter in 1:iter_cap

        # Compute pairwise affinities
        BLAS.syrk!('U', 'N', 1.0, Y, 0.0, Q)
        for i in 1:size(Q, 2)
            sum_YY[i] = Q[i, i]
        end

        sum_Q = 0.0
        for j in 1:size(Q, 2)
            sum_YYj_p1 = 1.0 + sum_YY[j]
            Qj = view(Q, :, j)
            Qj[j] = 0.0
            for i in 1:(j-1)
                sqdist_p1 = sum_YYj_p1 - 2.0 * Qj[i] + sum_YY[i]
                Qj[i] = ifelse(sqdist_p1 > 1.0, 1.0 / sqdist_p1, 1.0)
                sum_Q += Qj[i]
            end
        end

        sum_Q *= 2 # lower-triangular part of Q is zero

        inv_sum_Q = 1.0 / sum_Q
        fill!(Lcolsums, 0.0)

        for j in 1:size(L, 2)
            Lj = view(L, :, j)
            Pj = view(P, :, j)
            Qj = view(Q, :, j)
            Lsumj = 0.0
            for i in 1:(j-1)
                Lj[i] = l = (Pj[i] - Qj[i]*inv_sum_Q) * Qj[i]
                Lcolsums[i] += l
                Lsumj += l
            end
            Lcolsums[j] += Lsumj
        end

        for (i, ldiag) in enumerate(Lcolsums)
            L[i, i] = -ldiag
        end

        #input for update scheme
        BLAS.symm!('L', 'U', -4.0, L, Y, 0.0, dY)

        #Execute the update
        #check for momentum switch
        momentum = iter <= momentum_switch_iter ? initial_momentum : final_momentum

        for i in eachindex(gains)
            gains[i] = max(ifelse((dY[i] > 0) == (iY[i] > 0),
                                  gains[i] * 0.8,
                                  gains[i] + 0.2),
                                  min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        Y .-= mean!(Ymean, Y)

        #Stop cheating for now
        if sum_P != 1.0 && iter >= stop_cheat_iter
            P .*= 1/sum_P
            sum_P = 1.0
        end

        #Start cheating again
        if sum_P != 1.0 && iter >= start_cheat
            P .*= cheat_scale
            sum_P = cheat_scale
        end

        #Compute cost function value
        if !isfinite(last_kldiv) || iter == iter_cap ||
            (progress && mod(iter, max(iter_cap÷20, 10)) == 0)
            local kldiv = 0.0
            for j in 1:size(P, 2)
                Pj = view(P, :, j)
                Qj = view(Q, :, j)
                kldiv_j = 0.0

                for i in 1:(j-1)
                    kldiv_j += kldivel(Pj[i], Qj[i])
                end

                kldiv += 2*kldiv_j + kldivel(Pj[j], Q[j])
            end

            last_kldiv = kldiv/sum_P + log(sum_Q/sum_P)
        end

        progress && update!(pb, iter,
                            showvalues = Dict(:KL_divergence => @sprintf("%.4f%s", last_kldiv,
                                                                         iter <= stop_cheat_iter ? " (warmup)" : "")))
    end
    progress && (finish!(pb))
    verbose && @info(@sprintf("Final t-SNE KL-divergence=%.4f", last_kldiv))

    # Return solution
    if !info
        return Y
    else
        return Y, last_kldiv
    end
end

#Kullbeck-Leibler shortcut
kldivel(p, q) = ifelse(p > zero(p) && q > zero(q), p*log(p/q), zero(p))

#Calculate the point peplexities from the pairwisr dist
function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
    P .= exp.(-beta .* D)
    sum_P = sum(P)
    H = log(sum_P) + beta * dot(D, P)  / sum_P

    P .*= 1/sum_P

    return H
end

#Calculate perplexity matrix from the distance matrix through a search
function perplexity(D::AbstractMatrix{T}, eps::Number = 1e-5, perp::Number = 30.0; iter_cap::Integer = 50,) where T<:Number

    #initialize some variables for later use
    dim = size(D)[1]
    P = fill(zero(T), dim, dim)
    beta = fill(one(T), dim)
    H_t = log(perp)
    Dist_i = fill(zero(T), dim)
    P_col = fill(zero(T), dim)


    for i=1:dim
        #set initial beta values
        beta_i = 1.0
        beta_min = 0.0
        beta_max = Inf

        #set initial individual distances for all
        copyto!(Dist_i, view(D, :, i))
        Dist_i[i] = prevfloat(Inf) #to get the biggest workable number
        min_D = minimum(Dist_i)
        Dist_i .-= min_D

        #get values from hbeta function
        H = Hbeta!(P_col, Dist_i, beta_i)
        H_d = H - H_t
        iter = 0
        #Calculate perplexity until termination condition is met
        while iter < iter_cap && abs(H_d) > eps

            if H_d > 0
                beta_min = beta_i

                if !isfinite(beta_max)
                    beta_i = 2*beta_i
                else
                    beta_i = 0.5*(beta_i+beta_max)
                end

            else
                beta_max = beta_i
                beta_i = 0.5(beta_i + beta_min)
            end

            H = Hbeta!(P_col, Dist_i, beta_i)
            H_d = H - H_t

            #Add iteration counter and start over
            iter = iter + 1
        end

        P[:, i] .= vec(P_col)
        beta[i] = beta_i

    end

    return P, beta
end
