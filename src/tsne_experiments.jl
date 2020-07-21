using RDatasets
using TSne

include("tsne_init.jl")
include("tsne_distances.jl")
include("tsne_wordvec.jl")
include("tsne_perplexity.jl")
include("tsne_weights.jl")
include("tsne_insert.jl")
include("tsne_main.jl")
include("tsne_evaluation.jl")
include("tsne_save.jl")

"""
File info:
Contains functions that reproduce the figures and tables of the paper

"""

#Makes figure 1
function figure1()
    W, label, words = import_words()
    emb = get_embeddings(words)
    Y = init_pca(emb)
    graph = GF.plot(x=Y[:,1], y=Y[:,2], color=label, Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey(title="Word type"))
    draw(PDF("figure1.pdf", 15cm, 10cm), graph)
end

#Makes figure 2
function figure2()
    #for variable points
    pts = 100
    X, label0 = collect(range(0.001,stop=1,step=1/pts)), fill("Base similarities", pts)
    Y1, label1 = power_transform(X,0.5), fill("Power transform (f1, p = 0.5)", pts)
    Y2,label2 = power_transform(X,5), fill("Power transform (f1, p = 5)", pts)
    Y3,label3 = power_transform(X,100), fill("Power transform (f1, p = 100)", pts)
    Y4, label4 = exp_transform(X,0.5), fill("Exponential transform (f2, p = 0.5)", pts)
    Y5,label5 = exp_transform(X,5), fill("Exponential transform (f2, p = 5)", pts)
    Y6,label6 = exp_transform(X,100), fill("Exponential transform (f2, p = 100)",pts)
    Y = [X X; X Y1; X Y2; X Y3; X Y4; X Y5; X Y6]
    label = [label0;label1;label2;label3;label4;label5;label6]
    graph = GF.plot(x=Y[:,1], y=Y[:,2], color=label, Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey(title="Transform type"), Geom.line)
    #graph = GF.plot(x=Y[:,1], y=Y[:,2], color=label, Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey(title="Transform type"), Geom.point, style(point_size=1mm))
    draw(PDF("figure2.pdf", 15cm, 10cm), graph)
end

#Makes figure 3
function figure3()
    W, labels, words = import_words()
    X = get_embeddings(words)
    Y = tsne_main(X, 2, 10000, 25)
    output_pdf(Y, labels, "figure3")
end

#Makes table 1 and figure 4
function replicate_iris()
    iris = dataset("datasets","iris")
    X = convert(Matrix{Float64}, iris[:, 1:4])
    labels = iris[:, 5]
    initial_dims = -1
    iterations = 10000
    perp = 15

    #RND = init_rnd(X)
    #res1, res2 = eval_local(RND,X)

    PC = init_pca(X)
    res3, res4 = eval_local(PC,X)

    Y = tsne(X, 2, initial_dims, iterations, perp, pca_init = true)
    res5, res6 = eval_local(Y,X)

    Y2 = tsne_main(X, 2,  iterations, perp)
    res7, res8 = eval_local(Y2,X)

    #output_pdf(Y2, labels, "figure3_1")
    #output_pdf(Y, labels, "figure3_2")
    graph = GF.plot(x=Y[:,1], y=Y[:,2], color=label, Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey(title="Iris species"), Theme(point_size=0.7mm, highlight_width=0.1mm, key_position=:bottom))
    graph2 = GF.plot(x=Y2[:,1], y=Y2[:,2], color=label, Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey(title="Iris species"), Theme(point_size=0.7mm, highlight_width=0.1mm, key_position=:bottom))
    draw(PDF("figure4.pdf", 15cm, 10cm), stack)

    stack = hstack(graph, graph2)

    result = [res3 res4; res5 res6; res7 res8]
    output_csv(result, [], "table1")

    return result
end

#Makes table 2
function table2()
    res = zeros(10, 5)
    W, labels, words = import_words()
    X = get_embeddings(words)

    for i=1:10
        Y, kl_div = tsne_main(X, 2, 10000, 5*i, info = true)

        res[i, 1] = kl_div
        res[i, 2], res[i, 3] = eval_local(X, Y)
        res[i, 4], res[i, 5] = eval_global(X, Y)
    end

    output_csv(res, [], "table2")
    return res
end

#Makes table 3
function table3()
    res = zeros(10, 5)
    W, labels, words = import_words()
    X = get_embeddings(words)

    for i=1:10
        Y, kl_div = tsne_main(X, 2, 10000, 5*i, local_p = true, info = true)

        res[i, 1] = kl_div
        res[i, 2], res[i, 3] = eval_local(X, Y)
        res[i, 4], res[i, 5] = eval_global(X, Y)
    end

    output_csv(res, [], "table3")
    return res
end

#Makes table 4 and figure 5
function table4()
    res = zeros(5, 2)
    W, labels, words = import_words()
    X = get_embeddings(words)
    Y = tsne_main(X, 2, 10000, 35, local_p = true)
    add_words = ["econometrics"; "hippo"; "Mexico"; "parent"; "statistician"]
    for i=1:5
        X, Y, labels = insert_point(X, Y, labels, add_words[i], p = 15)
        res[i,:] = eval_point(X,Y)
    end
    #output_pdf(Y, labels, "figure5")
    #!reverse labels!
    Y = [Y[602:606,:]; Y[1:601,:]]
    labels = [labels[602:606,:]; labels[1:601,:]]
    output_pdf(Y, labels, "figure5")
    output_csv(res, [], "table4")
    return add_words, res
end

#Makes figure 5 without the newly inserted points
function figure5_alt()
    W, labels, words = import_words()
    X = get_embeddings(words)
    Y = tsne_main(X, 2, 10000, 35, local_p = true)
    output_pdf(Y, labels, "figure5_alt")
end
