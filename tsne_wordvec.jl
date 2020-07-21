using DelimitedFiles, Embeddings

"""
File info:
This module loads the pre-trained word2vec Google news model
The methods for extracting the dataset from the model are also provided

Code adapted from: https://github.com/JuliaText/Embeddings.jl
"""


#Load the Google News pre-trained word2vec dataset
#Might require installation on first use, this can take some time
embtable = load_embeddings(Word2Vec)
get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

#Returns word2vec embeddings of a single word
function get_embedding(word::AbstractString)
    ind = get_word_index[word]
    emb = embtable.embeddings[:,ind]
    return emb
end

#Returns word2vec embeddings of multiple words
function get_embeddings(words::AbstractArray)
    dim = size(words)[1]
    emb = zeros(dim, 300)

    for i=1:dim
        index = get_word_index[words[i]]
        emb[i,:] = embtable.embeddings[:,index]
    end
    return emb
end

#Import the dataset that is used in the paper
function import_words()
    W = readdlm("thesis_words.csv", ',', String)
    label, words = W[:,1], W[:,2]
    return W, label, words
end
