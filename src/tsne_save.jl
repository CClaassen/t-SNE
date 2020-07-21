using Dates, DelimitedFiles, Gadfly
import Cairo, Fontconfig
GF = Gadfly

"""
File info:
Provides ways to save and plot the t-SNE output directly.
Can create .csv, .pdf and .svg files.

Variables:
Y:= The output dataset to be used
label:= The provided labels to recognize observations with (default: [])
moment:= The current time used to create unique savefiles, or a user provided name (default: "")

"""

#Writes 2d t-Sne output and provided labels to a .csv file
function output_csv(Y::AbstractMatrix, label::AbstractArray = [], moment = "")
    if !isempty(label)
        Y = [Y label]
    end

    if isempty(moment)
        moment = Dates.now()
        moment = replace("$moment", ['.', ':'] => "-" )
        moment = replace("$moment", 'T' => "_T" )
    end

    println("Saving t-SNE output to: tsne_output_$moment.csv")
    writedlm("tsne_output_$moment.csv", Y, ',')
end

#Writes 2d t-Sne output plot and provided labels to a .pdf file
function output_pdf(Y::AbstractMatrix, label::AbstractArray = [], moment = "")
    if isempty(moment)
        moment = Dates.now()
        moment = replace("$moment", ['.', ':'] => "-" )
        moment = replace("$moment", 'T' => "_T" )
    end

    println("Saving t-SNE plot to: tsne_plot_$moment.pdf")
    graph = GF.plot(x=Y[:,1], y=Y[:,2], color=label, Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey(title="Word type"), Theme(point_size=0.7mm, highlight_width=0.1mm))
    draw(PDF("tsne_plot_$moment.pdf", 15cm, 10cm), graph)
end

#Writes 2d t-Sne output plot and provided labels to a .svg file
function output_svg(Y::AbstractMatrix, label::AbstractArray = [], moment = "")
    if isempty(moment)
        moment = Dates.now()
        moment = replace("$moment", ['.', ':'] => "-" )
        moment = replace("$moment", 'T' => "_T" )
    end

    println("Saving t-SNE plot to: tsne_plot_$moment.svg")
    graph = GF.plot(x=Y[:,1], y=Y[:,2], color=label, Guide.xlabel(""), Guide.ylabel(""), Guide.colorkey(title="Word type"), Theme(point_size=0.7mm, highlight_width=0.1mm))
    draw(SVG("tsne_plot_$moment.svg", 15cm, 10cm), graph)
end

#Save 2d T-Sne output and the coresponding plot to .csv and .pdf respectively
function output_save(Y::AbstractMatrix, label::AbstractArray = [], moment = "")
    if isempty(moment)
        moment = Dates.now()
        moment = replace("$moment", ['.', ':'] => "-" )
        moment = replace("$moment", 'T' => "_T" )
    end

    output_csv(Y, labels, moment)
    output_pdf(Y, labels, moment)
end
