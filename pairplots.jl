colmap = ["red", "green", "blue"]

function threecolors(class; classes=classes, cmap=colmap)
    for (ind, val) in enumerate(classes)
        if class == val
            return(cmap[ind])
        end
    end
    return("")
end

colors = threecolors.(cl);

# Subplots

heights = [1, 1, 1, 1]
widths = [1, 1, 1, 1]

fig = figure(constrained_layout=false, figsize=(10.0,10.0))

subplots_adjust(hspace=0.0) # Set the height spacing between subplots
subplots_adjust(wspace=0.0) # Set the width spacing between subplots

spec = fig.add_gridspec(
    ncols=length(widths),
    nrows=length(heights),
    width_ratios=widths,
    height_ratios=heights
)

for row in eachindex(heights)
    for col in eachindex(widths)
        ax = fig.add_subplot(spec[row, col])
        if row != 4
            setp(ax.get_xticklabels(), visible=false); # Disable x tick labels
        end
        if col != 1
            setp(ax.get_yticklabels(), visible=false); # Disable y tick labels
        end
        if col == row
            annotate(colnames[col], xy=[0.5,0.5], ha="center", va="center") 
        else
            scatter(df[!, colnames[col]], df[!, colnames[row]], marker=".", c=colors)
            grid(true)
        end
    end
end

fig.legend(bbox_to_anchor = (1.0, 0.93), loc="center right", labels=classes, labelcolor=colmap)
suptitle("Iris dataset")
