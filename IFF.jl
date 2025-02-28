using Statistics
using Random
using LinearAlgebra
using ProgressBars


# Implementation in Julia of Interesting Features Finder (IFF): another way to explore spectroscopic imaging data sets giving minor compounds and traces a chance to express themselves, Wu et al. 2022, Spectrochimica Acta Part B: Atomic Spectroscopy



function iff(sp, num_randvect)
     # Aim of the function: selection of the purest spectra in a data set
  
    # INPUTS
    #   sp : the data set you want to explore with spectra alongs the rows
    #   num_randvect : number of randow vectors to be generated (should be higher than 10000)
    # OUTPUTS
    #   ind: indexes of selected spectra (sorted by decreasing order of selection frequency)
    #   freq : selection frequency of each spectrum in ind
  
    # Calculate dimensions of the data set
    
    li, co = size(sp)
    
    # Mean centering of the data set
    m = mean.(eachcol(sp))
    sp_cent = sp .- transpose(m)

    # Generation of 'num_randvect' random vectors
    randvect = 2 .* rand(num_randvect, co) .- 1

    #Initialization of the selection frequency list
    votes = zeros(li)

    #Projection of the data onto a random vector in the loop
    for k in ProgressBar(1:num_randvect)
        tmp = transpose(randvect[k, :]) * transpose(sp_cent)
        ind_max = argmax(tmp)[2]
        votes[ind_max] += 1
        ind_min = argmin(tmp)[2]
        votes[ind_min] += 1
    end
    
    #Sorting spectra in a decreasing order of frequency selection
    sorted_indices = sortperm(votes, rev=true)
    freq = votes[sorted_indices]
    ind = sorted_indices

    return ind, freq

end




