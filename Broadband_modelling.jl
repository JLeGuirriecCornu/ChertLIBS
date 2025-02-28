using CSV
using DataFrames
using StatsBase
using Random
using Jchemo, CairoMakie, FreqTables
using ProgressBars
using LinearAlgebra
using ThreadsX
using ArgParse


# ======================================================================================
# command line interface

# parse function 
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--processing", "-p"
            help = "processing done"
            arg_type = String
            default = "AirPLS"
        "--multiblock", "-m"
            help = "compute multiblock models"
            action = :store_true
        "--wavelength", "-w"
            help = "compute only predefined interval (default false)"
            action = :store_true
        "--verbose", "-v"
            help = "show prints? (default false)"
            action = :store_true
        "--resampled", "-r"
            help = "use ressampled indexes for multiblock? (default false)"
            action = :store_true
        "file"
            help = "csv file to process"
            required = true
    end
    return parse_args(s)
end

# call parsing 
parsed_args = parse_commandline()

# assign arguments
const File::String = parsed_args["file"]
const processing::String = parsed_args["processing"]
const multiblock::Bool =  parsed_args["multiblock"]
const wavelength::Bool =  parsed_args["wavelength"]
const verbose::Bool =  parsed_args["verbose"]
const resampled::Bool =  parsed_args["resampled"]


# print 
if verbose
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("  $arg  =>  $val")
    end
    nthreads = Threads.nthreads()
    println("number of threads: $nthreads")  
end


# ======================================================================================
# Data

# Data
Geol_data = CSV.read(File, DataFrame, delim = ',')

# exclude the first two shots 
Geol_data = Geol_data[.!(x -> x in [1, 2, 3, 4, 5, 6, 7, 8]).(Geol_data.shot_number), :]

# Renaming collumns
rename!(Geol_data, :Nom => :Sample)
rename!(Geol_data, :Formation => :Source)
select!(Geol_data, Not(:Column1))
#filter!(row -> row.Sample != "OBS_MAOA-XRF", Geol_data) 

# wavelength
if wavelength
    const wl_range = [188,680] # 945, [188,370], [450,680]
    # Select Wavelength
    wl_index = [[1,2,3,4,5,6];6 .+ findall(x-> x >= wl_range[1] && x <= wl_range[2], parse.(Float64, names(Geol_data)[6:end]))]
    #wl_index = [[1,2,3,4,5,6];6 .+ findall(x-> (x >= wl_range[1] && x <= wl_range[2]) | (x >= wl_range2[1] && x <= wl_range2[2]), parse.(Float64, names(Geol_data)[6:end]))]
    Geol_data = Geol_data[:,wl_index]
end

# Variables for later
Sources = sort(unique(Geol_data[!,:Source]))
nSources = length(Sources)

# Create set of equal size for each group
Geol_grouped = groupby(Geol_data, :Source)

# output directory (needs try-except)
mkdir(processing)


# ======================================================================================
# settings 

# Eperiment parameterswr
const nrep::Int = 200
const nlv::Int = 30
const train_set_size::Int = 3000
const valid_set_size::Int = 100
const LnOCV::Int = 1

# model parameters

# svm 
kern_svm = :kpol
gamma_svm::Float64 = 0.1
cost_svm::Int = 1000
epsilon_svm::Float64 = 0.5;

# pca-svm
kern_pca = :krbf # radial
gamma_pca::Float64 = 0.001
cost_pca::Int = 1000
epsilon_pca::Float64 = 0.5    
scal_pca::Bool = true

# DK
kern_dk = :kpol

# S
meth_s = :soft

# MB
if resampled
    listbl_mb = [1:1771, 1773:3471,3473:(size(Geol_data, 2)-8)] # -8   
else
    listbl_mb = [1:2010, 2013:4008, 4011:(size(Geol_data, 2)-8)] # -8
end
bscal_mb = :mfa # :none, :frob, :mfa, :ncol, :sd
scal_mb::Bool = false


# LW
nlvdis_lw = nlv
metric_lw = :eucl
h_lw = 5 
k_lw = 400


# ======================================================================================
# Initialization of results dataframe

# PLS-DA
plsrda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)                                    # nSources*nlv
rename!(plsrda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]]) # ["LV$lv$class" for lv in 1:nlv for class in Sources]

# PLS-LDA
plslda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(plslda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# PLS-QDA
plsqda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(plsqda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# SVM
svmda_results = DataFrame(Matrix{Any}(undef, 0, (5)), :auto)
rename!(svmda_results, ["RunInfo", "Sample","Spectrum", "Group", "Prediction"])

# PCA-SVM
pca_svmda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(pca_svmda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# dk-PLS-DA
dkplsrda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(dkplsrda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# dk-PLS-LDA
dkplslda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(dkplslda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# dk-PLS-QDA
dkplsqda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(dkplsqda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# s-PLS-DA
splsrda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(splsrda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# s-PLS-LDA
splslda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(splslda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# s-PLS-QDA
splsqda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(splsqda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# PLS-KDE-DA
plskdeda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(plskdeda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# s-PLS-KDE-DA
splskdeda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(splskdeda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# MBPLSR-DA
mbplsrda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(mbplsrda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# MBPLSR-LDA
mbplslda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(mbplslda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# MBPLSR-QDA
mbplsqda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(mbplsqda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# kNN-LWPLSR-DA
lwplsrda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(lwplsrda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# kNN-LWPLSR-LDA
lwplslda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(lwplslda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# kNN-LWPLSR-QDA
lwplsqda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(lwplsqda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

# ======================================================================================
# Main model functions

### basic family ------------

# PLS-DA
function plsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.plsrda(; nlv=nlv) 
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...)) # hcat(res.posterior...)
    for k in 1:ntest
        push!(plsrda_results,tmp_results[k,:])
    end
    #mod.fitm.fitm.W * LinearAlgebra.diagm(mod.fitm.fitm.C[1,:])
end

# PLS-LDA
function plslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.plslda(; nlv=nlv) 
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(plslda_results,tmp_results[k,:])
    end  
    #mod.fitm.fitm.embfitm.W * LinearAlgebra.diagm(mod.fitm.fitm.embfitm.C[1,:])
end

# PLS-QDA
function plsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.plsqda(; nlv=nlv) 
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(plsqda_results,tmp_results[k,:])
    end
    #mod.fitm.fitm.embfitm.W * LinearAlgebra.diagm(mod.fitm.fitm.embfitm.C[1,:])
end

### SVM family --------------

# SVM
function svmda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.svmda(; kern=kern_svm, gamma=gamma_svm, cost=cost_svm, epsilon=epsilon_svm)
    Jchemo.fit!(mod, Xtrain, ytrain);
    res = Jchemo.predict(mod, Xtest)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred))
    for k in 1:ntest
        push!(svmda_results,tmp_results[k,:])
    end    
end

# PCA-SVM (loop for a set of variables)
function pca_svmda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # loop til nlv
    tmp_matrix = Array{String15}(undef, ntest, 0)
    for lv in 1:nlv
        # model, fit and predict
        mod1 = Jchemo.pcasvd(; nlv=lv, scal=scal_pca)
        mod2 = Jchemo.svmda(; kern=kern_pca, gamma=gamma_pca, cost=cost_pca, epsilon=epsilon_pca)
        mod = Jchemo.pip(mod1, mod2)
        Jchemo.fit!(mod, Xtrain, ytrain) ;
        res = Jchemo.predict(mod, Xtest) ;
        # store
        tmp_matrix = hcat(tmp_matrix, vec(res.pred))
    end
    # store results                                                       
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest), hcat(tmp_matrix))
    for k in 1:ntest
        push!(pca_svmda_results,tmp_results[k,:])
    end   
end

### DK family ---------------

# dk-PLS-DA
function dkplsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.dkplsrda(; nlv=nlv, kern=kern_dk)
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(dkplsrda_results,tmp_results[k,:])
    end  
end

# dk-PLS-LDA
function dkplslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict 
    mod = Jchemo.dkplslda(; nlv=nlv, kern=kern_dk)
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results                              
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(dkplslda_results,tmp_results[k,:])
    end  
end

# dk-PLS-QDA
function dkplsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.dkplsqda(; nlv=nlv, kern=kern_dk)
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results                                   
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(dkplsqda_results,tmp_results[k,:])
    end  
end

### S family ---------------

# s-PLS-DA
function splsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict 
    mod = Jchemo.splsrda(; nlv=nlv, meth=meth_s)
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results                                        
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(splsrda_results,tmp_results[k,:])
    end  
end

# s-PLS-LDA
function splslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.splslda(; nlv=nlv, meth=meth_s)
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(splslda_results,tmp_results[k,:])
    end  
end

# s-PLS-QDA
function splsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict 
    mod = Jchemo.splsqda(; nlv=nlv, meth=meth_s)
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(splsqda_results,tmp_results[k,:])
    end  
end

### KDE family ---------------

# PLS-KDE-DA
function plskdeda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict 
    mod = Jchemo.plskdeda(; nlv=nlv)
    Jchemo.fit!(mod, Xtrain, ytrain)
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(plskdeda_results,tmp_results[k,:])
    end    
end

# s-PLS-KDE-DA
function splskdeda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.splskdeda(; nlv=nlv, meth=meth_s)
    Jchemo.fit!(mod, Xtrain, ytrain) ;
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(splskdeda_results,tmp_results[k,:])
    end
end

### MB family ---------------

# MBPLSR-DA
function mbplsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # build blocks 
    Xbltrain = mblock(Xtrain, listbl_mb)
    Xbltest = mblock(Xtest, listbl_mb) 
    # model, fit and predict
    mod = Jchemo.mbplsrda(; nlv=nlv, bscal=bscal_mb, scal=scal_mb)
    Jchemo.fit!(mod, Xbltrain, ytrain)
    res = Jchemo.predict(mod, Xbltest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(mbplsrda_results,tmp_results[k,:])
    end
end

# MBPLS-LDA
function mbplslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # build blocks 
    Xbltrain = mblock(Xtrain, listbl_mb)
    Xbltest = mblock(Xtest, listbl_mb) 
    # model, fit and predict
    mod = Jchemo.mbplslda(; nlv=nlv, bscal=bscal_mb, scal=scal_mb)
    Jchemo.fit!(mod, Xbltrain, ytrain) ;
    res = Jchemo.predict(mod, Xbltest, nlv=1:nlv) ;
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(mbplslda_results,tmp_results[k,:])
    end
end

# MBPLS-QDA
function mbplsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # build blocks 
    Xbltrain = mblock(Xtrain, listbl_mb)
    Xbltest = mblock(Xtest, listbl_mb) 
    # model, fit and predict
    mod = Jchemo.mbplsqda(; nlv=nlv, bscal=bscal_mb, scal=scal_mb)
    Jchemo.fit!(mod, Xbltrain, ytrain)
    res = Jchemo.predict(mod, Xbltest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(mbplsqda_results,tmp_results[k,:])
    end
end

# LW family -----------------

# kNN-LWPLSR-DA
function lwplsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.lwplsrda(; nlvdis=nlvdis_lw, metric=metric_lw, h=h_lw, k=k_lw, nlv=nlv)
    Jchemo.fit!(mod, Xtrain, ytrain) ;
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(lwplsrda_results,tmp_results[k,:])
    end
end

# kNN-LWPLSR-LDA
function lwplslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.lwplslda(; nlvdis=nlvdis_lw, metric=metric_lw, h=h_lw, k=k_lw, nlv=nlv)
    Jchemo.fit!(mod, Xtrain, ytrain) ;
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(lwplslda_results,tmp_results[k,:])
    end
end

# kNN-LWPLSR-DA
function lwplsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    # train-test
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    # model, fit and predict
    mod = Jchemo.lwplsqda(; nlvdis=nlvdis_lw, metric=metric_lw, h=h_lw, k=k_lw, nlv=nlv)
    Jchemo.fit!(mod, Xtrain, ytrain) ;
    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    # store results
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(lwplsqda_results,tmp_results[k,:])
    end
end

# ======================================================================================

# Function to remove samples from dataset
function selectsamples(df, samplenames)
    issample = filter(row -> row.Sample ∈ samplenames, df)
    isnotsample = filter(row -> row.Sample ∉ samplenames, df)
    return issample, isnotsample
end

# set seed 
Random.seed!(2545658)

# CV function
#all_tasks = []
for i in ProgressBar(1:nrep)
    
    # sample procedure 
    set = [selectsamples(Geol_grouped[j], sample(unique(Geol_grouped[j].Sample), LnOCV; replace = false, ordered = true)) for j in 1:length(Geol_grouped)]
    calib_set = reduce(vcat, [set[j][2][sample(axes(set[j][2],1),train_set_size),:] for j in 1:length(set)], cols=:union )
    valid_set = reduce(vcat, [set[j][1][sample(axes(set[j][1],1),valid_set_size),:] for j in 1:length(set)], cols=:union )

    # set train and test 
    Xtrain = Matrix(select(calib_set, Not([:Sample, :Source, :index, :shot_number, :test_number])))
    ytrain = vec(Matrix(select(calib_set, :Source)))
    Xtest = Matrix(select(valid_set, Not([:Sample, :Source, :index, :shot_number, :test_number])))
    ytest = vec(Matrix(select(valid_set, :Source)))

    # names for testset 
    sample_name = vec(Matrix(select(valid_set, :Sample)))
    spectrum_name = vec(Matrix(select(valid_set, :index))) 
    
    # models
    tasks = [
        Threads.@spawn plsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn plslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)    
        Threads.@spawn plsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        #Threads.@spawn svmda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
        #Threads.@spawn pca_svmda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn dkplsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn dkplslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn dkplsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn splsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn splslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn splsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn plskdeda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn splskdeda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn 
        #Threads.@spawn lwplslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        #Threads.@spawn lwplsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        if multiblock
            Threads.@spawn mbplsrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
            Threads.@spawn mbplslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
            Threads.@spawn mbplsqda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        end
    ]

    # add the tasks
    #append!(all_tasks, tasks)

    # sincronize tasks 
    for task in tasks
        fetch(task)  # wait
    end
end


# ======================================================================================

# calculate metrices
# k is the index of the class in conf_matrix.y
function _metrics_class(conf_matrix::DataFrame, k::Int)
    # obtain classes and values
    numeric_values = Matrix(conf_matrix[:, 2:end])
    # compute for given claass
    TP = numeric_values[k, k]                     # True positives
    FP = sum(numeric_values[:, k]) - TP           # False positives
    FN = sum(numeric_values[k, :]) - TP           # False negatives 
    TN = sum(numeric_values) - (TP + FP + FN)     # True negatives
    # metrices 
    PPV = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)
    NPV = TN / (TN + FN)
    Specificity = TN / (TN + FP)
    # Store results 
    return [PPV, Sensitivity, NPV, Specificity]
end

# accuracy metrices
function _metrics_model(conf_matrix::DataFrame)
    # numeric matrix
    numeric_values = Matrix(conf_matrix[:, 2:end])
    # values
    p_c = sum(diag(numeric_values))         # total number of elements correctly predicted
    p_s = sum(numeric_values)               # total number of elements
    row_sums = sum(numeric_values, dims=2)  # number of times that class k truly occurred (row total)
    col_sums = sum(numeric_values, dims=1)  # number of times that class k was predicted (column total)
    # sum of product 
    Sum = sum(row_sums' .* col_sums)
    # compute 
    overall = sum(diag(numeric_values)) / sum(numeric_values)    
    kappa = (p_c*p_s - Sum) / (p_s^2 - Sum)
    mmc = (p_c*p_s - Sum) / sqrt( (p_s^2 - sum(row_sums.^2)) * (p_s^2 - sum(col_sums.^2)) )
    # Store results 
    return [overall, kappa, mmc]
end

# proportion of well classified by sample 
function _metrics_samples(Model_str::String, Model_results::DataFrame, column::String)
    # loop till samples 
    for Sample in unique(Model_results[:,2])
        # subset
        Subset = Model_results[isequal.(Model_results.Sample, Sample), :]
        # obtain confussion matrix
        conf_matrix = conf(vec(Matrix(select(Subset, Symbol(column)))), vec(Matrix(select(Subset, :Group))))
        # numeric matrix
        numeric_values = Matrix(conf_matrix.cnt[:, 2:end])
        overall = sum(diag(numeric_values)) / sum(numeric_values)    
        # push
        push!(results_samples, [processing, Model_str, column, Sample, overall])
    end
end

# build results by run (model metrics)
function _metrics_run(Model_str::String, Model_results::DataFrame, column::String)
    # loop tin runs
    for run in 1:nrep
        # subset
        Subset = Model_results[isequal.(Model_results.RunInfo, run), :]
        # obtain confussion matrix
        conf_matrix = conf(vec(Matrix(select(Subset, Symbol(column)))), vec(Matrix(select(Subset, :Group))))
        # model metrics
        modelMet = _metrics_model(conf_matrix.cnt)
        push!(results_run, [[processing, Model_str, column, run]; modelMet])
    end
end 

# function to export the confussion matrix in csv and svg  
function _conf_outputs(Model_str::String, conf_matrix)
    # set results name
    fileName = processing * "/" * Model_str
    # export csv
    CSV.write(fileName * "_conf_matrix.csv", conf_matrix.cnt)
    # plot 
    Plot = plotconf(conf_matrix, size = (800, 700), fontsize=30)
    save(fileName * "_conf_matrix_plot.svg", Plot.f)
end

# buA_peaksild results dataframes and generate outputs 
function conf_metrices(Model_str::String, Model_results::DataFrame, Columns::Vector)
    for col in Columns
        # obtain confussion matrix
        conf_matrix = conf(vec(Matrix(select(Model_results, Symbol(col)))), vec(Matrix(select(Model_results, :Group))))
        # export outputs 
        _conf_outputs(Model_str * "_" * col, conf_matrix)
        # model metrics
        modelMet = _metrics_model(conf_matrix.cnt)
        push!(results_models, [[processing, Model_str, col]; modelMet])
        # class metrics
        classes = conf_matrix.cnt.y
        n_classes = size(conf_matrix.cnt, 1)
        for k in 1:n_classes
            classMet = _metrics_class(conf_matrix.cnt, k)
            push!(results_class, [[processing, Model_str, col, classes[k]]; classMet])
        end
        # sample metrics (aldready push in the function)
        _metrics_samples(Model_str, Model_results, col)
        # metrics by run
        _metrics_run(Model_str, Model_results, col)
    end
end 

# model results dataframe 
results_models = DataFrame(Matrix{Any}(undef, 0, 6), :auto)
rename!(results_models, ["processing", "Model", "lv", "Overall", "Kappa", "MMC"])

# class results dataframe 
results_class = DataFrame(Matrix{Any}(undef, 0, 8), :auto)
rename!(results_class, ["processing", "Model", "lv", "Class", "PPV", "Sensitivity", "NPV", "Specificity"])

# sample results dataframe
results_samples = DataFrame(Matrix{Any}(undef, 0, 5), :auto)
rename!(results_samples, ["processing", "Model", "lv", "Sample", "Overall"])

# model results dataframe 
results_run = DataFrame(Matrix{Any}(undef, 0, 7), :auto)
rename!(results_run, ["processing", "Model", "lv", "run", "Overall", "Kappa", "MMC"])

# execute outputs
conf_metrices("plsrda", plsrda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("plslda", plslda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("plsqda", plsqda_results, ["LV_pred_$lv" for lv in 1:nlv])
#conf_metrices("svmda", svmda_results, ["Prediction"])
#conf_metrices("pca_svmda", pca_svmda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("dkplsrda", dkplsrda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("dkplslda", dkplslda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("dkplsqda", dkplsqda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("splsrda", splsrda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("splslda", splslda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("splsqda", splsqda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("plskdeda", plskdeda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("splskdeda", splskdeda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("lwplsrda", lwplsrda_results, ["LV_pred_$lv" for lv in 1:nlv])
#conf_metrices("lwplslda", lwplslda_results, ["LV_pred_$lv" for lv in 1:nlv])
#conf_metrices("lwplsqda", lwplsqda_results, ["LV_pred_$lv" for lv in 1:nlv])
if multiblock
    conf_metrices("mbplsrda", mbplsrda_results, ["LV_pred_$lv" for lv in 1:nlv])
    conf_metrices("mbplslda", mbplslda_results, ["LV_pred_$lv" for lv in 1:nlv])
    conf_metrices("mbplsqda", mbplsqda_results, ["LV_pred_$lv" for lv in 1:nlv])
end

# subset
#results_models[isequal.(results_models.lv, "LV_pred_20"), :] 

# save the dfs
CSV.write(processing * "/results_models.csv", results_models)
CSV.write(processing * "/results_class.csv", results_class)
CSV.write(processing * "/results_samples.csv", results_samples)
CSV.write(processing * "/results_run.csv", results_run)

