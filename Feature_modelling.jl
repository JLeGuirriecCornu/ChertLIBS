using CSV
using DataFrames
using StatsBase
using Random
using Jchemo, CairoMakie, FreqTables
using ProgressBars
using LinearAlgebra
using ThreadsX
using ArgParse
using DecisionTree


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
        "--verbose", "-v"
            help = "show prints? (default false)"
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
const verbose::Bool =  parsed_args["verbose"]


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

#File = "/home/spegeochert/Data/LIBS/LIBS_analysis_code/Analysis_Pipeline_article/LIBS_analysis_testing/dataframes/Peaks/article_peaks_AirPLS_area"

# Data
Geol_data = CSV.read(File, DataFrame, delim = ',')

# exclude the first two shots 
Geol_data = Geol_data[.!(x -> x in [1, 2, 3, 4, 5, 6, 7, 8]).(Geol_data.shot_number), :]

# Renaming collumns
rename!(Geol_data, :Nom => :Sample)
rename!(Geol_data, :Formation => :Source)
select!(Geol_data, Not(:Column1))
#filter!(row -> row.Sample != "OBS_MAOA-XRF", Geol_data) 

# Variables for later
Sources = sort(unique(Geol_data[!,:Source]))
nSources = length(Sources)

Geol_data[:,6:end] = DataFrame([zscore(Geol_data[!,col]) for col in names(Geol_data[:,6:end])], names(Geol_data[:,6:end]))

# Create set of equal size for each group
Geol_grouped = groupby(Geol_data, :Source)

# output directory (needs try-except)
mkdir(processing)

nlv = 10

# ======================================================================================
# settings 

# Eperiment parameterswr
const nrep::Int = 200
const train_set_size::Int = 3000
const valid_set_size::Int = 100
const LnOCV::Int = 1

# svm 
kern_svm = :kpol
gamma_svm::Float64 = 1/27
cost_svm::Int = 10
epsilon_svm::Float64 = 0.5;

# SVM
svmda_results = DataFrame(Matrix{Any}(undef, 0, (5)), :auto)
rename!(svmda_results, ["RunInfo", "Sample","Spectrum", "Group", "Prediction"])

#RandomForest

rf_results = DataFrame(Matrix{Any}(undef, 0, (5+nSources)), :auto)
rename!(rf_results, [["RunInfo", "Sample","Spectrum", "Group", "Prediction"];["Proba_$source" for source in 1:nSources]])

rf_var_importance = DataFrame(Matrix{Any}(undef, 0, (size(Geol_data)[2]-4)), :auto)
rename!(rf_var_importance, ["RunInfo";names(Geol_data)[6:end]])

# DecisionTree

dtc_results = DataFrame(Matrix{Any}(undef, 0, (5+nSources)), :auto)
rename!(dtc_results, [["RunInfo", "Sample","Spectrum", "Group", "Prediction"];["Proba_$source" for source in 1:nSources]])

dtc_var_importance = DataFrame(Matrix{Any}(undef, 0, (size(Geol_data)[2]-4)), :auto)
rename!(dtc_var_importance, ["RunInfo";names(Geol_data)[6:end]])


#ADA BOOST
ada_results = DataFrame(Matrix{Any}(undef, 0, (5+nSources)), :auto)
rename!(ada_results, [["RunInfo", "Sample","Spectrum", "Group", "Prediction"];["Proba_$source" for source in 1:nSources]])

ada_var_importance = DataFrame(Matrix{Any}(undef, 0, (size(Geol_data)[2]-4)), :auto)
rename!(ada_var_importance, ["RunInfo";names(Geol_data)[6:end]])

#RDA
rda_results = DataFrame(Matrix{Any}(undef, 0, (5)), :auto)
rename!(rda_results, ["RunInfo", "Sample","Spectrum", "Group", "Prediction"])


# RRDA
rrda_results = DataFrame(Matrix{Any}(undef, 0, (5)), :auto)
rename!(rrda_results, ["RunInfo", "Sample","Spectrum", "Group", "Prediction"])

# PLS-DA
plsrda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)                                    # nSources*nlv
rename!(plsrda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]]) # ["LV$lv$class" for lv in 1:nlv for class in Sources]

# PLS-LDA
plslda_results = DataFrame(Matrix{Any}(undef, 0, (4+nlv)), :auto)
rename!(plslda_results, [["RunInfo", "Sample","Spectrum", "Group"];["LV_pred_$lv" for lv in 1:nlv]])

#QDA
qda_results = DataFrame(Matrix{Any}(undef, 0, (5)), :auto)
rename!(qda_results, ["RunInfo", "Sample","Spectrum", "Group", "Prediction"])


#LDA
lda_results = DataFrame(Matrix{Any}(undef, 0, (5)), :auto)
rename!(lda_results, ["RunInfo", "Sample","Spectrum", "Group", "Prediction"])






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

function rf_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, n_trees=200, max_depth=-1, min_samples_leaf=1, min_samples_split=2)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = DecisionTree.RandomForestClassifier(n_trees = n_trees, max_depth=max_depth, min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split, impurity_importance=true)

    DecisionTree.fit!(mod, Xtrain, ytrain)

    MDI = DecisionTree.impurity_importance(mod)
    push!(rf_var_importance, [test_info;MDI])
    res = DecisionTree.predict(mod, Xtest)
    proba = DecisionTree.predict_proba(mod, Xtest)

    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res), hcat(proba))
    for k in 1:ntest
        push!(rf_results,tmp_results[k,:])
    end

end


function adaboost_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, n_iterations=200, rng=Random.GLOBAL_RNG)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = DecisionTree.AdaBoostStumpClassifier(n_iterations=n_iterations, rng=rng)

    DecisionTree.fit!(mod, Xtrain, ytrain)

    MDI = DecisionTree.impurity_importance(mod)
    push!(ada_var_importance, [test_info;MDI])
    res = DecisionTree.predict(mod, Xtest)
    proba = DecisionTree.predict_proba(mod, Xtest)

    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res), hcat(proba))
    for k in 1:ntest
        push!(ada_results,tmp_results[k,:])
    end

end

function dtc_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, purity_threshold = 0, max_depth=-1, min_samples_leaf=1, min_samples_split=2)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = DecisionTree.DecisionTreeClassifier(; pruning_purity_threshold=purity_threshold, max_depth=max_depth, min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split, impurity_importance=true)

    DecisionTree.fit!(mod, Xtrain, ytrain)

    MDI = DecisionTree.impurity_importance(mod)
    push!(dtc_var_importance, [test_info;MDI])
    res = DecisionTree.predict(mod, Xtest)
    proba = DecisionTree.predict_proba(mod, Xtest)

    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res), hcat(proba))
    for k in 1:ntest
        push!(dtc_results,tmp_results[k,:])
    end

end

function rda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = Jchemo.rda()

    Jchemo.fit!(mod, Xtrain, ytrain)

    res = Jchemo.predict(mod, Xtest)
    

    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred))
    
    for k in 1:ntest
        push!(rda_results,tmp_results[k,:])
    end

end

function rrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = Jchemo.rrda()

    Jchemo.fit!(mod, Xtrain, ytrain)

    res = Jchemo.predict(mod, Xtest)
    
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred))
    
    for k in 1:ntest
        push!(rrda_results,tmp_results[k,:])
    end

end

function plsda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = Jchemo.plsrda(; nlv)

    Jchemo.fit!(mod, Xtrain, ytrain)

    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
     
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(plsrda_results,tmp_results[k,:])
    end  

end

function plslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info, nlv)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = Jchemo.plslda(; nlv)

    Jchemo.fit!(mod, Xtrain, ytrain)

    res = Jchemo.predict(mod, Xtest, nlv=1:nlv)
    
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred...))
    for k in 1:ntest
        push!(plslda_results,tmp_results[k,:])
    end  

end

function lda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = Jchemo.lda()

    Jchemo.fit!(mod, Xtrain, ytrain)

    res = Jchemo.predict(mod, Xtest)
    
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred))
    
    for k in 1:ntest
        push!(lda_results,tmp_results[k,:])
    end

end

function qda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, test_info)
    ntrain = nro(Xtrain)
    ntest = nro(Xtest)
    mod = Jchemo.qda()

    Jchemo.fit!(mod, Xtrain, ytrain)

    res = Jchemo.predict(mod, Xtest)
    
    tmp_results = hcat(repeat([test_info],ntest), sample_name, spectrum_name, vec(ytest),hcat(res.pred))
    
    for k in 1:ntest
        push!(qda_results,tmp_results[k,:])
    end

end

#Function to remove samples from dataset
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
        Threads.@spawn svmda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
        
        Threads.@spawn rf_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
        #Threads.@spawn dtc_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
        Threads.@spawn adaboost_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)

        Threads.@spawn plsda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)
        Threads.@spawn plslda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i, nlv)

        #Threads.@spawn rda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
        Threads.@spawn rrda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
        Threads.@spawn lda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
        Threads.@spawn qda_fun(Xtrain, ytrain, Xtest, ytest, sample_name, spectrum_name, i)
    ]

    # add the tasks
    #append!(all_tasks, tasks)

    # sincronize tasks 
    for task in tasks
        fetch(task)  # wait
    end
end


#Function to test normality

function LinDep(A::Array, threshold1::Float64 = 1e-6, threshold2::Float64 = 1e-1; eigvec_output::Bool = false)
    (L, Q) = LinearAlgebra.eigen(A'*A)
    max_L = maximum(broadcast(abs, L))
    conditions = max_L ./ broadcast(abs, L)
    max_C = maximum(conditions)
    println("Max Condition = $max_C")
    Collinear_Groups = []
    Tricky_EigVecs = []
    for (idx, lambda) in enumerate(L)
        if lambda < threshold1
            push!(Collinear_Groups, findall(broadcast(abs, Q[:,idx]) .> threshold2))
            push!(Tricky_EigVecs, Q[:,idx])
        end
    end
    if eigvec_output
        return (Collinear_Groups, Tricky_EigVecs)
    else
        return Collinear_Groups       
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
    p_c = Float64(sum(diag(numeric_values)))                 # total number of elements correctly predicted
    p_s = Float64(sum(numeric_values))                       # total number of elements
    row_sums = Matrix{Float64}(sum(numeric_values, dims=2))  # number of times that class k truly occurred (row total)
    col_sums = Matrix{Float64}(sum(numeric_values, dims=1))  # number of times that class k was predicted (column total)
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
#conf_metrices("plsqda", plsqda_results, ["LV_pred_$lv" for lv in 1:nlv])
conf_metrices("svmda", svmda_results, ["Prediction"])
conf_metrices("rfda", rf_results, ["Prediction"])
conf_metrices("adaboostda", ada_results, ["Prediction"])
conf_metrices("rrda", rrda_results, ["Prediction"])
conf_metrices("lda", lda_results, ["Prediction"])
conf_metrices("qda", qda_results, ["Prediction"])



# subset
#results_models[isequal.(results_models.lv, "LV_pred_20"), :] 

# save the dfsaks/article_peaks_AirPLS_SNV_area -p Peaks_AirPLS_SNV_area -v
CSV.write(processing * "/results_models.csv", results_models)
CSV.write(processing * "/results_class.csv", results_class)
CSV.write(processing * "/results_samples.csv", results_samples)
CSV.write(processing * "/results_run.csv", results_run)

CSV.write(processing * "/rf_var_importance.csv", rf_var_importance)
CSV.write(processing * "/adaboost_var_importance.csv", ada_var_importance)
