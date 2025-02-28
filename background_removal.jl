using SparseArrays, LinearAlgebra

#Implementation in Julia of the AirPLS algorithm by Zhang et al., 2010 

function WhittakerSmooth(x, w, lambda_::Int64, differences::Int64)
    """
    Penalized least squares algorithm for background fitting
    """
    X = x[:]
    m = length(X)
    E = spdiagm(0 => ones(m))
    
    for i in 1:differences
        E = E[2:end, :] - E[1:end-1, :]
    end

    W = spdiagm(0 => w)
    A = W + lambda_ * (E' * E)
    B = W * X
    background = A \ B
    return collect(background)
end

function airPLS(x, lambda_::Int64=100, porder::Int64=1, itermax::Int64=100)
    """
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    """
    m = length(x)
    w = ones(m)
    local z
    #print(z)
    local d

    for i in 1:itermax
        #print(i)
        z = WhittakerSmooth(x, w, lambda_, porder)
        #print(z)
        d = x - z
        dssn = sum(abs.(d[d .< 0]))
        
        if dssn < 0.001 * sum(abs.(x)) || i == itermax
            if i == itermax
                println("WARNING: max iteration reached!")
            end
            break
        end

        w[d .>= 0] .= 0
        w[d .< 0] .= exp.(i .* abs.(d[d .< 0]) ./ dssn)
        w[1] = exp(i * maximum(d[d .< 0]) / dssn)
        w[end] = w[1]
    end

    return z
end
