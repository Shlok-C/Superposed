using HomotopyContinuation
using QuantumToolbox
using CairoMakie

function solve_critical_states()
    @var λ₀, λ₁, μ

    #=

    Cartan Subspace matrix
    |ψ> = λ₀|00> + λ₁|11>
    [  λ₀ 0  ]
    [  0  λ₁ ]

    det^2(ψ) = (λ₀λ₁)^2

    ∇[det(ψ)] = μ * ∇[||ψ||^2]
    
    ||ψ||^2 = λ₁^2 + λ₂^2 = 1
    
    =#

    norm = λ₀^2 + λ₁^2 - 1
    det2 = (λ₀ * λ₁)^2

    vars = [λ₀, λ₁]

    grad_det = [differentiate(det2, v) for v in vars]
    grad_norm = [differentiate(norm, v) for v in vars]

    lagrange_equations = [
        grad_det[i] - μ * grad_norm[i] for i in 1:2
    ] 

    println(grad_det)
    println(grad_norm)

    system = [
        lagrange_equations...,
        norm
    ]

    result = solve(system)

    real_sols = real_solutions(result, tol=1e-1)

    println("Found $(length(real_sols)) real solutions")

    for (i, sol) in enumerate(real_sols)
        println("Solution $i: $sol")
        
        det = (sol[1] * sol[2])^2
        println("|Det(ψ)|^2 = $det\n")
    end

    return real_sols
end

function plot_states()

end

res = solve_critical_states()

println(res)
