using DifferentialEquations, Plots, StatsBase, LinearAlgebra, Laplacians

# Parameters
N = Int64(1e4)
mₚ = 1
mₛ = 1
k = 2 * mₚ / N
K = Diagonal(fill(k, N + 1))
k_bT = 1
var = k_bT / mₚ
M_inv = Diagonal(1 ./ vcat([mₚ], fill(mₛ, N)))
L = √(K) * Laplacians.star_graph(N + 1) * √(K)

# Initial conditions
x₀ = zeros(N + 1)
dx₀ = rand(N + 1)

tspan = (0., 10.)

function n_oscillators(ddx, dx, x, t)
    ddx .= M_inv * L * x
end

# Solve
prob = DynamicalODEProblem(n_oscillators, dx₀, x₀, tspan)
sol = solve(prob, VelocityVerlet())

# Plot
plot(sol, vars=[2,1], linewidth=2, title ="Simple Harmonic Oscillator", xaxis = "Time", yaxis = "Elongation", label = ["x" "dx"])
gui()
