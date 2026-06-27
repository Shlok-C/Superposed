using QuantumOptics
using Plots

plotly()

# 1. Spin-1/2 basis (i.e., a qubit)
b = SpinBasis(1//2)

# 2. Basis states |0⟩ and |1⟩
ψ0 = basisstate(b, 1)   # m = +1/2
ψ1 = basisstate(b, 2)   # m = -1/2

# 3. Example state: (|0⟩ + e^{iπ/3}|1⟩)/√2
psi = normalize(ψ0 + exp(im*pi/3)*ψ1)

# 4. Pauli operators
sx = sigmax(b)
sy = sigmay(b)
sz = sigmaz(b)

# 5. Expectation values → Bloch coordinates
x = real(expect(sx, psi))
y = real(expect(sy, psi))
z = real(expect(sz, psi))

# 6. Draw a Bloch sphere and your point
θ = range(0, stop=π, length=80)
φ = range(0, stop=2π, length=80)
X = [sin(t)*cos(p) for t ∈ θ, p ∈ φ]
Y = [sin(t)*sin(p) for t ∈ θ, p ∈ φ]
Z = [cos(t)        for t ∈ θ, p ∈ φ]

surface(X, Y, Z, color=:lightblue, alpha=0.3, legend=false)
scatter!([x], [y], [z], markersize=8)
