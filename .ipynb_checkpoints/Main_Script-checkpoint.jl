#First Run

GitHub_repository_path = "/ihome/jmendoza-arenas/jop204/Heat_Rectification_in_Tilted_Systems/TN_Superfermionic_Mesoscopic_Leads"
include(GitHub_repository_path*"/Code/Interacting_functions.jl")

precompile_package(GitHub_repository_path, false) #GPU = false
precompile_package(GitHub_repository_path, true) #GPU = true

using LinearAlgebra
using FlexiMaps

# W = 8.0 #8 for D = 4, 14 for D = 6 and 18 for D = 8

D = parse(Int64,ARGS[2])

D == 4 ? W = 8.0 : nothing
D == 6 ? W = 14.0 : nothing
D == 8 ? W = 18.0 : nothing

@show W

Γ = 1.0
J(ω) = abs(ω) <= abs(W) ? Γ : 0 #Spectral density Γ = 1.0 in this case

function Logarithmic_linear_arrays(L, J)
    #L: lead size
    #J: Spectral density    
    
    #We need to define εk for the lead. In particular, we use the logarithmic-linear discretization of the paper. As this is just for the article, it is not worthy to put it with the other functions in NonInteacting_functions.jl
    
    W_log = W #W
    W_lin = W_log/2 #W* = W/2
    
    L_log = L*0.2
    L_lin = L - L_log
    
    Lin_Window = LinRange(-W_lin, W_lin, Int(L_lin))
    Log_Window = maprange(log, W_lin, W_log, length=Int(L_log/2 + 1))
    
    εk = Float64[]
    append!(εk, -reverse( Log_Window[2:length(Log_Window)]), Lin_Window, Log_Window[2:length(Log_Window)])
    
    γk = [εk[k+1] - εk[k] for k=1:Int(L/2)]
    append!(γk, reverse(γk))

    κkp = ComplexF64[sqrt(J(εk[k])*γk[k]/(2.0*pi)) for k=1:L] #Kp = sqrt(J(e)*γ/(2*pi))

    return εk, γk,  κkp
end

#Global variables by default
μ_L, μ_R = 0.0, 0.0 #As we want to focus in Heat rectification, we should not have gradient of chemical potential.
ts = 1.0

function fk_arrays(εk_array_L, εk_array_R)
    
    fk_array_L = [fermi_dirac_distribution(ε, μ_L, β_L) for ε = εk_array_L]
    fk_array_R = [fermi_dirac_distribution(ε, μ_R, β_R) for ε = εk_array_R]
    
    return fk_array_L, fk_array_R
end

if ARGS[1] == "Forward" 
    β_L, β_R = 1/(1000*ts), 1/(1*ts) #Forward Bias: TL > TR
else
    β_L, β_R = 1/(1*ts), 1/(1000*ts) #Reverse Bias: TL < TR
end

#Lead Parameters

D == 4 ? L = 10 : nothing
D == 6 ? W = 12 : nothing
D == 8 ? W = 14 : nothing
R = L

εk, γk, κp = Logarithmic_linear_arrays(L, J)
fk_L, fk_R = fk_arrays(εk, εk);

#System Parameters
U = 10.0*ts
E = parse(Float64,ARGS[3])
μ = -E*(D+1)/4

ε_system = [(μ+0.5E*j) for j =1:D]; #Tilted system energies

#TN Parameters
M = L + D + R
sites = siteinds("S=1/2",2*M)

dt = 0.05
maxdim = 100
cutoff = 10e-12
Measurements = ["JP", "JE"]

Percentage_for_measurement = 1.0
Params = Params_for_Measurements(Percentage_for_measurement, εk, γk, κp, fk_L, εk, γk, κp, fk_R, ε_system, ts, U, dt);

PRINT = true
GPU = false #Thermal state is faster without GPU

I_vec = Build_left_vacuum(sites);
Swap_Gates, TEBD_Gates = Build_Gates(sites, εk, γk, 0*κp, fk_L, εk, γk, 0*κp, fk_R, ε_system, ts, U); #0*Kp_array = no coupling between the lead and the system

NumSteps = 1000
@time Thermal_State, observables = Apply_TEBD(I_vec, I_vec, Swap_Gates, TEBD_Gates, NumSteps, maxdim, cutoff, Measurements, Params); 
println("Thermal State calculated:")
@show observables.JP_t[end]
@show observables.JE_t[end]

#Now, let's start the evolution from this Thermal State. This is faster with GPU
GPU = true
Thermal_State = gpu(Thermal_State)
I_vec = gpu(I_vec)

Swap_Gates, TEBD_Gates = Build_Gates(sites, εk, γk, κp, fk_L, εk, γk, κp, fk_R, ε_system, ts, U, dt);

#TN Parameters
NumSteps = 2000

# Folder = "/jet/home/penuelap/Heat_rectification_Data/" #PSC
# Folder = "Local_Data/" #Local PC
Folder = "/ihome/jmendoza-arenas/jop204/Heat_Rectification_Data/" #CRC

Name = ARGS[1]*"_E=$E"*"_L=$L"*"_D=$D"

@time NESS, observables = Apply_TEBD(Thermal_State, I_vec, Swap_Gates, TEBD_Gates, NumSteps, maxdim, cutoff, Measurements, Params, Folder*Name); #maxdim = 40 by default

println("Name Completed. JE = $(observables.JE_t[end]) and JP = $(observables.JP_t[end])")