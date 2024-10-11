include("TN_Superfermionic_Mesoscopic_Leads/Code/Interacting_functions.jl")

using LinearAlgebra
using FlexiMaps

W = 8.0
Γ = W/8
J(ω) = abs(ω) <= abs(W) ? Γ : 0 #Spectral density Γ = W/8 in this case

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
μ_L, μ_R = W/16, W/16 #As we want to focus in Heat rectification, we should not have gradient of chemical potential.
ts = (W/8)

function fk_arrays(εk_array_L, εk_array_R)
    
    fk_array_L = [fermi_dirac_distribution(ε, μ_L, β_L) for ε = εk_array_L]
    fk_array_R = [fermi_dirac_distribution(ε, μ_R, β_R) for ε = εk_array_R]
    
    return fk_array_L, fk_array_R
end

if ARGS[1] == "Forward" 
    β_L, β_R = 1/(10*ts), 1/(2*ts) #Forward Bias: TL > TR
else
    β_L, β_R = 1/(2*ts), 1/(10*ts) #Reverse Bias: TL < TR
end

#Lead Parameters
L, R = 10, 10
εk, γk, κp = Logarithmic_linear_arrays(L, J)
fk_L, fk_R = fk_arrays(εk, εk);

#System Parameters
D = 2
U = 1.2*ts
E = parse(Float64,ARGS[2])
μ = -E*(D+1)/4

ε_system = [(μ+0.5E*j) for j =1:D]; #Tilted system energies

#TN Parameters
M = L + D + R
sites = siteinds("S=1/2",2*M)

dt = 0.05
maxdim = 80
cutoff = 10e-12
Measurements = ["JP", "JE"]

Percentage_for_measurement = 1.0
Params = Params_for_Measurements(Percentage_for_measurement, εk, γk, κp, fk_L, εk, γk, κp, fk_R, ε_system, ts, U, dt);

#------------------------------------------------------------------ First run ------------------------------------------------------------------ 
PRINT = false

for GPU_value = [true, false]
    global GPU = GPU_value

    Measurements = ["Occupation", "JP", "JE"]
    NumSteps = 5    
    maxdim = 40
    cutoff = 10e-12
    
    I_vec = Build_left_vacuum(sites);
    Swap_Gates, TEBD_Gates = Build_Gates(sites, εk, γk, 0*κp, fk_L, εk, γk, 0*κp, fk_R, ε_system, ts, U); #0*κp no coupling between the lead and the system
    Psi_t, observables = Apply_TEBD(I_vec, I_vec, TEBD_Gates, NumSteps, maxdim, cutoff, Measurements, Params); #maxdim = 40 by default
end

println("First run completed.")
#------------------------------------------------------------------ Real code ------------------------------------------------------------------  
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
NumSteps = 500

Folder = "/jet/home/penuelap/Heat_rectification_Data/" #PSC
# Folder = "Local_Data/" #Local PC

Name = ARGS[1]*"_E_$E"*"L_$L"

@time NESS, observables = Apply_TEBD(Thermal_State, I_vec, Swap_Gates, TEBD_Gates, NumSteps, maxdim, cutoff, Measurements, Params, Folder*Name); #maxdim = 40 by default