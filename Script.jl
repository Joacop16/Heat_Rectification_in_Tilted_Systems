include("TN_Superfermionic_Mesoscopic_Leads/Code/Interacting_functions_wGPU.jl")

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
E = 4
μ = -E*(D+1)/4

ε_system = [(μ+0.5E*j) for j =1:D]; #Tilted system energies

M = L + D + R
sites = siteinds("S=1/2",2*M)

############################### First run #########################################

GPU = false 

I_vec = Build_left_vacuum(sites);
Swap_Gates, TEBD_Gates = Build_Gates(sites, εk, γk, 0*κp, fk_L, εk, γk, 0*κp, fk_R, ε_system, ts, U); #0*Kp_array = no coupling between the lead and the system

NumSteps = 5
@time Thermal_State, Occupation_Matrix_t = Apply_TEBD(I_vec, I_vec, Swap_Gates, TEBD_Gates, NumSteps); #maxdim = 40 by default
@show JP = Particle_Current(Thermal_State, I_vec, γk, fk_L, Float64[], Float64[])
@show JE = Energy_Current(Thermal_State, I_vec, γk, fk_L, εk, κp, Float64[], Float64[], Float64[], ComplexF64[]);

GPU = true 

I_vec = Build_left_vacuum(sites);
Swap_Gates, TEBD_Gates = Build_Gates(sites, εk, γk, κp, fk_L, εk, γk, κp, fk_R, ε_system, ts, U); #0*Kp_array = no coupling between the lead and the system

@time Thermal_State, Occupation_Matrix_t = Apply_TEBD(I_vec, I_vec, Swap_Gates, TEBD_Gates, NumSteps); #maxdim = 40 by default
@show JP = Particle_Current(Thermal_State, I_vec, γk, fk_L, Float64[], Float64[])
@show JE = Energy_Current(Thermal_State, I_vec, γk, fk_L, εk, κp, Float64[], Float64[], Float64[], ComplexF64[]);

############################### Real run #########################################

function Apply_TEBD_MODIFIED(Psi_0::MPS, I_vec::MPS, Swap_Gates::Vector{ITensor}, TEBD_Gates::Vector{ITensor}, NumSteps::Int64, maxdim::Int64 = 40, file_path::String = "") 

    cutoff = 1E-15 #Should I include cutoff as optional parameter?
    
    length(Swap_Gates) == 0 ? L = 0 : L = length(Swap_Gates) + 1

    Occupation_Matrix_t = zeros(100 + 1, Int(length(Psi_0)/2))
    JE_t = []
    times_array = [0.0]
    
    Psi_t = Psi_0
    norm_t = inner(I_vec, Psi_t)
    Psi_t = Psi_t/norm_t
    
    Occupation_Matrix_t[1,:] = Occupations_per_site(Psi_t, I_vec)
    append!(JE_t, Energy_Current(Psi_t, I_vec, γk, fk_L, εk, κp, Float64[], Float64[], Float64[], ComplexF64[])) #USES GLOBAL VARIABLES
    
    
    Reverse_SWAP_Gates = reverse(Swap_Gates)

    for i = 1:NumSteps 
        
        L != 0 ? orthogonalize!(Psi_t,site_SF(L) + 1) : nothing #orthonormalization center between the final site of the lead and the first site of the system
        L != 0 ? Psi_t = apply(Swap_Gates, Psi_t; cutoff=cutoff, maxdim=maxdim) : nothing #moving system and his ancilla to the second physical site i.e. Lead1,lead1',S1,S1',...,leadL, leadL', S2,S2'..., SD,SD',Lead1,Lead1',...,LeadR,LeadR'. maxdim = 1, SWAPS should no shange bond dimension.
        
        orthogonalize!(Psi_t,1) #make all the lattice right normalized
        Psi_t = apply(TEBD_Gates, Psi_t; cutoff=cutoff, maxdim=maxdim)
        
        L != 0 ? Psi_t = apply(Reverse_SWAP_Gates, Psi_t; cutoff=cutoff, maxdim=maxdim) : nothing #Moving system and his ancilla to his usual position i.e. Lead1,lead1', ...,leadL,leadL',S1,S1',...,SD,SD',Lead1,Lead1',...,LeadR,LeadR'. maxdim = 1, SWAPS should no shange bond dimension.
        
        norm_t = inner(I_vec, Psi_t)
        Psi_t = Psi_t/norm_t 

        GPU ? CUDA.reclaim() : nothing #GPU Modification

        if (i*100/NumSteps)%1 == 0.0
            row = Int((i*100/NumSteps)) + 1
            # @time Occupation_Matrix_t[row,:] = Occupations_per_site(Psi_t, I_vec)
            append!(JE_t, Energy_Current(Psi_t, I_vec, γk, fk_L, εk, κp, Float64[], Float64[], Float64[], ComplexF64[])) #USES GLOBAL VARIABLES
            append!(times_array, dt*i) #USES GLOBAL VARIABLES
            CUDA.memory_status()
        end
        
        if (i*100/NumSteps)%10 == 0.0
            println(string((i*100/NumSteps))*" % Completed.")

            if file_path != ""

                GPU ? Psi_cpu = NDTensors.cpu(Psi_t) : Psi_cpu = Psi_t #We need to move the MPS back to the CPU to save it into a .h5 file.
                f = h5open(file_path*"_NESS_MPS.h5","w")
                write(f,"MPS",Psi_cpu)
                close(f)        
                
                writedlm(file_path*"_NESS_Occupations.txt", Occupation_Matrix_t)        
            end
        end
    end
    return Psi_t, Occupation_Matrix_t, JE_t, times_array

end

GPU = false

sites = siteinds("S=1/2",2*M)
I_vec = Build_left_vacuum(sites);

Swap_Gates, TEBD_Gates = Build_Gates(sites, εk, γk, 0*κp, fk_L, εk, γk, 0*κp, fk_R, ε_system, ts, U); #0*Kp_array = no coupling between the lead and the system

NumSteps = 1000
@time Thermal_State, Occupation_Matrix_t = Apply_TEBD(I_vec, I_vec, Swap_Gates, TEBD_Gates, NumSteps); #maxdim = 40 by default
@show JP = Particle_Current(Thermal_State, I_vec, γk, fk_L, Float64[], Float64[])
@show JE = Energy_Current(Thermal_State, I_vec, γk, fk_L, εk, κp, Float64[], Float64[], Float64[], ComplexF64[]);

#TN Parameters
maxdim = 80;
dt = 0.05

GPU = true

Thermal_State = gpu(Thermal_State)
I_vec = gpu(I_vec)

Swap_Gates, TEBD_Gates = Build_Gates(sites, εk, γk, κp, fk_L, εk, γk, κp, fk_R, ε_system, ts, U, dt);

#TN Parameters
NumSteps = 100
@time NESS, Occupation_Matrix_t, JE_t, times_array = Apply_TEBD_MODIFIED(Thermal_State, I_vec, Swap_Gates, TEBD_Gates, NumSteps, maxdim, "/jet/home/penuelap/MPS_for_gpu_testing/Heat_rectification/"*ARGS[1]); #maxdim = 40 by default