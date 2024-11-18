# Matrix representations for local operators:

σx = [0 1; 1 0]
σy = [0 -1im; 1im 0]
σz = [1 0; 0 -1]

Identity = [1 0; 0 1]; #or Using LinearAlgebra and Matrix{Float64}(I, 2, 2)

σplus = (σx +1im*σy)/2
σminus = adjoint(σplus);

#Note 1: These functions build all the operators multiplying ⊗ at the right. The basis and everything should follow this convention.

function Basis_Element_j(N, j)

function Enlarge_Matrix_site_j(N, j, matrix)
    # I⊗...⊗I⊗M⊗I...⊗I

    Identity = zeros(size(matrix)) #In case that Identity have not been defined globally before.
    for i=1:size(matrix)[1]; Identity[i,i] = 1; end  
    
    M = Identity
    j == 1 ? M = matrix : nothing
    
    for i=2:N 
        i == j ? M = kron(M, matrix) :  M = kron(M, Identity)        
    end

    return M
end

function Enlarge_Matrix_i_Matrix_j(N, i, j, matrix_i, matrix_j)
    # I⊗...⊗I⊗M_i⊗I...⊗I⊗M_j⊗I⊗I...⊗I

    Identity = zeros(size(matrix_i)) #In case that Identity have not been defined globally before.
    for i=1:size(matrix_i)[1]; Identity[i,i] = 1; end 

    M = Identity

    j == 1 ? M = matrix_j : nothing
    i == 1 ? M = matrix_i : nothing
 
    for k=2:N 
        if k == j
            M = kron(M, matrix_j)
        elseif k == i
            M = kron(M, matrix_i)
        else
            M = kron(M, Identity)        
        end
    end

    return M
end

function Build_Cdag_Matrix(N, j)
    # C†_j = (∑_i<j σz_i) σ-_j. JW transformation.

    if j == 1
        Matrix = σminus
    else
        Matrix = σz
    end

    for i=2:N 
        if i <= j-1
        Matrix = kron(Matrix, σz)
        elseif i == j
        Matrix = kron(Matrix, σminus)
        else
        Matrix = kron(Matrix, Identity)  
        end
    end 

    return Matrix

end

function Build_C_Matrix(N, j)
    # C_j = (∑_i<j σz_i) σ+_j. JW transformation.

    if j == 1
        Matrix = σplus
    else
        Matrix = σz
    end
    
    for i=2:N 
        if i <= j-1
        Matrix = kron(Matrix, σz)
        elseif i == j
        Matrix = kron(Matrix, σplus)
        else
        Matrix = kron(Matrix, Identity)  
        end
    end 

    return Matrix

end

function Build_C_dagi_Cj_Matrix(N, i, j)
    #Some JW strings cancel between them. C†_iC_j = σ-_i (Π_i<k<j σz_k) σ+_j

    if i == j
        return Enlarge_Matrix_site_j(N, j, σminus*σplus) #C†_iC_i = σ-_i σ+_i
    end       

    Matrix = Identity
    i == 1 ? Matrix = σminus : nothing
    j == 1 ? Matrix = σplus : nothing
 
    for k=2:N 

        if k == i
            Matrix = kron(Matrix, σminus)
        elseif k == j
            Matrix = kron(Matrix, σplus)
        elseif k > minimum([i,j]) && k < maximum([i,j])
            Matrix = kron(Matrix, σz)        
        else
            Matrix = kron(Matrix, Identity)      
        end
    end

    return Matrix
end

fermi_dirac_distribution(ε, μ, β) = 1/(exp((ε - μ)*β) + 1) #μ: chemical potential, \beta =1/T