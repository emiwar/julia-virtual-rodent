using ComponentArrays
using Flux
import CUDA

nt = (a=5, b=[10.0], c=[0.5, 0.2], d=(e=0.1, f=[0.2, 0.3]))

ca = ComponentArray(nt)

ca.c = [0.7, 0.8]

gpu_ca = Flux.gpu(ca);


ax, = getaxes(ca)

mat = ComponentArray(randn(7, 100, 100), (ax, FlatAxis(), FlatAxis()))

@CUDA.time mat_gpu = Flux.gpu(mat);

@btime view($mat, :, 1, 1).d.f = $nt.d.f

@btime ComponentArray($nt)

map(k->getindex(mat, k, 1, 1), valkeys(ax))


nt = (a=[1], b=zeros(3), c=100*rand(7))
mat = ComponentArray(randn(11, 15), (getaxes(ComponentArray(nt))[1], FlatAxis()))
mat[:a, :] = nt

function custom_setindex!(A::ComponentArray, X::NamedTuple, inds...)
    value(::Val{k}) where k = k
    for k in valkeys(getaxes(A)[1])
        A[k, inds...] = X[value(k)]
    end
end

#Base.getindex(nt::NamedTuple, k::Val{S}) where S<:Symbol = return nt[S]

@btime custom_setindex!($mat, $nt, 1)




struct DummyStruct{T <: ComponentArray}
    data::T
    my_favorite_index::Int64
end

small_comp_array = ComponentArray(a=[0.1, 0.2], b=0.3)
comp_matrix = ComponentMatrix(randn(3, 5), getaxes(small_comp_array)[1], FlatAxis())
dummy = DummyStruct(ca, 1)

get_a1(x) = view(x, :a, 1)
get_a1(x::DummyStruct) = view(x, :a, 1)
get_favorite_a(x::DummyStruct) = view(x.data, Val(:a), x.my_favorite_index)
get_a(x, i) = view(x, :a, i)

@code_warntype get_a1(comp_matrix)
@code_warntype get_a1(dummy)
@code_warntype get_favorite_a(dummy)


using ComponentArrays
small_comp_array = ComponentArray(a=[0.1, 0.2], b=0.3)
comp_matrix = ComponentMatrix(randn(3, 5), getaxes(small_comp_array)[1], FlatAxis())
get_a1(x) = view(x, Val(:a), 1)
get_a(x, i) = view(x, Val(:a), i)

@code_warntype get_a1(comp_matrix)
@code_warntype get_a(comp_matrix, 1)
