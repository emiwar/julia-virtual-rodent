@testset "VariationalBottleneck" begin
    import Flux
    import CUDA
    import Random

    v1 = VariationalLayer(rand(UInt64), zeros(4,5,6), ones(2, 1, 1))
    v2 = deepcopy(v1)

    v1(reshape([0.4, 0.2], 2, 1, 1))
    v2(reshape([0.4, 0.2], 2, 1, 1))




    regularization_loss(v1)
    regularization_loss(v2)

    v1_gpu = Flux.gpu(v1)
    v2_gpu = Flux.gpu(v2)
    v1_gpu(reshape(CUDA.cu([0.4, 0.2]), 2, 1, 1))
    v2_gpu(reshape(CUDA.cu([0.4, 0.2]), 2, 1, 1))

    regularization_loss(v1)
    regularization_loss(v1_gpu)
end