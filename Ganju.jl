module Ganju

using Printf: @printf

using Flux, Flux.Data.MNIST  # julia ML library
using Flux.Tracker
using Flux.Tracker: update!
using Flux: binarycrossentropy


export get_mnist, training_loop!, make_generator, make_discriminator, TrainingModel


# target: generate digits using MNIST data

# TODO:
# 1. define a generator net
# 2. define a discriminator net
# 3. define the cost function and then perform a gradient descent update
INPUT_DIM_DIS = 784
HIDDEN_DIM = 64
OUTPUT_DIM_DIS = 1
INPUT_DIM_GEN = 64
OUTPUT_DIM_GEN = INPUT_DIM_DIS


DEFAULT_BATCH_SIZE = 32



function make_discriminator()
    Chain(Dense(INPUT_DIM_DIS, HIDDEN_DIM, relu),
          Dense(HIDDEN_DIM, OUTPUT_DIM_DIS, sigmoid))
end

function make_generator()
    Chain(Dense(INPUT_DIM_GEN, HIDDEN_DIM, relu),
          Dense(HIDDEN_DIM, OUTPUT_DIM_GEN, sigmoid))
end


function fetch_training_sample()
end


function get_mnist(batch_size)
    imgs = MNIST.images()
    Iterators.partition(imgs, batch_size)
end


function get_mnist()
    get_mnist(DEFAULT_BATCH_SIZE)
end


function fetch_next_batch()
end


# struct LabelledSample
#     sample::Array{Float32}
#     label::Int
# end


struct TrainingModel
    gen
    disc
    discr_losses
    gen_losses

    TrainingModel() = new(make_generator(),
                          make_discriminator(),
                          [], [])
end


function training_loop!(model::TrainingModel, k::Int, num_epochs::Int,
                        batch_size::Int)
    disc_opt = ADAM()  # TODO: why does this not work with Momentum?
    gen_opt = ADAM()

    # disc_opt = Descent(0.0001)
    # gen_opt = Descent(0.0001)

    function training_step!(opt, model, predictions, labels, losses)
        function model_loss()
            EPS = 0.0001
            loss = sum(binarycrossentropy.((1-EPS)*predictions[:].+EPS/2, labels)) / batch_size
            # manually add an epsilon term for numerical stability
            push!(losses, loss)
            return loss
        end

        pars = params(model)
        grads = Tracker.gradient(model_loss, pars)

        for p in pars
            update!(opt, p, grads[p])
        end
    end

    for i in 1:num_epochs
        @printf "epoch %d\n" i
        for (j, batch) in enumerate(get_mnist(batch_size))
            batch = map(Float32, hcat([reshape(z, :) for z in batch]...))
            # reshape and
            # collect along new axis

            fake_samples = model.gen(rand(INPUT_DIM_GEN, batch_size))
            all_samples = [batch fake_samples]
            @assert size(all_samples) == (INPUT_DIM_DIS, 2*batch_size)

            labels = [repeat([Float32(1.0)], batch_size); 
                      repeat([Float32(0.0)], batch_size)]

            predictions = model.disc(all_samples)

            @assert size(labels) == size(predictions[:])

            training_step!(disc_opt, model.disc, predictions, labels,
                           model.discr_losses)

            if j % k == 0
                predictions = model.disc(model.gen(rand(INPUT_DIM_GEN, batch_size)))
                labels = repeat([Float32(0.0)], batch_size) 
                training_step!(gen_opt, model.disc, predictions, labels,
                               model.gen_losses)
            end
        end
    end
end


end
