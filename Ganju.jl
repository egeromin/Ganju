module Ganju

using Flux, Flux.Data.MNIST  # julia ML library
using Flux.Tracker
using Flux.Tracker: update!
using Flux: binarycrossentropy


export get_mnist, training_loop!, make_generator, make_discriminator


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


function training_loop!(gen, disc, k::Int, num_epochs::Int, batch_size::Int)

    disc_opt = Descent(0.001)  # TODO: why does this not work with Momentum?
    gen_opt = Descent(0.001)

    for i in 1:num_epochs
        for batch in get_mnist(batch_size)
            println("la")
            batch = map(Float32, hcat([reshape(z, :) for z in batch]...))
            # reshape and
            # collect along new axis

            fake_samples = gen(rand(INPUT_DIM_GEN, batch_size))
            all_samples = [batch fake_samples]
            @assert size(all_samples) == (INPUT_DIM_DIS, 2*batch_size)

            labels = [repeat([Float32(1.0)], batch_size); 
                      repeat([Float32(0.0)], batch_size)]

            predictions = disc(all_samples)

            @assert size(labels) == size(predictions[:])

            discr_loss = () -> sum(binarycrossentropy.(predictions[:],
                                                       labels)) / batch_size
            pars = params(disc)
            grads = Tracker.gradient(discr_loss, pars)

            for p in pars
                update!(disc_opt, p, grads[p])
            end

            # TODO: use k

            predictions = disc(gen(rand(INPUT_DIM_GEN, batch_size)))
            labels = repeat([Float32(0.0)], batch_size) 
            gen_loss = () -> sum(binarycrossentropy.(predictions[:], labels)) / batch_size
            pars = params(gen)
            grads = Tracker.gradient(gen_loss, pars)

            for p in pars
                update!(disc_opt, p, grads[p])
            end
        end

    end

end


end
