module Ganju

using Flux, Flux.Data.MNIST  # julia ML library


export get_mnist


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
          Dense(HIDDEN_DIM, OUTPUT_DIM_DIS),
          sigmoid)
end

function make_generator()
    Chain(Dense(INPUT_DIM_GEN, HIDDEN_DIM, relu),
          Dense(HIDDEN_DIM, OUTPUT_DIM_GEN),
          sigmoid)
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


function training_loop(m::Int, num_epochs::Int, batch_size::Int)

    for i in 1:num_epochs
        for batch in get_mnist(batch_size)
            #...
        end
    end

end


end
