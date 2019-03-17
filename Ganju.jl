module Ganju

using Flux, FLUX.data.MNIST  # julia ML library

imgs = MNIST.images()

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


function training_loop(m:Int, num_training_steps:Int)

end


end
