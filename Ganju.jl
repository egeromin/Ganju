module Ganju

using Flux  # julia ML library


# target: generate digits using MNIST data

# TODO: 
# 1. define a generator net
# 2. define a discriminator net
# 3. define the cost function and then perform a gradient descent update


function make_discriminator()
    Chain(Dense(28^2, 32, relu),
          Dense(32, 10),
          softmax) 
end

function make_generator()
    Chain(Dense(10, 32, relu),
          Dense(32, 28^2), sigmoid)
end

function fetch_training_sample()
end


function training_loop(m:Int, num_training_steps:Int)

end


end
