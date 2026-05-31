import torch
from loss_unflatten import loss , system_with_unflatten
from cma_es import inverse_by_optimization

c = 3e8*1e3             # speed of light
f = 400e9               # 400GHz
lambda0 = c/f           # wavelength
p = 128
dimensions = torch.tensor([80,80,p,p])
distances = [30,30,30]
images = torch.ones(3,p,p)
true_displacement = torch.tensor([[4,-3],[-3,4],[3,2]]) # less than a pixel off
true_displacement = torch.flatten(true_displacement)
tilt = torch.tensor([0.1,-0.2])

model = system_with_unflatten(lambda0, dimensions, distances, images, tilt)
true_output = model.eval(true_displacement)
#true_output, loss, system, initial_inputs, initial_losses, no_samples_next_gen, no_samples_cycle
no_samples_next_gen = 20
no_samples_cycle = 200
normalising_inputs = 10*(torch.rand( no_samples_cycle,6)-0.5)
normalising_outputs = []
for i in range (no_samples_cycle):
    normalising_outputs.append(model.eval(normalising_inputs[i]))
loss_method = loss(true_output, torch.stack(normalising_outputs))

initial_losses = []
initial_inputs = []

for i in range (no_samples_next_gen):
    print("i+1")
    temp_inputs = 2*(torch.rand( no_samples_cycle,6)-0.5)
    temp_outputs = []
    for i in range (no_samples_cycle):
        temp_outputs.append(model.eval(temp_inputs[i]).squeeze(0))
    temp_outputs = torch.stack(temp_outputs)
    losses = loss_method.eval(temp_outputs)
    best_idx = torch.argmin(losses)
    best_input = temp_inputs[best_idx]
    best_loss = losses[best_idx]
    initial_losses.append(best_loss)
    initial_inputs.append(best_input)



cma = inverse_by_optimization(loss_method.eval, model.eval, initial_inputs, initial_losses, no_samples_next_gen, no_samples_cycle)
optimized = cma.run_optimization(1e-7)
print(optimized)

'''
key work for now is to work out size matching of tensors
and to work out what should be nn.Module
then can see if i can get the method to work for three stage system
then maybe try and build a system to try deterministic optimizaiton
for more images

later i need to gpu prepare all this code
'''