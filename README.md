# d2nn_misalignment
A codebase for training diffractive deep neural networks (d2nn) to perform MNIST digit task, measuring impact of misalignment on accuracy, and compensation.

The concept behind the compensation is that general compensation can be achieved by measuring the size of misalignments, from which it is easy to compensate. Axial misalignment effects are small, and tilt and lateral displacement are close to degenerate so the compensation is just for these two which are assumed degenerate. The compensation is driven by the fact that size and position of the output is strongly impacted by lateral misalignment/ tilt. Thus moments of the output are measured. For deep systems, multiple inputs can be constructed through deliberate beam steering of modulation layers to lift degeneracy.

## repository structure

```text

├── physics_modelling/ simulating the physical system
│   ├── subsystems  the basic units as nn.Modules
│   └── N_layer_system  building into general system
│ 
├── training_testing/  training systems to perform MNIST digit task 
│   ├── digits_generator  extracting MNIST digits from torchvision lib
│   ├── target_class  preparing DataSet object with inputs and label pairs
│   ├── train_validate  general classification training and validation loops
│   ├── npcc  negative pearson correlation coefficient loss
│   └── training_system/ results of 1 to 6 layer training
│
├── training_ideal  notebook of training of 1 to 6 layer systems
│
├── imperfection_testing/  codes for building imperfection experiments
│   ├── extract_modulation  extractinglearned parameters from dictiionary in .pt 
│   ├── misalgnment_classes  for single scale misalignment experiment
│   ├── misalignment_sweep  for sweeps across multiple scales
│   ├── accuracy_plot  for plotting accuracy against size of misalignment
│   └── results/ results from notebooks below
│
├── quant_waist  notebook of testing impact of qunatization of phase and beam size
├── misalign  notebook of testing impact of misalignments
│
note these below as of now need some work
│
├── compensation/
│   ├── moment_class  prepares random data for either of the two methods
│   ├── inverse_train_nn  for training a neural network to learn a function
│   └── inverse_by_optimization  for learning misalignment with cma es 
│ 
old codebases for until I've completed corrections to compensation
│
├── original_codebase/  contains neural network stuff
└── misalign/  contains stochastic optimization stuff
```

