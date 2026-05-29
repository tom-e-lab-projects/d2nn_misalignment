import torch
import csv
import imperfection_testing.misalignment_classes as mc
import training_testing.train_validate as tv
import imperfection_testing.extract_modulation as em


def acc_m_std_dev(config,
                  distances,    
                  misalignment_config,  
                  testing_config,
                  n_samples, 
                  scale,
                  input_file
                  ):
    '''
    for an experiment defined by misalignment_config at a particular scale
    runs accuracies for all samples, then calculated mean and standard deviation
    '''
    
    accs = torch.zeros(n_samples)

    for i in range ( n_samples ):
        # for loop here is because my computer is weak
        
        system = mc.Misalign(config,
                             distances,
                             False,
                             misalignment_config,
                             scale,
                             em.extract_params(input_file)
                             )
        
        _, acc, _, _ = tv.validation(system, 
                                     testing_config
                                     )
        
        # convert to %
        accs[i] = 100 * acc

    acc_mean = torch.mean(accs).item()
    
    if accs.size(0) == 1:
        acc_std = 0
    else:
        acc_std = torch.std(accs).item()

    return acc_mean, acc_std


def save_csv(file, header, rows, append):
    '''
    optionally for appending existing or creating a new csv
    '''
    
    mode = 'a' if append else 'w'

    with open(file, mode, newline='') as f:
        writer = csv.writer(f)

        if not append:
            writer.writerow(header)

        writer.writerows(rows)

def full_experiment(config,
                    distances,
                    misalignment_config, 
                    testing_config,
                    chosen_field,
                    scales_samples,
                    destination_file,
                    input_files,
                    append=False,
                    ):
    '''
    scales_sampels is a list of samples containing the scale of misaligment and the number of samples taken at this scale
    input_files is a list of tuples containing number of layers, input file containing modulation trained parameters,
    distances between layers and the misalignment config
    misalignment_config is a list of msialignment_configs as defined in misalgmnet_classes corresponding to each system
    see misalign.ipynb for an example implementation

    for chosen_field, 0,1,2 indicates tilt, xy_displacement, z_displacement
    destination_file is where you want the data saved
    '''

    #setting up lists for csv
    header = ['N_layers']
    scales = [scale for scale,_ in scales_samples]
    header = header + scales
    rows = []
    
    for i, input_file, distances, misalignment_config in input_files:

        row_mean = [ i ]
        row_std = [ f'{i}_std']

        for scale, sample in scales_samples:

            '''
            config,
                  distances,    
                  misalignment_config,  
                  testing_config,
                  n_samples, 
                  scale,
                  input_file
                  '''

            scale_tensor = torch.zeros(3)
            scale_tensor[chosen_field] = scale

            acc_mean, acc_std = acc_m_std_dev( config,
                                                distances,
                                                misalignment_config, 
                                                testing_config,
                                                sample, 
                                                scale_tensor,
                                                input_file
                                            )
            
            row_mean.append( acc_mean )
            row_std.append( acc_std )

        rows.append(row_mean)
        rows.append(row_std)
        print(f'results for {i} layers: means: {row_mean}, std: {row_std}')

    save_csv(destination_file, header, rows, append)