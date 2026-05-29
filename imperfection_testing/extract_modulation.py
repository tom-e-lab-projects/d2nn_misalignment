import torch

def extract_params(file_name):
        '''
        extracts modulation tensors from state dictionary
        '''
        state_dict = torch.load(file_name)
        # system.mod. is new version
        mods_1 = [v for k, v in sorted(
                ((k, v) for k, v in state_dict.items() if k.startswith("system.mod.") ),
                key=lambda x: int(x[0].split('.')[2])
                )]
        mods_2 = [v for k, v in sorted(
                ((k, v) for k, v in state_dict.items() if k.startswith("mod.") ),
                key=lambda x: int(x[0].split('.')[1])
                )]
        return mods_1 + mods_2
    
def quantise_nbit(modulation, n =8):
    '''
    for quantizing the phase down 
    '''
    n_b = 2**n
    n_b_1 =n_b -1
    mod = torch.stack(modulation, dim=0) 
    mod_norm = mod / (2 * torch.pi)
    mod_q = torch.round(mod_norm * n_b_1)
    mod_q = torch.remainder(mod_q, n_b)
    mod_rescale = mod_q / n_b_1 * (2 * torch.pi)
    return mod_rescale