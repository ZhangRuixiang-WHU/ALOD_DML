import torch


def main():
    
    checkpoint = 'PATH_OF_PRETRAINED_MODEL'
    output = 'PATH_OF_PRETRAINED_MODEL_EXPAND_FOR_T-S_MODEL'
    ck = torch.load(checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(meta=ck['meta'], state_dict=dict(), author='alod')
    for key, value in ck['state_dict'].items():
        output_dict['state_dict']['teacher.'+key] = value
    for key, value in ck['state_dict'].items():
        output_dict['state_dict']['student.'+key] = value

    torch.save(output_dict, output)


if __name__ == '__main__':
    main()
