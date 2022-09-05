import torch
ckpt_path = '/home/sungmanc/scripts/ote/training_extensions/external/model-preparation-algorithm/submodule/logs/train_only_det/20220902_161549/stage00_train/epoch_10.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

state_dict = ckpt['state_dict']

new_dict = {'meta':ckpt['meta'], 'state_dict':{}}
for k, v in state_dict.items():
    if k.startswith('detector.'):
        new_k = k.replace('detector.', '')
        new_dict['state_dict'][new_k] = v

torch.save(new_dict, 'only_det.pth')
        
    