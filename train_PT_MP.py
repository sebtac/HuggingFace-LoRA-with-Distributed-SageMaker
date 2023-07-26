import time, os, argparse
os.system("! pip install transformers")

import torch
#import torch.distirbuted as dist # No module named 'torch.distirbuted' ?!?!?!?!?!?
from torch.utils.data import Dataset
from transformers import ViTForImageClassification, ViTConfig

class FakeDataset(Dataset):
    def __len__(self):
        return 128# 1000000
    def __getitem__(self, index):
        #rand_image = torch.randn([3, 224, 224], dtype=torch.float32) # InExample ??!?!?!?
        rand_image = torch.randn([3, 256, 256], dtype=torch.float32)        
        label = torch.tensor(data=[index % 1000], dtype=torch.int64)
        return rand_image, label

def build_model():
    model_args = {
        "image_size": 256,
        "patch_size": 16,
        "hidden_size": 1024,
        "num_hidden_layers": 50,
        "num_attention_heads": 16,
        "intermediate_size": 4*1024,
        "num_labels": 1000
        }
    model = ViTForImageClassification(ViTConfig(**model_args))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='/tmp', type=str)
    args, _ = parser.parse_known_args()
    
    # init
    import smdistributed.modelparallel.torch as smp
    smp.init()
    
    from deepspeed.runtime.zero import stage3
    stage3.assert_ints_same_as_other_ranks = lambda x: None
    torch.cuda.set_device(smp.local_rank())
    
    dataset = FakeDataset()
    
    #"""
    data_loader = torch.utils.data.DataLoader(dataset,
                                              #batch_size=4, # InExample: 4
                                              batch_size=8*8, # InExample: 4 -- this is for microbatches test
                                              num_workers=12)
    #"""
    
    """ ALTERNATIVE DATA GENERATION
    from torch.utils.data.distributed import DistributedSampler

    data_loader = DistributedSampler(dataset, 
                                     num_replicas = dist.get_world_size(), 
                                     rank = dist.get_rank()
                                    )
    """
    model = build_model()
    model = smp.DistributedModel(model)
    
    # add checkpoint activation
    for m in model.get_module().vit.encoder.layer.children():
        smp.set_activation_checkpointing(m)
    
    optimizer = torch.optim.Adam(model.parameters())
    optimizer = smp.DistributedOptimizer(optimizer)
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    model.train()
    
    t0 = time.perf_counter()
    summ = 0
    count = 0
    
    @smp.step
    def train_step(model, inputs, targets):
        #print("TARGETS 1", targets)
        
        outputs = model(inputs)
        targets = targets.squeeze(1)
        loss = loss_function(outputs['logits'], targets)
        model.backward(loss)
        return loss
    
    for idx, (inputs, targets) in enumerate(data_loader, start=1):
        inputs = inputs.to(torch.cuda.current_device())
        targets = targets.to(torch.cuda.current_device())
        #print("TARGETS 2", targets)
        
        optimizer.zero_grad()
        
        loss_mb = train_step(model, inputs, targets)
        loss = loss_mb.reduce_mean()
        
        optimizer.step()
        
        if torch.distributed.get_rank() == 0:
            batch_time = time.perf_counter() - t0
            print(f'step: {idx}: step time is {batch_time}')
            
            if idx > 1:  # skip first step
                summ += batch_time
                count += 1
                
    if torch.distributed.get_rank() == 0:
        print(f'average/sum step time: {batch_time/count}, {batch_time}')
