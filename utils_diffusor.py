import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from diffusers import UNet1DModel

# ---------------------------------------------------------- #
# --------------------- CONFIG ----------------------------- #
# ---------------------------------------------------------- #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = "09-12-2023_22-07-27_final_step_13100"

class SamplingConfig:
  batch_size = 3
  horizon = 160 # Must be multiple of 8
  state_dim = 2
  action_dim = 2
  learning_rate = 1e-4 # Only relevant to load the optimizer
  eta = 1.0
  num_train_timesteps = 1000
  min = 0
  max = 20
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #

def get_model(type="unet1d"):
    '''
        Initializes and sets up the U-Net model.
    '''
    if type == "unet1d":
        model = UNet1DModel(
            sample_size = 24, #TODO: Still not sure how to update this and when.
            sample_rate = None,
            in_channels= 4, # Update depending on the number of channels in the data (i.e., dim states + dim actions)
            out_channels= 4, # Update depending on the number of channels in the data (i.e., dim states + dim actions)
            extra_in_channels= 0,
            time_embedding_type = "positional",
            flip_sin_to_cos = True,
            use_timestep_embedding = True,
            freq_shift = 0.0,
            down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
            up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D"),
            mid_block_type = "MidResTemporalBlock1D",
            out_block_type = ("OutConv1DBlock"),
            block_out_channels = (32, 64, 128, 256),
            act_fn= 'mish',
            norm_num_groups= 8,
            layers_per_block= 1,
            downsample_each_block = False,
        ).to(DEVICE)
    else:
        raise ValueError("Model type not supported")
    return model

def reset_start_and_target(x_in, cond, act_dim):
    if cond == {}:
        return x_in
    for key, val in cond.items():
        try:
            x_in[:, act_dim:, key] = val.clone()
        except Exception as e:
            print("Error in reset_start_and_target")
            print("x_in.shape: ", x_in.shape)
            print("act_dim: ", act_dim)
            print("key: ", key)
            print("val.shape: ", val.shape)
            print("x_in[:,act_dim:, key].shape: ", x_in[:, act_dim:, key].shape)
            print(e)
    return x_in

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Loaded checkpoint from epoch {} with loss {} at path {}".format(epoch, loss, filepath))
    return model, optimizer

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")

def limits_normalizer(x):
    '''
        Normalizes the input to the range [-1, 1]
    '''
    print("Normalizing data according to limits: ")
    print("x.min(): ", x.min())
    print("x.max(): ", x.max())
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1

def limits_unnormalizer(x, min, max):
    '''
        Unormalizes the input from the range [-1, 1] to [min, max]
    '''
    if x.max() > 1 + 0.0001 or x.min() < -1 - 0.0001:
        print("Input is not normalized!", x.min(), x.max())
        x = np.clip(x, -1, 1)
    return 0.5 * (x + 1) * (max - min) + min


if __name__ == "__main__":
  config = SamplingConfig()
  shape = (config.batch_size,config.state_dim+config.action_dim, config.horizon)
  scheduler = get_noise_scheduler(config)
  model = get_model("unet1d")
  optimizer = get_optimizer(model, config)
  model, optimizer = load_checkpoint(model, optimizer, "models/"+CHECKPOINT+".ckpt")
  conditions = {
                0: torch.ones((config.batch_size, config.state_dim))*(0.6),
                -1: torch.ones((config.batch_size, config.state_dim))*(-0.6) *torch.tensor([-1, 1])
              }

  x = torch.randn(shape, device=DEVICE)
  print("Initial noise vector: ", x[0,:,:])

  x = reset_start_and_target(x, conditions, config.action_dim)
  print("Initial noise vector after setting start and target: ", x[0,:,:])

  for i in tqdm.tqdm(scheduler.timesteps):

      timesteps = torch.full((config.batch_size,), i, device=DEVICE, dtype=torch.long)

      with torch.no_grad():
        # print("shape of x and timesteps: ", x.shape, timesteps.shape)
        residual = model(x, timesteps).sample

      obs_reconstruct = scheduler.step(residual, i, x)["prev_sample"]

      if config.eta > 0:
        noise = torch.randn(obs_reconstruct.shape).to(obs_reconstruct.device)
        posterior_variance = scheduler._get_variance(i)
        obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * config.eta* noise  # no noise when t == 0

      obs_reconstruct_postcond = reset_start_and_target(obs_reconstruct, conditions, config.action_dim)
      x = obs_reconstruct_postcond

      # convert tensor i to int 
      i = int(i)
      if i == 1:
        print(f"At step {i}:", x[0,:,:],"\n" , limits_unnormalizer(x[0,:,:].cpu(), config.min, config.max))
        for k in range(3):
          unnormalized_output = limits_unnormalizer(x[k,:,:].cpu(), config.min, config.max)

          num_points = unnormalized_output.shape[1]
          colors = np.linspace(0, 1, num_points)
          colors[0] = 1

          plt.ylim(config.min, config.max)
          plt.xlim(config.min, config.max)
          plt.scatter(unnormalized_output[2,:], unnormalized_output[3,:], c=colors, cmap='Reds')
          plt.quiver(unnormalized_output[2,:], unnormalized_output[3,:], unnormalized_output[0,:], unnormalized_output[1,:], color='g', scale=30, headwidth=1)
          print(i, k)
          plt.show()  
