import torch
import torch.nn as nn

class ContextualCritic(nn.Module):
  def __init__(self, in_ch = 3, first_ch = 64, final_ch = 256, is_local = True):
    super(ContextualCritic, self).__init__()

    layers = []

    layers.append(nn.Conv2d(in_ch, first_ch, kernel_size = 5, stride = 2, padding = 2))
    layers.append(nn.LeakyReLU(negative_slope = 0.2))
      
    curr_ch = first_ch
    while(curr_ch < final_ch):
      layers.append(nn.Conv2d(curr_ch, curr_ch * 2, kernel_size = 5, stride = 2, padding = 2))
      layers.append(nn.LeakyReLU(negative_slope = 0.2))
      curr_ch = curr_ch * 2

    if(is_local):
      layers.append(nn.Conv2d(curr_ch, curr_ch * 2, kernel_size = 5, stride = 2, padding = 2))
    else:
      layers.append(nn.Conv2d(curr_ch, curr_ch, kernel_size = 5, stride = 2, padding = 2))
    
    layers.append(nn.LeakyReLU(negative_slope = 0.2))
    self.critic_net = nn.Sequential(*layers)

  def forward(self, x, is_local, f_obj_to_img = None, batch_size = None):
    x = self.critic_net(x)
    x = x.view(x.shape[0], -1)

    if (is_local):
      return self.ave_imgs(x, f_obj_to_img, batch_size)
    else:
      return x
      
  def ave_imgs(self, x, f_obj_to_img, batch_size):
    print("X: " + str(x.shape))
    """
    # Get average scores of all objects per image
    """
    fake_avgs = []
    real_avgs = []
    fake_scores, real_scores = torch.split(x, x.shape[0] // 2, dim = 0)
    
    print("OBJS: " + str(f_obj_to_img))
    print("FAKE: " + str(fake_scores.shape))
    print("REAL: " + str(real_scores.shape))
    for i in range(batch_size):
      if i in f_obj_to_img:
        # get indices of objects that belong to image i
        img_inds = (f_obj_to_img == i).nonzero().squeeze(1)
        
        fake_mean = torch.mean(fake_scores[img_inds], dim = 0).unsqueeze(0)
        real_mean = torch.mean(real_scores[img_inds], dim = 0).unsqueeze(0)
        fake_avgs.append(fake_mean)
        real_avgs.append(real_mean)
      else:
        fake_mean = torch.zeros([1, fake_scores.shape[1]], device = fake_scores.device)
        real_mean = torch.zeros([1, real_scores.shape[1]], device = real_scores.device)
        fake_avgs.append(fake_mean)
        real_avgs.append(real_mean)

    fake_avgs = torch.cat(fake_avgs)
    real_avgs = torch.cat(real_avgs)
    ave_x = torch.cat([fake_avgs, real_avgs])
    
    print("AVE X: " + str(ave_x.shape))
    
    return ave_x