import torch
from tqdm import tqdm
from functools import partial

from diffusers import DDIMScheduler,DPMSolverMultistepScheduler
import math



class SR3scheduler(DDIMScheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001,beta_end: float = 0.02, beta_schedule: str = 'linear',diff_chns=3):
        super().__init__(num_train_timesteps, beta_start ,beta_end,beta_schedule)
        # Initialize other attributes specific to SR3scheduler class
        # ...
        self.diff_chns = diff_chns
        # self.config.prediction_type = "sample"

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        # temporal_noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Only modify the last three channels of the tensor (assuming channels are in the second dimension)
        num_channels = original_samples.shape[1]
        if num_channels > self.diff_chns:
            # print(num_channels)
            # print(self.diff_chns)
            original_samples_select = original_samples[:, -self.diff_chns:].contiguous()
            noise_select = noise[:, -self.diff_chns:].contiguous()
            # temporal_noise_select = temporal_noise[:, -self.diff_chns:].contiguous()

            noisy_samples_select = sqrt_alpha_prod * original_samples_select + sqrt_one_minus_alpha_prod * noise_select

            noisy_samples = original_samples.clone()
            noisy_samples[:, -self.diff_chns:] = noisy_samples_select
        else:
            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

def create_SR3scheduler(opt,phase):
    
    steps= opt['num_train_timesteps'] if phase=="train" else opt['num_test_timesteps']
    scheduler=SR3scheduler(
        num_train_timesteps = steps,
        beta_start = opt['beta_start'],
        beta_end = opt['beta_end'],
        beta_schedule = opt['beta_schedule']
    )
    return scheduler
    
class SR3Sampler():
    
    def __init__(self,model: torch.nn.Module, scheduler:SR3scheduler,eta: float =.0):
        self.model = model
        # self.refine = refine
        self.scheduler = scheduler
        self.eta = eta
        


    def sample_high_res(self,x_batch: torch.Tensor, noisy_y, conditions=None, track_timesteps=None):
        """
        Using Diffusers built-in samplers
        
        Args:
            x_batch: Input images
            noisy_y: Initial noisy state
            conditions: [y0_cond, patches, attn]
            track_timesteps: List of timestep indices to track (e.g., [0, 20, 50, 100])
                            Returns embeddings at these timesteps if provided
        
        Returns:
            If track_timesteps is None: noisy_y (final denoised state)
            If track_timesteps is not None: (noisy_y, embeddings_dict)
                where embeddings_dict = {'t{idx}': features at timestep}
        """
        device = next(self.model.parameters()).device
        eta = torch.Tensor([self.eta]).to(device)
        x_batch=x_batch.to(device)
        y0, patches, attn = conditions[0].to(device),conditions[1].to(device),conditions[2].to(device)
        bz,nc,h,w = y0.shape
        
        # Track embeddings if requested
        embeddings_dict = {}
        total_timesteps = len(self.scheduler.timesteps)
        
        # Normalize requested indices to actual timestep range
        # If user requests [0, 20, 50, 100] but we have 10 timesteps, map proportionally
        if track_timesteps is not None:
            # Assume user's max index represents full sequence (typically 100 for diffusion)
            max_requested = max(track_timesteps) if track_timesteps else 100
            # Map requested indices to actual indices in our sequence
            # Create mapping: requested_index -> actual_index
            requested_to_actual = {}
            for req_idx in sorted(track_timesteps):
                # Map proportionally: req_idx / max_requested * (total_timesteps - 1)
                # e.g., req_idx=20, max=100, total=10 -> 20/100 * 9 = 1.8 -> 2
                actual_idx = int(round(req_idx / max_requested * (total_timesteps - 1)))
                actual_idx = min(actual_idx, total_timesteps - 1)  # Clamp to valid range
                actual_idx = max(actual_idx, 0)  # Ensure non-negative
                requested_to_actual[req_idx] = actual_idx
            
            # Create reverse mapping for fast lookup
            actual_to_requested = {}
            for req_idx, act_idx in requested_to_actual.items():
                if act_idx not in actual_to_requested:
                    actual_to_requested[act_idx] = []
                actual_to_requested[act_idx].append(req_idx)
            
            actual_indices_set = set(requested_to_actual.values())
        else:
            requested_to_actual = {}
            actual_to_requested = {}
            actual_indices_set = set()
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            self.model.eval()
            timesteps = t * torch.ones(bz*h*w, dtype=t.dtype, device=x_batch.device)
            with torch.no_grad():
                noise = self.model(x_batch, torch.cat([y0,noisy_y],dim=1), timesteps, patches, attn)
            noisy_y = self.scheduler.step(model_output = noise,timestep = t,  sample = noisy_y).prev_sample #eta = eta
            
            # Track embeddings at selected timesteps
            if track_timesteps is not None and t_idx in actual_indices_set:
                # Extract features: mean over spatial dimensions [batch, num_classes]
                features = noisy_y.mean(dim=[2, 3])  # [bz, nc]
                # Store with requested index as key (e.g., 't0', 't20', 't50', 't100')
                # Handle case where multiple requested indices map to same actual index
                requested_indices = actual_to_requested.get(t_idx, [])
                for req_idx in requested_indices:
                    embeddings_dict[f't{req_idx}'] = features.cpu().numpy()
            
            del noise
            torch.cuda.empty_cache()
        
        if track_timesteps is not None:
            return noisy_y, embeddings_dict
        return noisy_y



def create_SR3Sampler(model,opt):
    
    scheduler = create_SR3scheduler(opt,"test")
    scheduler.set_timesteps(opt['num_test_timesteps'])
    sampler = SR3Sampler(
        model = model,
        scheduler = scheduler,
        eta = opt['eta']
    )
    return sampler

def KL(logit1,logit2,reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1) 
    return (p1*(logp1-logp2)).sum(1)
