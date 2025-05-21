import torch
import torch.nn.functional as F
import torch.nn as nn


class JSDivergence(nn.Module):
    def __init__(self, eps=1e-8, topk=None):
        """
        Initialize Jensen-Shannon Divergence calculator.
        
        Args:
            eps: Small epsilon value for numerical stability
            topk: If provided, only consider top-k tokens from both distributions
        """
        super().__init__()
        self.eps = eps
        self.k = topk
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, p_probs, q_probs):
        """
        Compute Jensen-Shannon Divergence between two probability distributions.
        
        Args:
            p_probs: First probability distribution [batch_size, seq_len, vocab_size]
            q_probs: Second probability distribution [batch_size, seq_len, vocab_size]
            
        Returns:
            torch.Tensor: JS-Divergence for each position [batch_size, seq_len]
        """
        # Apply top-k filtering if needed
        if self.k:
            p_probs, q_probs = self._filter_topk_probs(p_probs, q_probs)
            
        # Add small epsilon for numerical stability and renormalize
        p = p_probs + self.eps
        q = q_probs + self.eps
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # Compute mixture distribution
        m = 0.5 * (p + q)

        # Compute KL divergences
        kl_p_m = self.kl_div(torch.log(m), p).sum(-1)
        kl_q_m = self.kl_div(torch.log(m), q).sum(-1)

        # JS divergence is the average of the KL divergences
        return 0.5 * (kl_p_m + kl_q_m)
    
    def _filter_topk_probs(self, p_probs, q_probs):
        """
        Filter to top-k tokens based on probability values.
        
        Args:
            p_probs: Probabilities of first distribution
            q_probs: Probabilities of second distribution
            
        Returns:
            tuple: (filtered_p_probs, filtered_q_probs) - Probabilities after top-k filtering
        """
        # Find the top-k probabilities for each distribution
        values_p, _ = torch.topk(p_probs, self.k, dim=-1)
        values_q, _ = torch.topk(q_probs, self.k, dim=-1)
        
        # Get the minimum value among the top-k for each position
        min_top_k_values_p = values_p[:, :, -1].unsqueeze(-1)
        min_top_k_values_q = values_q[:, :, -1].unsqueeze(-1)
        
        # Create boolean masks for top-k values
        top_k_mask_p = p_probs >= min_top_k_values_p
        top_k_mask_q = q_probs >= min_top_k_values_q
        
        # Use union of both masks
        top_k_mask = top_k_mask_p | top_k_mask_q
        
        # Apply the masks and zero out other values
        filtered_p_probs = torch.where(top_k_mask, p_probs, torch.zeros_like(p_probs))
        filtered_q_probs = torch.where(top_k_mask, q_probs, torch.zeros_like(q_probs))
        
        # Renormalize
        filtered_p_probs = filtered_p_probs / filtered_p_probs.sum(dim=-1, keepdim=True)
        filtered_q_probs = filtered_q_probs / filtered_q_probs.sum(dim=-1, keepdim=True)
        
        return filtered_p_probs, filtered_q_probs


class SymmetricKL(nn.Module):
    def __init__(self, eps=1e-8, topk=None):
        super().__init__()
        self.eps = eps
        self.k = topk
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, logits_p, logits_q):
        if self.k:
            # Find the top-k values for each distribution
            values_p, _ = torch.topk(logits_p, self.k, dim=-1)
            values_q, _ = torch.topk(logits_q, self.k, dim=-1)
            
            # Get the minimum value among the top-k for each position
            # This is our threshold - anything below this gets set to -inf
            min_top_k_values_p = values_p[:, :, -1].unsqueeze(-1)
            min_top_k_values_q = values_q[:, :, -1].unsqueeze(-1)
            
            # Create boolean masks where True indicates a value is in the top-k
            # Values >= the threshold are kept, values < threshold are set to -inf
            top_k_mask_p = logits_p >= min_top_k_values_p
            top_k_mask_q = logits_q >= min_top_k_values_q
            # Find union of both masks
            top_k_mask = top_k_mask_p | top_k_mask_q
            
            # Apply the masks to create filtered logits tensors
            filtered_logits_p = torch.where(top_k_mask, logits_p, torch.tensor(float('-inf'), device=logits_p.device))
            filtered_logits_q = torch.where(top_k_mask, logits_q, torch.tensor(float('-inf'), device=logits_q.device))

            # Get probabilities through softmax
            p = F.softmax(filtered_logits_p, dim=-1)
            q = F.softmax(filtered_logits_q, dim=-1)
        else:
            p = F.softmax(logits_p, dim=-1)
            q = F.softmax(logits_q, dim=-1)

        # Add small epsilon for numerical stability
        p = p + self.eps
        q = q + self.eps

        # Renormalize
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # Use PyTorch's optimized KLDivLoss
        kl_p_q = self.kl_div(torch.log(p), q).sum(-1)
        kl_q_p = self.kl_div(torch.log(q), p).sum(-1)
        return 0.5 * (kl_p_q + kl_q_p)
