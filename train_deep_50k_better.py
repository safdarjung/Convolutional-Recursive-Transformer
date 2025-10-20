"""
CRT v2 DEEP - 50K Training Samples with RESUME CAPABILITY
==========================================================
Deep model optimized for RTX 4050 6GB with maximum capacity
- 8 recursive loops (vs 6 before)
- 8 TRM cycles (vs 6 before)
- 50K training samples (vs 20K before)
- Resume training from checkpoint support
- Expected perplexity: ~1.08-1.10
"""


import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
from datetime import datetime
from typing import Optional, List
import warnings
import argparse
warnings.filterwarnings('ignore')


print("🔧 Initializing Deep Model Training...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("⚠️  No GPU detected - training will be slow!")


# ============================================================================
# MODEL COMPONENTS
# ============================================================================


class DepthwiseDilatedConv1d(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, 
                                   groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.depthwise(h)
        h = F.gelu(h)
        h = self.pointwise(h)
        h = h.transpose(1, 2)
        return self.dropout(h)



class FlexibleConvTRMBlock(nn.Module):
    def __init__(self, d_model, num_cycles=6, kernel_size=3, dropout=0.1,
                 norm_type='layer', dilation_schedule: Optional[List[int]] = None):
        super().__init__()
        self.num_cycles = num_cycles
        self.dropout = nn.Dropout(dropout)
        
        if dilation_schedule is None:
            dilation_schedule = [1, 2, 4]
        self.dilations = [dilation_schedule[i % len(dilation_schedule)] 
                         for i in range(num_cycles)]
        
        self.convs = nn.ModuleList([
            DepthwiseDilatedConv1d(d_model, kernel_size=kernel_size,
                                   dilation=self.dilations[i], dropout=dropout)
            for i in range(num_cycles)
        ])


        if norm_type == 'layer':
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_cycles)])
        elif norm_type == 'batch':
            self.norms = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(num_cycles)])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(num_cycles)])
        self.norm_type = norm_type


        self.lin1 = nn.ModuleList([nn.Linear(d_model, d_model * 4) for _ in range(num_cycles)])
        self.lin2 = nn.ModuleList([nn.Linear(d_model * 4, d_model) for _ in range(num_cycles)])


    def forward(self, x):
        h = x
        for i in range(self.num_cycles):
            h_conv = self.convs[i](h)
            if self.norm_type == 'batch':
                h_conv = self.norms[i](h_conv.transpose(1, 2)).transpose(1, 2)
            else:
                h_conv = self.norms[i](h_conv)
            mlp_out = self.lin2[i](F.gelu(self.lin1[i](h_conv)))
            h = h + self.dropout(mlp_out)
        return h



class TRMEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, trm_cycles=3, kernel_size=3, 
                 norm_type='layer', dropout=0.1, dilation_schedule=None):
        super().__init__()
        embed_dim = d_model // 2
        self.base_embedding = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, d_model)
        self.refine = FlexibleConvTRMBlock(d_model, trm_cycles, kernel_size,
                                           dropout, norm_type, dilation_schedule)
    
    def forward(self, input_ids):
        x = self.base_embedding(input_ids)
        x = self.projection(x)
        x = self.refine(x)
        return x



class EarlyExitController(nn.Module):
    def __init__(self, mode='max_softmax', tau0=0.92, tau_min=0.80, 
                 decay=0.98, temperature=1.0):
        super().__init__()
        self.mode = mode
        self.tau0 = tau0
        self.tau_min = tau_min
        self.decay = decay
        self.temperature = temperature


    def threshold(self, step):
        return max(self.tau_min, self.tau0 * (self.decay ** step))


    def confident(self, logits=None, h=None, h_prev=None, step=0):
        tau = self.threshold(step)
        if self.mode == 'max_softmax':
            assert logits is not None
            probs = F.softmax(logits / self.temperature, dim=-1)
            confidence = probs.amax(dim=-1)
            exit_mask = (confidence >= tau)
            return confidence, exit_mask
        else:
            assert h is not None and h_prev is not None
            cos_sim = F.cosine_similarity(h, h_prev, dim=-1)
            confidence = cos_sim
            exit_mask = (confidence >= tau)
            return confidence, exit_mask



class CRTv2WithIntermediateHeads(nn.Module):
    """CRT v2 with Intermediate Supervision"""
    def __init__(self, vocab_size=50257, d_model=512, num_heads=8, trm_cycles=6,
                 num_loops=6, kernel_size=3, norm_type='layer', max_seq_len=1024,
                 dropout=0.1, use_trm_embeddings=True, embedding_trm_cycles=3,
                 dilation_schedule=(1, 2, 4), use_checkpointing=True, early_exit=False,
                 ee_mode='max_softmax', ee_tau0=0.92, ee_tau_min=0.80, 
                 ee_decay=0.98, ee_temperature=1.0, intermediate_loops=[2, 4]):
        super().__init__()
        
        self.d_model = d_model
        self.num_loops = num_loops
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_trm_embeddings = use_trm_embeddings
        self.use_checkpointing = use_checkpointing
        self.early_exit = early_exit
        self.intermediate_loops = intermediate_loops


        if use_trm_embeddings:
            self.embed = TRMEmbedding(vocab_size, d_model, embedding_trm_cycles,
                                      kernel_size, norm_type, dropout, dilation_schedule)
        else:
            self.embed = nn.Embedding(vocab_size, d_model)
        
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.trm = FlexibleConvTRMBlock(d_model, trm_cycles, kernel_size,
                                        dropout, norm_type, dilation_schedule)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.loop_encodings = nn.Parameter(torch.randn(num_loops, 1, 1, d_model) * 0.02)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.intermediate_heads = nn.ModuleDict({
            f'loop_{loop}': nn.Linear(d_model, vocab_size, bias=False)
            for loop in intermediate_loops
        })
        self.ee_controller = EarlyExitController(ee_mode, ee_tau0, ee_tau_min, ee_decay, ee_temperature)


    def _attention(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training:
            attn_weights = F.dropout(attn_weights, p=0.1)
        attn_out = torch.matmul(attn_weights, v)
        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


    def _loop_body(self, x, loop_idx):
        x = x + self.loop_encodings[loop_idx]
        attn_out = self._attention(x)
        x = self.norm1(x + self.dropout_layer(attn_out))
        trm_out = self.trm(x)
        x = self.norm2(x + trm_out)
        return x


    def forward(self, input_ids, return_all_outputs=False, return_exit_info=False):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed(input_ids)
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = x + self.pos_embed(positions)


        if self.early_exit and not self.training:
            active = torch.ones(B, T, dtype=torch.bool, device=device)
            exited_at = torch.full((B, T), fill_value=self.num_loops, dtype=torch.int, device=device)
        
        states = x
        prev_states = None
        intermediate_outputs = {}


        for loop_idx in range(self.num_loops):
            prev_states = states
            if self.use_checkpointing and self.training:
                states = checkpoint(self._loop_body, states, loop_idx, use_reentrant=False)
            else:
                states = self._loop_body(states, loop_idx)


            current_loop = loop_idx + 1
            if current_loop in self.intermediate_loops:
                head_key = f'loop_{current_loop}'
                intermediate_outputs[head_key] = self.intermediate_heads[head_key](states)


            if self.early_exit and not self.training:
                if current_loop in self.intermediate_loops:
                    logits_for_exit = intermediate_outputs[f'loop_{current_loop}']
                else:
                    logits_for_exit = self.lm_head(states)
                
                conf_scores, should_exit = self.ee_controller.confident(
                    logits=logits_for_exit if self.ee_controller.mode == 'max_softmax' else None,
                    h=states, h_prev=prev_states, step=loop_idx
                )
                new_exits = active & should_exit
                exited_at[new_exits] = current_loop
                active = active & (~should_exit)
                if (~active).any():
                    states = torch.where((~active).unsqueeze(-1), prev_states, states)
                if not active.any():
                    break


        final_logits = self.lm_head(states)
        if return_all_outputs:
            return final_logits, intermediate_outputs
        if return_exit_info and self.early_exit:
            return final_logits, exited_at
        return final_logits


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# ============================================================================
# DATASET
# ============================================================================


from datasets import load_dataset
from transformers import GPT2Tokenizer


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=384):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                   max_length=max_length, return_tensors='pt')
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        ids = self.encodings['input_ids'][idx]
        return {'input_ids': ids, 'labels': ids}



# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load checkpoint and resume training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with resume information (start_epoch, train_losses, val_losses)
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model'])
    print("✅ Model state loaded")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("✅ Optimizer state loaded")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("✅ Scheduler state loaded")
    
    # Extract resume information
    resume_info = {
        'start_epoch': checkpoint.get('epoch', 0) + 1,  # Start from next epoch
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
    }
    
    print(f"📊 Resuming from epoch {resume_info['start_epoch']}")
    print(f"📈 Previous training history: {len(resume_info['train_losses'])} epochs")
    if resume_info['best_val_loss'] < float('inf'):
        print(f"🏆 Best validation loss so far: {resume_info['best_val_loss']:.4f}")
    
    return resume_info


def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, 
                   train_losses, val_losses, best_val_loss=None, is_best=False):
    """
    Save training checkpoint
    
    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        train_losses: List of training losses
        val_losses: List of validation losses
        best_val_loss: Best validation loss achieved
        is_best: Whether this is the best model
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.replace('latest_checkpoint.pt', 'best_model.pt')
        torch.save(model.state_dict(), best_path)


# ============================================================================
# TRAINING
# ============================================================================


@torch.no_grad()
def evaluate(model, loader, device, with_intermediate=False):
    model.eval()
    total_loss = 0.0
    intermediate_losses = {f'loop_{loop}': 0.0 for loop in model.intermediate_loops}
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        if with_intermediate:
            final_logits, intermediate_outputs = model(input_ids, return_all_outputs=True)
            loss = F.cross_entropy(final_logits[:, :-1].reshape(-1, final_logits.size(-1)),
                                  labels[:, 1:].reshape(-1))
            total_loss += loss.item()
            for key, logits in intermediate_outputs.items():
                loss_int = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                          labels[:, 1:].reshape(-1))
                intermediate_losses[key] += loss_int.item()
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                  labels[:, 1:].reshape(-1))
            total_loss += loss.item()
    
    results = {'final': total_loss / len(loader)}
    if with_intermediate:
        for key in intermediate_losses:
            results[key] = intermediate_losses[key] / len(loader)
    return results



def train_with_intermediate_supervision(model, train_loader, val_loader, 
                                       epochs=10, lr=3e-4, device='cuda',
                                       intermediate_weight=0.3, grad_accum_steps=8,
                                       checkpoint_dir='checkpoints_deep',
                                       resume_from_checkpoint=None):
    """
    Train model with support for resuming from checkpoint
    
    Args:
        resume_from_checkpoint: Path to checkpoint to resume from (None for fresh start)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    
    total_steps = len(train_loader) // grad_accum_steps * epochs
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, 
                           pct_start=0.05, anneal_strategy='cos')
    
    # Initialize training state
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Resume from checkpoint if provided
    if resume_from_checkpoint:
        resume_info = load_checkpoint(
            resume_from_checkpoint, 
            model, 
            optimizer, 
            scheduler, 
            device
        )
        
        if resume_info:
            start_epoch = resume_info['start_epoch']
            train_losses = resume_info['train_losses']
            val_losses = resume_info['val_losses']
            best_val_loss = resume_info.get('best_val_loss', float('inf'))
            print(f"🔄 Resuming training from epoch {start_epoch}")
        else:
            print("⚠️  Could not load checkpoint, starting fresh")
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"\n🚀 Training Configuration:")
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Dataset: {len(train_loader.dataset):,} train, {len(val_loader.dataset):,} val")
    print(f"   Total Epochs: {epochs}")
    print(f"   Starting Epoch: {start_epoch}")
    print(f"   Remaining Epochs: {epochs - start_epoch}")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Grad accumulation: {grad_accum_steps}")
    print(f"   Effective batch: {train_loader.batch_size * grad_accum_steps}")
    print(f"   Mixed precision: {amp_dtype}")
    print(f"   Intermediate weight: {intermediate_weight:.1%}")
    if best_val_loss < float('inf'):
        print(f"   Best val loss (so far): {best_val_loss:.4f}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(device_type='cuda', dtype=amp_dtype):
                final_logits, intermediate_outputs = model(input_ids, return_all_outputs=True)
                loss_final = F.cross_entropy(final_logits[:, :-1].reshape(-1, final_logits.size(-1)),
                                            labels[:, 1:].reshape(-1))
                
                loss_intermediate = 0.0
                if len(intermediate_outputs) > 0:
                    for key, logits in intermediate_outputs.items():
                        loss_int = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                                   labels[:, 1:].reshape(-1))
                        loss_intermediate += loss_int
                    loss_intermediate = loss_intermediate / len(intermediate_outputs)
                
                loss = (1 - intermediate_weight) * loss_final + intermediate_weight * loss_intermediate
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix({'loss': f'{loss.item()*grad_accum_steps:.4f}',
                            'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        val_results = evaluate(model, val_loader, device, with_intermediate=True)
        train_loss = total_loss / len(train_loader)
        val_loss = val_results['final']
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f} (Perplexity: {np.exp(val_loss):.2f})")
        for key, loss_val in val_results.items():
            if key != 'final':
                print(f"   Val {key}: {loss_val:.4f} (PPL: {np.exp(loss_val):.2f})")
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"   ✅ Best model saved!")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        save_checkpoint(
            checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            train_losses,
            val_losses,
            best_val_loss,
            is_best
        )
        print(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        torch.cuda.empty_cache()
    
    return train_losses, val_losses



@torch.no_grad()
def test_early_exit(model, loader, device, thresholds=[0.85, 0.90, 0.95, 0.97, 0.99]):
    model.eval()
    original_ee = model.early_exit
    original_tau = model.ee_controller.tau0
    
    print("\n🔬 Testing Early Exit...")
    model.early_exit = False
    baseline_results = evaluate(model, loader, device, with_intermediate=True)
    baseline_loss = baseline_results['final']
    print(f"Baseline ({model.num_loops} loops): loss={baseline_loss:.4f}, ppl={np.exp(baseline_loss):.2f}")
    
    results = []
    for tau in thresholds:
        model.early_exit = True
        model.ee_controller.tau0 = tau
        model.ee_controller.tau_min = max(0.75, tau - 0.15)
        
        total_loops, total_tokens, total_loss = 0, 0, 0.0
        for batch in tqdm(loader, desc=f"  tau={tau:.2f}", leave=False):
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            logits, exited_at = model(input_ids, return_exit_info=True)
            total_loops += exited_at.sum().item()
            total_tokens += exited_at.numel()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                  labels[:, 1:].reshape(-1))
            total_loss += loss.item()
        
        avg_loops = total_loops / total_tokens
        avg_loss = total_loss / len(loader)
        quality_ok = avg_loss < (baseline_loss * 1.25)
        
        results.append({
            'threshold': tau,
            'avg_loops': avg_loops,
            'speedup': model.num_loops / avg_loops,
            'val_loss': avg_loss,
            'perplexity': np.exp(avg_loss),
            'quality_preserved': quality_ok,
            'degradation_pct': ((avg_loss - baseline_loss) / baseline_loss) * 100
        })
        
        status = "✅" if quality_ok else "❌"
        print(f"  {status} tau={tau:.2f}: {avg_loops:.2f} loops, {model.num_loops/avg_loops:.2f}x, "
              f"loss={avg_loss:.4f} ({results[-1]['degradation_pct']:+.1f}%)")
    
    model.early_exit = original_ee
    model.ee_controller.tau0 = original_tau
    
    valid = [r for r in results if r['quality_preserved']]
    best = min(valid, key=lambda x: x['avg_loops']) if valid else min(results, key=lambda x: x['val_loss'])
    return best, results, baseline_loss



def plot_results(train_losses, val_losses, ee_results, baseline_loss):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training curves
    ax = axes[0, 0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'o-', label='Train', linewidth=2)
    ax.plot(epochs, val_losses, 's-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training Progress (50K Samples)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if ee_results:
        thresholds = [r['threshold'] for r in ee_results]
        loops = [r['avg_loops'] for r in ee_results]
        speedups = [r['speedup'] for r in ee_results]
        losses = [r['val_loss'] for r in ee_results]
        
        ax = axes[0, 1]
        ax.plot(thresholds, loops, 'o-', linewidth=2, color='blue')
        ax.set_xlabel('Threshold', fontweight='bold')
        ax.set_ylabel('Avg Loops', fontweight='bold')
        ax.set_title('Early Exit: Loops', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(thresholds, speedups, 's-', linewidth=2, color='orange')
        ax.set_xlabel('Threshold', fontweight='bold')
        ax.set_ylabel('Speedup', fontweight='bold')
        ax.set_title('Early Exit: Speedup', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(thresholds, losses, '^-', linewidth=2, color='red')
        ax.axhline(y=baseline_loss, color='g', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlabel('Threshold', fontweight='bold')
        ax.set_ylabel('Val Loss', fontweight='bold')
        ax.set_title('Early Exit: Quality', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results_deep_50k.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Results saved to training_results_deep_50k.png")



# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CRT v2 Deep Model')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (default: checkpoints_deep/latest_checkpoint.pt if exists)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Total number of epochs to train (default: 10)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_deep',
                       help='Directory to save checkpoints (default: checkpoints_deep)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 CRT v2 DEEP: 50K SAMPLES + INCREASED DEPTH + RESUME SUPPORT")
    print("="*70)
    
    # Auto-detect checkpoint if --resume not specified
    if args.resume is None:
        default_checkpoint = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(default_checkpoint):
            print(f"\n🔍 Found existing checkpoint: {default_checkpoint}")
            user_input = input("Resume from this checkpoint? [Y/n]: ").strip().lower()
            if user_input != 'n':
                args.resume = default_checkpoint
                print("✅ Will resume from checkpoint")
            else:
                print("⚠️  Starting fresh training")
        else:
            print("\n📝 No existing checkpoint found, starting fresh training")
    
    print("\n📝 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n📚 Loading 50K samples...")
    print("💡 This is 2.5x larger than previous run")
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
    train_texts = [t for t in ds['train']['text'] if len(t) > 50][:50000]
    val_texts = [t for t in ds['validation']['text'] if len(t) > 50][:2000]
    print(f"✅ Train: {len(train_texts):,}, Val: {len(val_texts):,}")
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=384)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=384)
    
    print("\n🔧 Creating DataLoaders...")
    print("💡 Batch: 2, Grad Accum: 8 (effective: 16)")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, 
                            num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    
    print("\n🏗️  Building DEEP model...")
    print("=" * 70)
    print("DEPTH INCREASES:")
    print("  • num_loops: 6 → 8 (more recursive refinement)")
    print("  • trm_cycles: 6 → 8 (deeper convolution processing)")
    print("  • d_model: 512 (kept same for memory)")
    print("  • intermediate_loops: [2, 4] → [2, 4, 6] (more supervision)")
    print("=" * 70)
    
    model = CRTv2WithIntermediateHeads(
        vocab_size=50257,
        d_model=512,
        num_heads=8,
        trm_cycles=8,
        num_loops=8,
        kernel_size=3,
        norm_type='layer',
        dropout=0.1,
        use_trm_embeddings=True,
        embedding_trm_cycles=3,
        dilation_schedule=(1, 2, 4, 8),
        use_checkpointing=True,
        intermediate_loops=[2, 4, 6]
    )
    
    print(f"✅ Parameters: {model.count_parameters():,}")
    
    print("\n🚀 Starting training with resume support...")
    train_losses, val_losses = train_with_intermediate_supervision(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=3e-4,
        device=device,
        intermediate_weight=0.3,
        grad_accum_steps=8,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Final val loss: {val_losses[-1]:.4f}")
    print(f"   Final perplexity: {np.exp(val_losses[-1]):.2f}")
    
    print("\n" + "="*70)
    print("🔍 EARLY EXIT EVALUATION")
    print("="*70)
    
    best_ee, all_ee, baseline_loss = test_early_exit(model, val_loader, device)
    
    print(f"\n🏆 Best Config:")
    print(f"   Threshold: {best_ee['threshold']:.2f}")
    print(f"   Speedup: {best_ee['speedup']:.2f}x")
    print(f"   Degradation: {best_ee['degradation_pct']:+.1f}%")
    
    plot_results(train_losses, val_losses, all_ee, baseline_loss)
    
    results = {
        'training': {
            'dataset_size': 50000,
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'final_perplexity': float(np.exp(val_losses[-1]))
        },
        'model_config': {
            'parameters': model.count_parameters(),
            'd_model': 512,
            'num_loops': 8,
            'trm_cycles': 8,
            'intermediate_loops': [2, 4, 6]
        },
        'early_exit': {'best_config': best_ee, 'all_results': all_ee},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results_deep_50k.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': results['model_config'],
        'final_metrics': results['training']
    }, 'crt_v2_deep_50k.pt')
    
    print("\n" + "="*70)
    print("🎉 COMPLETE!")
    print("="*70)
    print(f"\n📊 Final perplexity: {np.exp(val_losses[-1]):.2f}")
    print(f"📁 Results: results_deep_50k.json")
    print(f"💾 Model: crt_v2_deep_50k.pt")
    print(f"💾 Checkpoint: {args.checkpoint_dir}/latest_checkpoint.pt")
