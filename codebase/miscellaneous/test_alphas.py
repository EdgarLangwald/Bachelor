import torch
from codebase.train import train_exhaustively
from codebase.model import Model
from codebase.utils import plot_loss

'''
if __name__ == '__main__':
    alphas = [0, 0.25, 0.5, 0.75, 1]

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH ALPHA = {alpha}")
        print(f"{'='*60}\n")

        train_exhaustively(
            batch_size=50,
            lr=5e-5,
            num_steps=10000,
            device="cuda",
            model=Model(360, 4, 4, 6, 1480, 0.1),
            print_every=100,
            model_path=f"alpha_{alpha}.pt",
            dataset_path="400Mb",
            accumulation_steps=2,
            num_rotations=1,
            num_workers=0,
            alpha=alpha,
            record_loss=100,
            ema_decay=0.98,
        )

        torch.cuda.empty_cache()
'''
plot_loss("alpha_0.pt_loss_history.pkl", detailed=True, plot_lr=False, title="Training Loss (alpha = 0)")
plot_loss("alpha_0.25.pt_loss_history.pkl", detailed=True, plot_lr=False, title="Training Loss (alpha = 0.25)")
plot_loss("alpha_0.5.pt_loss_history.pkl", detailed=True, plot_lr=False, title="Training Loss (alpha = 0.5)")
plot_loss("alpha_0.75.pt_loss_history.pkl", detailed=True, plot_lr=False, title="Training Loss (alpha = 0.75)")
plot_loss("alpha_1.pt_loss_history.pkl", detailed=True, plot_lr=False, title="Training Loss (alpha = 1)")