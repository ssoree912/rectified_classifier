import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch

from validate import validate
from dataset import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions

try:
    import wandb
except ImportError:
    wandb = None


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = "val"
    val_opt.jpg_method = ["pil"]
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def collect_real_fake_folders(split_root):
    real_folders, fake_folders = [], []
    for entry in sorted(os.listdir(split_root)):
        if entry.startswith("."):
            continue
        class_root = os.path.join(split_root, entry)
        if not os.path.isdir(class_root):
            continue
        real_dir = os.path.join(class_root, "0_real")
        fake_dir = os.path.join(class_root, "1_fake")
        if os.path.isdir(real_dir):
            real_folders.append(real_dir)
        if os.path.isdir(fake_dir):
            fake_folders.append(fake_dir)
    if not real_folders or not fake_folders:
        raise ValueError(f"No real/fake folders found under: {split_root}")
    return real_folders, fake_folders


def resolve_split_root(opt, attr_name, split_name):
    explicit = getattr(opt, attr_name)
    if explicit:
        return explicit
    return os.path.join(opt.dataset_root, split_name)


def resolve_wandb_mode(mode_arg: str) -> str:
    if mode_arg != "auto":
        return mode_arg
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def init_wandb(opt, train_root, val_root, steps_per_epoch):
    if not opt.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed in the current environment. Install it or re-run without --wandb.")

    wandb_dir = Path(opt.checkpoints_dir) / opt.name / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=opt.wandb_project,
        entity=opt.wandb_entity,
        name=opt.wandb_run_name or opt.name,
        mode=resolve_wandb_mode(opt.wandb_mode),
        dir=str(wandb_dir),
        config={
            "arch": opt.arch,
            "head_type": opt.head_type,
            "rectifier_mode": opt.rectifier_mode,
            "latent_kind": opt.latent_kind,
            "latent_view_mode": opt.latent_view_mode,
            "rectifier_ckpt": opt.rectifier_ckpt,
            "dataset_root": opt.dataset_root,
            "train_root": train_root,
            "val_root": val_root,
            "sr_cache_root": opt.sr_cache_root,
            "sr_cache_input_root": opt.sr_cache_input_root,
            "batch_size": opt.batch_size,
            "num_threads": opt.num_threads,
            "lr": opt.lr,
            "beta1": opt.beta1,
            "niter": opt.niter,
            "loss_freq": opt.loss_freq,
            "save_epoch_freq": opt.save_epoch_freq,
            "steps_per_epoch": steps_per_epoch,
            "fix_backbone": opt.fix_backbone,
        },
    )


def main():
    seed = 418
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    train_root = resolve_split_root(opt, "train_root", opt.train_split)
    val_root = resolve_split_root(opt, "val_root", opt.val_split)
    val_data_root = [val_root]

    print(f"Train root: {train_root}")
    print(f"Val root: {val_root}")

    real_folders, fake_folders = collect_real_fake_folders(train_root)
    data_loader = create_dataloader(opt, real_folders, fake_folders)

    model = Trainer(opt)

    val_loader_list = []
    for root in val_data_root:
        real_folders, fake_folders = collect_real_fake_folders(root)
        val_loader_list.append(create_dataloader(val_opt, real_folders, fake_folders))

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    wandb_run = init_wandb(opt, train_root=train_root, val_root=val_root, steps_per_epoch=len(data_loader))

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print("Length of data loader: %d" % (len(data_loader)))
    results_dict = {}
    try:
        for epoch in range(opt.niter):
            for _, data in enumerate(tqdm(data_loader)):
                model.total_steps += 1
                model.set_input(data)
                model.optimize_parameters()

                if model.total_steps % opt.loss_freq == 0:
                    loss_value = float(model.loss.detach().item() if hasattr(model.loss, "detach") else model.loss)
                    iter_time = (time.time() - start_time) / model.total_steps
                    print(f"Train loss: {loss_value} at step: {model.total_steps}")
                    train_writer.add_scalar("loss", loss_value, model.total_steps)
                    print("Iter time: ", iter_time)
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "train/loss": loss_value,
                                "train/global_step": model.total_steps,
                                "train/iter_time": iter_time,
                                "train/epoch": epoch + 1,
                                "train/lr": model.optimizer.param_groups[0]["lr"],
                            },
                            step=model.total_steps,
                        )

            if epoch % opt.save_epoch_freq == 0:
                print("saving the model at the end of epoch %d" % (epoch))
                model.save_networks("model_epoch_best.pth")
                model.save_networks("model_epoch_%s.pth" % epoch)

            model.eval()
            acc_list = []
            ap_list = []
            b_acc_list = []
            threshold_list = []
            y_pred_list = []
            y_true_list = []
            for i, val_loader in enumerate(val_loader_list):
                ap, r_acc0, f_acc0, acc, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true = validate(model.model, val_loader, find_thres=True)
                acc_list.append(acc)
                ap_list.append(ap)
                b_acc_list.append(acc1)
                threshold_list.append(best_thres)
                val_writer.add_scalar("accuracy", acc, model.total_steps)
                val_writer.add_scalar("ap", ap, model.total_steps)
                print(f"(Val on {val_data_root[i]} @ epoch {epoch}) acc: {acc}; ap: {ap};r_acc0:{r_acc0}, f_acc0:{f_acc0}, r_acc1:{r_acc1}, f_acc1:{f_acc1}, acc1:{acc1}, best_thres:{best_thres}")
                if wandb_run is not None:
                    ds_name = Path(val_data_root[i]).name
                    wandb.log(
                        {
                            f"val/{ds_name}/ap": ap,
                            f"val/{ds_name}/acc": acc,
                            f"val/{ds_name}/best_acc": acc1,
                            f"val/{ds_name}/threshold": best_thres,
                            "train/epoch": epoch + 1,
                        },
                        step=model.total_steps,
                    )
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)

            ap_list.append(sum(ap_list) / len(val_loader_list))
            acc_list.append(sum(acc_list) / len(val_loader_list))
            b_acc_list.append(sum(b_acc_list) / len(val_loader_list))
            threshold_list.append(sum(threshold_list) / len(val_data_root))
            results_dict[f"epoch_{epoch}_ap"] = ap_list
            results_dict[f"epoch_{epoch}_acc"] = acc_list
            results_dict[f"epoch_{epoch}_b_acc"] = b_acc_list
            results_dict[f"epoch_{epoch}_b_threshold"] = threshold_list
            results_df = pd.DataFrame(results_dict)
            results_df.to_excel(os.path.join(opt.checkpoints_dir, opt.name, "results.xlsx"), sheet_name="sheet1", index=False)
            print(f"(average Val on all dataset @ epoch {epoch}) acc: {acc_list[-1]}; ap: {ap_list[-1]}")
            np.savez(os.path.join(opt.checkpoints_dir, opt.name, f"y_pred_eval_{epoch}.npz"), *y_pred_list)
            np.savez(os.path.join(opt.checkpoints_dir, opt.name, f"y_true_eval_{epoch}.npz"), *y_true_list)
            if wandb_run is not None:
                wandb.log(
                    {
                        "val/avg_ap": ap_list[-1],
                        "val/avg_acc": acc_list[-1],
                        "val/avg_best_acc": b_acc_list[-1],
                        "val/avg_threshold": threshold_list[-1],
                        "train/epoch": epoch + 1,
                    },
                    step=model.total_steps,
                )

            acc = acc_list[-1]
            early_stopping(acc, model)
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("Early stopping.")
                    break
            model.train()
    finally:
        train_writer.close()
        val_writer.close()
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
