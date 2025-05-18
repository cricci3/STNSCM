import copy
import time
import torch
import numpy as np
from tqdm import tqdm
import os
import psutil

from model.tester import model_val, model_test

def baseline_train(runid, model, model_name, dataloader, static_norm_adjs, device, logger, cfg, simple_model):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Device used: {DEVICE}")

    # Define save path for checkpoints
    save_path = os.path.join(cfg['save'], cfg['model_name'], cfg['data']['freq'], 'ckpt')
    os.makedirs(save_path, exist_ok=True)
    
    # Define the final models folder and create it if it doesn't exist
    final_model_path = os.path.join(cfg['save'], "final_models")
    os.makedirs(final_model_path, exist_ok=True)

    # Define resource log file
    resource_log_path = os.path.join(cfg['save'], f"resource_usage_run{runid}.log")

    scaler = dataloader['scaler']

    if 'dummy' in cfg['model_name'].lower() or 'agcrn' in cfg['model_name'].lower():
        from model.helper import SimpleTrainer as Trainer
        engine = Trainer(
            model=model,
            lr=cfg['train']['base_lr'],
            weight_decay=cfg['train']['weight_decay'],
            loss_type=cfg['model']['loss_type'],
            scaler=scaler,
            device=device
        )
    else:
        from model.helper import Trainer
        engine = Trainer(
            model,
            base_lr=cfg['train']['base_lr'],
            weight_decay=cfg['train']['weight_decay'],
            milestones=cfg['train']['milestones'],
            lr_decay_ratio=cfg['train']['lr_decay_ratio'],
            min_learning_rate=cfg['train']['min_learning_rate'],
            max_grad_norm=cfg['train']['max_grad_norm'],
            cl_decay_steps=cfg['train']['cl_decay_steps'],
            num_for_target=cfg['data']['num_for_target'],
            num_for_predict=cfg['data']['num_for_predict'],
            loss_type=cfg['model']['loss_type'],
            scaler=scaler,
            device=device,
            curriculum_learning=cfg['train']['use_curriculum_learning'],
            new_training=cfg['train']['new_training'],
        )

    # Setup training loop
    begin_epoch = cfg['train']['epoch_start']
    epochs = cfg['train']['epochs']
    tolerance = cfg['train']['tolerance']
    print_every = cfg['train']['print_every']

    best_val_loss = float('inf')
    stable_count = 0
    best_model = None
    global_step = 0
    his_loss = []
    train_time, val_time = [], []

    torch.cuda.reset_peak_memory_stats(device=device) if torch.cuda.is_available() else None
    process = psutil.Process(os.getpid())

    for epoch in range(begin_epoch, begin_epoch + epochs + 1):
        train_loss, train_mae, train_mape, train_rmse = [], [], [], []
        t1 = time.time()
        train_loader = dataloader['train']

        for _, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            if len(batch) >= 6:
                x, x_time, target, target_time, pos, target_cl = batch
            else:
                x, target = batch
                x_time = target_time = pos = target_cl = None

            x = x.to(engine.device)
            target = target.to(engine.device)
            if x_time is not None:
                x_time = x_time.to(engine.device)
            if target_time is not None:
                target_time = target_time.to(engine.device)
            if target_cl is not None:
                target_cl = target_cl.to(engine.device)

            if simple_model:
                metrics = engine.train(input=x, target=target)
            else:
                metrics = engine.train(input=x, input_time=x_time, target=target, target_time=target_time, target_cl=target_cl)

            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            global_step += 1
        
        if simple_model:
            output = engine.model(x)
        else:
            output = engine.model(x, x_time, target_time, target_cl=target_cl, task_level=2, global_step=global_step)

        print("Output stats - min:", output.min().item(), 
                "max:", output.max().item(), 
                "mean:", output.mean().item())
        
        print("Target stats - min:", target.min().item(), 
            "max:", target.max().item(), 
            "mean:", target.mean().item(),
            "std:", target.std().item())

        t2 = time.time()
        train_time.append(t2 - t1)

        # VALIDATION
        s1 = time.time()
        valid_loss, valid_mae, valid_mape, valid_rmse, _ = model_val(
            runid, engine, dataloader, device, logger, epoch
        )
        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mvalid_loss = np.mean(valid_loss)

        logger.info(f"Epoch {epoch:03d}, Train Loss: {mtrain_loss:.4f}, Valid Loss: {mvalid_loss:.4f}")

        his_loss.append(mvalid_loss)
        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            stable_count = 0
            best_model = copy.deepcopy(engine.model.state_dict())
            ckpt_name = f"exp{model_name}_no_od_matrix_epoch{epoch}_ValLoss-{mvalid_loss:.4f}.pth"
            best_path = os.path.join(save_path, ckpt_name)
            torch.save({'model_state_dict': best_model}, best_path)
            logger.info(f"Better model saved: {best_path}")
        else:
            stable_count += 1
            logger.info(f"No improvement ({stable_count}/{tolerance})")
            if stable_count >= tolerance:
                logger.info("Early stopping triggered.")
                if best_model is None:
                    logger.warning("No best model available! Using final model.")
                    best_model = copy.deepcopy(engine.model.state_dict())
                break

    # LOAD BEST MODEL
    if best_model is not None:
        engine.model.load_state_dict(best_model)
        logger.info("Loaded best model for final evaluation.")
    else:
        logger.warning("No best model available! Using final trained model.")

    # Save the best model as the model name in the 'final_models' folder
    if best_model is not None:
        final_model_save_path = os.path.join(final_model_path, f"{model_name}_no_od_matrix_best_model.pth")
        torch.save({'model_state_dict': best_model}, final_model_save_path)
        logger.info(f"Best model saved as: {final_model_save_path}")
    else:
        logger.warning("No best model found to save in 'final_models'.")

    logger.info("Training completed. Evaluating on test set...")

    valid_loss, valid_mae, valid_mape, valid_rmse, _ = model_val(
        runid, engine, dataloader, device, logger, epoch
    )
    test_loss, test_mae, test_mape, test_rmse, _ = model_test(
        runid, engine, dataloader, device, logger, cfg
    )

    # === LOG RESOURCE USAGE ===
    total_train_time = sum(train_time)
    total_val_time = sum(val_time)
    peak_gpu_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3) if torch.cuda.is_available() else 0
    peak_cpu_mem = process.memory_info().rss / (1024 ** 3)
    total_params = sum(p.numel() for p in model.parameters())

    with open(resource_log_path, 'w') as f:
        f.write(f"Run ID: {runid}\n")
        f.write(f"Device used: {DEVICE}\n")
        f.write(f"Total training time: {total_train_time / 3600:.2f} hours\n")
        f.write(f"Total validation time: {total_val_time / 3600:.2f} hours\n")
        f.write(f"Peak GPU memory usage: {peak_gpu_mem:.2f} GB\n")
        f.write(f"Peak CPU memory usage: {peak_cpu_mem:.2f} GB\n")
        f.write(f"Total model parameters: {total_params}\n")

    logger.info(f"Resource usage logged in {resource_log_path}")

    return valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse
