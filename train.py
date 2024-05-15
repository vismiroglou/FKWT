"""Script for training KWT model"""
from argparse import ArgumentParser
import torch
import wandb

from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from utils.config_parser import parse_config
from utils.trainer import LightningKWT
from src.dataset import SpeechCommands
    
def training_pipeline(config, logger, model, train_loader, val_loader):
    
    # Create callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", verbose=True)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=config['hparams']['early_stopping_patience'], verbose=True)
    callbacks = [model_checkpoint, early_stopping]

    trainer = L.Trainer(devices=1, 
                        accelerator='gpu', 
                        max_epochs=config['hparams']['n_epochs'], 
                        logger=logger,
                        callbacks=callbacks,
                        log_every_n_steps=50,
                        #strategy='ddp_find_unused_parameters_true',
                        default_root_dir=config['exp']['save_dir'])

    trainer.fit(model, train_loader, val_loader)


def get_model(ckpt, config, useFNet=False):

    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if ckpt:
        print('Loading from checkpoint')
        model = LightningKWT.load_from_checkpoint(ckpt, config=config, useFnet=useFNet)
    elif useFNet:
        model = LightningKWT(config, True)
    else:
        model = LightningKWT(config)
    model.to(device)
    return model


def get_dataloaders(config):
    # Make datasets

    train_set = SpeechCommands(root=config['dataset_root'], 
                               audio_config=config['audio_config'], 
                               labels_map=config['labels_map'], 
                               subset='training',
                               augment=True)
    val_set = SpeechCommands(root=config['dataset_root'], 
                             audio_config=config['audio_config'], 
                             labels_map=config['labels_map'], 
                             subset='validation')
    if config['dev_mode']:
        train_set._walker = train_set._walker[:1000]
        val_set._walker = val_set._walker[:1000]
        print(train_set.__len__())

    # Make dataloaders - added shuffle to train_loader
    train_loader = DataLoader(train_set, batch_size=config['hparams']['batch_size'], shuffle=True, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=config['hparams']['batch_size'], num_workers=5)
    
    return train_loader, val_loader


def main(args):

    config = parse_config(args.config)
    if args.dataset_root:
        config['dataset_root'] = args.dataset_root
    if args.labels_map:
        config['labels_map'] = args.labels_map
    config['dev_mode'] = args.dev_mode
    
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id
    
    if config["exp"]["wandb"]:
        wandb.login()
        logger = WandbLogger(project=config["exp"]["proj_name"],
                                name=config["exp"]["exp_name"],
                                entity=config["exp"]["entity"],
                                config=config["hparams"],
                                log_model=True,
                                save_dir=config["exp"]["save_dir"])
    else:
        logger = None
    
    seed_everything(config['hparams']['seed'])
    model = get_model(args.ckpt_path, config, args.useFNet)
    train_loader, val_loader = get_dataloaders(config)
    training_pipeline(config, logger, model, train_loader, val_loader)


if __name__ == '__main__':
    from argparse import ArgumentParser

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
            
    ap = ArgumentParser("Driver code")
    ap.add_argument('--config', type=str, required=True, help='Path to configuration file')
    ap.add_argument('--dataset_root', type=str, help='Dataset root directory')
    ap.add_argument('--labels_map', type=str, help='Path to lbl_map.json')
    ap.add_argument('--ckpt_path', type=str, help='Path to model checkpoint.')
    ap.add_argument('--useFNet', type=bool, default=False)
    ap.add_argument('--id', type=str, help='Unique experiment identifier')
    ap.add_argument('--dev_mode', action='store_true', help='Flag to limit the dataset for testing purposes.')
    args = ap.parse_args()

    main(args)