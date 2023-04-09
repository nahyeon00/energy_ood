import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoConfig


from model import *
from mix_data import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                       type=int,
                       default=0)
    parser.add_argument('--data_path',
                        type=str,
                        default='/workspace/energy/data/',
                        help='where to prepare data')
    parser.add_argument("--dataset", 
                        default='MixSNIPS_clean', 
                        type=str, 
                        help="The name of the dataset to train selected")
    parser.add_argument("--known_cls_ratio",
                        default=0.5,
                        type=float,
                        help="The number of known classes")
    parser.add_argument("--labeled_ratio",
                        default=1.0,
                        type=float,
                        help="The ratio of labeled samples in the training set")
    parser.add_argument('--max_epoch',
                       type=int,
                        default=10,
                       help='maximum number of epochs to train')
    parser.add_argument('--num_gpus',
                       type=int,
                       default=1,
                       help='number of available gpus')
    parser.add_argument('--ckpt_path',
                       type=str,
                       default='/workspace/energy',
                       help='checkpoint file path')
    parser.add_argument('--model_save_path',
                       type=str,
                       default='checkpoints',
                       help='where to save checkpoint files')
    parser.add_argument('--max_seq_len',
                       type=int,
                       default=70,
                       help='maximum length of input sequence data')
    parser.add_argument('--batch_size',
                       type=int,
                       default=128,
                       help='batch size')
    parser.add_argument('--device',
                       type=int,
                       default=0,
                       help='device')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='num of worker for dataloader')
    parser.add_argument('--output_dir',
                       type=str,
                       default='/workspace/energy/results/',
                       help='output dir')
    parser.add_argument("--results_file_name", 
                        type=str, 
                        default = 'results.csv', 
                        help="The file name of all the results.")
    parser.add_argument("--num_labels", 
                        type=int, 
                        default = 0, 
                        help="known label list + unseen label")
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    seed_everything(args.seed, workers=True)

    dm = MixDataModule(args = args)

    dm.setup('fit')

    # config = AutoConfig.from_pretrained('bert-base-uncased')
    model = en_model(args)


    # tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dir_path, 'tb_logs'))
    # lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epoch,
        accelerator="gpu",
        devices=[0],
        auto_select_gpus=True
    )
    
    # train
    # trainer.fit(model, dm)

    # predict to calculate centroids
    model.freeze()
    model.eval()
    trainer.test(model, dm)


if __name__ == '__main__':
    main()