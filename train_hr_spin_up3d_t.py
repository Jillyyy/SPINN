from utils import TrainOptions
from train.trainer_hr_up3d_t import HSTrainer
import argparse

if __name__ == '__main__':
    options, cfg = TrainOptions().parse_args()
    trainer = HSTrainer(options, cfg)
    trainer.train()
