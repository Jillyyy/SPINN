from utils.train_options_origin import TrainOptions
from train import Trainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
