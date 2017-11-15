from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    config = get_config(mode='train')
    test_config = get_config(mode='test')
    print(config)
    train_loader = get_loader(config.video_root_dir, config.mode)
    test_loader = get_loader(test_config.video_root_dir, test_config.mode)
    solver = Solver(config, train_loader, test_loader)

    solver.build()
    solver.train()
