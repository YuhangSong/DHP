import numpy as np
import config

sum_cc = 0.0
for game in config.game_dic:
    cc = np.load(
        '{}/cc/{}/all.npy'.format(
            config.log_dir,
            game,
        ),
    )
    sum_cc += cc
    print('{}|cc|{}'.format(
        game,
        cc,
    ))
print('Avg|{}'.format(
    sum_cc/len(config.game_dic)
))
