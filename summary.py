import numpy as np
import config

sum_cc = 0.0
for game in config.game_dic:
    try:
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
    except Exception as e:
        print('load {} failed'.format(
            game
        ))

print('Avg|{}'.format(
    sum_cc/len(config.game_dic)
))
