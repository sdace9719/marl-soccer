from pettingzoo.test.parallel_test import parallel_api_test
from soccer_env import soccerenv


def main():
    env = soccerenv(render_mode=None)
    parallel_api_test(env, num_cycles=50)


if __name__ == "__main__":
    main()


