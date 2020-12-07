import numpy as np
from pixel_other import DeepQAgent, env_make
from utils import plot_learning
import pandas
import random

EPISODES = 50000

# Hiperparametry
LEARNING_RATE = 1e-4
DISCOUNTING_RATE = 0.98
MEMORY_SIZE = 40000
BATCH_SIZE = 64
EPS = 1
EPS_DEC = 5e-6
EPS_MIN = 0.08

NO_OP_MAX = 30


# Inicjalizacja srodowiska i agenta
env = env_make('BeamRider-v4', 'gym', 4)
agent = DeepQAgent(alpha=LEARNING_RATE, gamma=DISCOUNTING_RATE, n_actions=env.action_space.n, memory_size=MEMORY_SIZE,
                   batch_size=BATCH_SIZE, input_dims=(4, 80, 80), eps=EPS, eps_dec=EPS_DEC, eps_min=EPS_MIN,
                   main_name="pixel_main.h5", target_name="pixel_target.h5")


scores = []
steps = []
epsilons = []
average_score = 0
best_score = 0
n_steps = 0

for episode in range(EPISODES):
    done = False
    ep_step = 0
    no_op_end = random.randrange(1, NO_OP_MAX+1)
    state = env.reset()
    score = 0

    while not done:
        if ep_step == 0:
            choice = 4
        elif ep_step < no_op_end:
            choice = 0
        else:
            choice = agent.choose(state)
        new_state, reward, done, _ = env.step(choice)
        score += reward
        agent.remember(state, choice, reward, new_state, int(done))
        agent.learn()
        state = new_state
        n_steps += 1
        ep_step += 1

    scores.append(score)
    steps.append(episode)
    epsilons.append(agent.eps)
    average_score = np.mean(scores[-40:])
    print('Episode:', episode, ' Score: %.1f' % score, ' Average: %.1f' % average_score,
          ' Explore probability: %.2f' % agent.eps, ' Best average: %.2f' % best_score)

    if average_score > best_score:
        print('Best average score %.1f!' % average_score)
        best_score = average_score
        agent.save()
        plot_learning(steps, scores, epsilons, "pixel_wykres")
        df = pandas.DataFrame(data={"step": steps, "score": scores, "epsilon": epsilons})
        df.to_csv("pixel_log.csv", sep=';', index=False)

    if episode % 50 == 0:
        plot_learning(steps, scores, epsilons, "pixel_wykres")
        df = pandas.DataFrame(data={"step": steps, "score": scores, "epsilon": epsilons})
        df.to_csv("pixel_log.csv", sep=';', index=False)

env.close()
print('Best score overall %.1f' % np.max(scores))
