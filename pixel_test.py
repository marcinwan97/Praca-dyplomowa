import time
import numpy as np
from pixel_other import DeepQAgent, env_make

EPISODES = 200
RENDERING = False
FPS_REDUCTION = 0.03

# Hiperparametry
LEARNING_RATE = 1e-4
DISCOUNTING_RATE = 0.98
MEMORY_SIZE = 40000
BATCH_SIZE = 64


# Inicjalizacja srodowiska i agenta
env = env_make('BeamRider-v4', 'gym', 4)

agent = DeepQAgent(alpha=LEARNING_RATE, gamma=DISCOUNTING_RATE, n_actions=env.action_space.n, memory_size=MEMORY_SIZE,
                   batch_size=BATCH_SIZE, input_dims=(4, 80, 80),
                   main_name="pixel_main.h5", target_name="pixel_target.h5")

agent.load()

scores = []

for episode in range(EPISODES):
    done = False
    state = env.reset()
    score = 0

    while not done:
        choice = agent.choose(state)
        new_state, reward, done, _ = env.step(choice)
        score += reward
        state = new_state
        if RENDERING:
            env.render()
            time.sleep(FPS_REDUCTION)

    scores.append(score)

    print('Episode:', episode, ' Score:', int(score))


average_score = np.mean(scores)
median = np.median(scores)
best = np.max(scores)
print(average_score, median, best)
env.close()
