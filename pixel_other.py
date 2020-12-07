import gym
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


# Architektura sieci DQNN
def deep_q_network(alpha, n_actions, input_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=(*input_dims,),
                     data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions))
    model.compile(optimizer=Adam(lr=alpha), loss='mse')
    return model


# Pamiec agenta (wspomnienia)
class Memory(object):
    def __init__(self, max_size, input_shape):
        self.memory_size, self.memory_counter = max_size, 0
        self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.done_memory = np.zeros(self.memory_size, dtype=np.uint8)

    def remember(self, state, action, reward, new_state, done):
        i = self.memory_counter % self.memory_size
        self.state_memory[i] = state
        self.new_state_memory[i] = new_state
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.done_memory[i] = done
        self.memory_counter += 1

    def sample(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, new_states, dones


# Agent
class DeepQAgent(object):
    def __init__(self, alpha, gamma, n_actions, memory_size, batch_size, input_dims, target_step=4000, eps=0.01,
                 eps_dec=0, eps_min=0, main_name='main.h5', target_name='target.h5'):
        self.main_network = deep_q_network(alpha, n_actions, input_dims)
        self.target_network = deep_q_network(alpha, n_actions, input_dims)
        self.gamma = gamma
        self.action_space = [i for i in range(n_actions)]
        self.memory = Memory(memory_size, input_dims)
        self.batch_size = batch_size
        self.target_step = target_step
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.main_name = main_name
        self.target_name = target_name
        self.step = 0

    def choose(self, observation):
        if np.random.random() < self.eps:  # eksploracja
            choice = np.random.choice(self.action_space)
        else:  # eksploatacja
            state = np.array([observation], copy=False, dtype=np.float32)
            observation = observation[np.newaxis, :]
            qs = self.main_network.predict(observation)
            choice = np.argmax(qs)
        return choice

    # Zapis i wczytywanie sieci
    def save(self):
        self.main_network.save(self.main_name)
        self.target_network.save(self.target_name)

    def load(self):
        self.main_network = load_model(self.main_name)
        self.target_network = load_model(self.target_name)

    # Uczenie agenta na porcji danych z pamieci
    def learn(self):
        if self.memory.memory_counter > self.batch_size:
            self.step += 1

            if self.eps > self.eps_min:
                self.eps = self.eps - self.eps_dec
            else:
                self.eps = self.eps_min

            state, action, reward, new_state, done = self.memory.sample(self.batch_size)
            self.replace_target_network()

            q_next = self.target_network.predict(new_state)
            q_eval = self.main_network.predict(new_state)
            q_pred = self.main_network.predict(state)
            max_actions = np.argmax(q_eval, axis=1)
            q_target = q_pred

            i = np.arange(self.batch_size)
            q_target[i, action] = reward + self.gamma * q_next[i, max_actions.astype(int)] * (1 - done)
            self.main_network.train_on_batch(state, q_target)

    def replace_target_network(self):
        if self.step % self.target_step == 0:  # update wag sieci "target"
            self.target_network.set_weights(self.main_network.get_weights())

    def remember(self, state, action, reward, new_state, done):
        self.memory.remember(state, action, reward, new_state, done)


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, game_type, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.game_type = game_type
        if self.game_type == 'gym':
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 80, 1), dtype=np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs, self)

    @staticmethod
    def process(frame, self):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299 * new_frame[:, :, 0] + 0.587 * new_frame[:, :, 1] + 0.114 * new_frame[:, :, 2]
        if self.game_type == 'gym':
            new_frame = new_frame[35:195:2, ::2].reshape(80, 80, 1)
        return new_frame.astype(np.uint8)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_space.shape[-1],
                                                                          self.observation_space.shape[0],
                                                                          self.observation_space.shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class SkipEnv(gym.Wrapper):
    def __init__(self, skip, env=None):
        super(SkipEnv, self).__init__(env)
        self.skip = skip

    def step(self, action):
        obs, score, done, info = 0, 0, False, 0
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            score += reward
            if done:
                break
        return obs, score, done, info


def env_make(game, version, buffer_size):
    env = gym.make(game)
    env = SkipEnv(buffer_size, env)
    env = PreProcessFrame(version, env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, buffer_size)
    return ScaleFrame(env)
