import numpy as np
from PIL import Image
import cv2
from collections import deque
from tqdm import tqdm


class Env:
    def __init__(self, *, get_observation, get_action, take_action, \
        get_start_state, display_env, display=True, call_every=None, **kwargs):
        self._init_kwargs = kwargs
        self.get_observation = get_observation # takes in state
        self.get_action = get_action    # takes in input
        self.take_action = take_action  # takes in state, action (int)
                                        # returns new_state, reward, done
        self.get_start_state = get_start_state
        self.display_env = display_env  # Takes in state, returns numpy array of (0-255,0-255,0-255)
        self.display = display
        self.call_every = call_every    # Optional, dict of methods called every N episodes
                                        # for example {10: lambda env: print(33)} prints 33 every 10 episodes
                                        # The methods get passed a keyword argument env that holds this Env object
        self.reset()

    @staticmethod
    def copy(env):
        new_env = Env(
            get_observation=env.get_observation,
            get_action=env.get_action,
            take_action=env.take_action,
            get_start_state=env.get_start_state,
            display_env=env.display_env,
            display=env.display,
            call_every=env.call_every,
            **env._init_kwargs
        )
        new_env.reset()
        return new_env

    def reset(self):
        self.episodes = self._init_kwargs.get('episodes', 100)
        self.steps_per_ep = self._init_kwargs.get('steps_per_ep', 200)
        self.show_every = self._init_kwargs.get('show_every', 200)
        self.stat_every = self._init_kwargs.get('stat_every', 50)
        self.printing = self._init_kwargs.get('printing', True)
        self.framedelay = self._init_kwargs.get('framedelay', 500)
        self.track_rewards = self._init_kwargs.get('track_rewards', True)
        self.track_aggr_rewards = self._init_kwargs.get('track_aggr_rewards', True)
        if self.track_aggr_rewards and not self.track_rewards:
            self.track_aggr_rewards = False
        self.track_steps_taken = self._init_kwargs.get('track_steps_taken', False)
    
    def run(self):
        if self.track_rewards:
            self.ep_rewards = [0 for _ in range(self.episodes)]
        if self.track_aggr_rewards:
            self.aggr_ep_rewards = {
                'avg': [],
                'min': [],
                'max': []
            }
        if self.track_steps_taken:
            self.ep_steps = [0 for _ in range(self.episodes)]

        for ep in range(1, self.episodes + 1):
            episode_reward = 0
            state = self.get_start_state()
            for step in range(self.steps_per_ep):
                # Get inputs ("observation")
                obs = self.get_observation(state)
                action = self.get_action(obs)
                
                new_state, reward, done = self.take_action(state, action)

                if self.display and ep % self.show_every == 0:
                    # Display env
                    pixelarray = self.display_env(new_state)

                    img = Image.fromarray(pixelarray,mode="RGB")
                    img = img.resize((300,300))
                    cv2.imshow("",np.array(img))
                    if done:
                        if cv2.waitKey(1500) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break
                    else:
                        if cv2.waitKey(self.framedelay)& 0xFF == ord('q'):
                            break
                    
                episode_reward += reward
                if done:
                    break
            
            if self.track_rewards:
                self.ep_rewards[ep-1] = episode_reward
            if ep % self.stat_every == 0 and self.track_aggr_rewards:
                eps = self.ep_rewards[-self.stat_every:]
                avg, epmin, epmax = sum(eps)/len(eps), min(eps), max(eps)
                self.aggr_ep_rewards['avg'].append(avg)
                self.aggr_ep_rewards['min'].append(epmin)
                self.aggr_ep_rewards['max'].append(epmax)
                if self.printing:
                    print(f'Episode {ep}/{self.episodes}')
                    print(f'Reward statistics of the past {self.stat_every} episodes:')
                    print(f'Average: {avg}, Minimum: {epmin}, Maximum: {epmax}')
            
            if self.track_steps_taken:
                self.ep_steps[ep-1] = step

            if self.call_every is not None:
                for (k,v) in self.call_every.items():
                    if k % ep == 0:
                        if self.printing:
                            print(f'Calling {v} on episode {ep}')
                        v(self)

        return {
            'rewards': self.ep_rewards if self.track_rewards else None,
            'sumrewards': sum(self.ep_rewards) if self.track_rewards else None,
            'aggr': self.aggr_ep_rewards if self.track_aggr_rewards else None,
            'steps': self.ep_steps if self.track_steps_taken else None
        }
    
    def help(self):
        helpstring = """
These are the main constructor arguments:

get_observation     - takes in state, returns observation
get_action          - takes in observation, returns action
take_action         - takes in (state, action), returns (new_state, reward, done)
get_start_state     - returns state
display_env         - takes in state, returns (width,height,3) numpy array
display             - optional (default True), whether to display occassionaly (every show_every episodes) or not
call_every          - optional (default None), dictionary of methods called every N episodes
                      keys are numbers (key 4 means call every 4 episodes)
                      values are methods taking the env as a parameter

These are the optional keyword arguments you can provide in the constructor:
episodes            - Number of episodes to run when you call run(). Default is 100
steps_per_ep        - How many steps per episode. Default is 200
show_every          - Display the environment every X episodes. Default is 200
stat_every          - Count aggregate stats every X episodes. Default is 50
printing            - Whether to print information while running, like aggregate stats. Default is True
framedelay          - Delay between frames when displaying. Default is 500
track_rewards       - Whether to track and return episode rewards. Default is True
track_aggr_rewards  - Whether to track aggregate rewards (max, min, avg). Default is True
track_steps_taken   - Whether to track how many steps each episode took. Default is False

Empty method bodies and example constructor below:
def get_observation(state):
    return obs

def get_action(obs):
    return action

def take_action(state,action):
    return new_state,reward,done

def get_start_state():
    return state

def display_env(state):
    pixelarray = np.zeros((width,height,3), dtype=np.uint8)
    # Fill out certain pixels based on the state
    return pixelarray

env = Env(
    get_observation=get_observation,
    get_action=get_action,
    take_action=take_action,
    get_start_state=get_start_state,
    display_env=display_env,
    episodes=50,
    steps_per_ep=150
)
        """
        print(helpstring)


class Block:
    '''
    Basic class for objects on a 2D grid
    '''
    def __init__(self, size_x, size_y):
        self.SIZE_X = size_x
        self.SIZE_Y = size_y
        self.x = np.random.randint(0, self.SIZE_X)
        self.y = np.random.randint(0, self.SIZE_Y)
    
    def __str__(self):
        return f"{self.x},{self.y}"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def action(self, choice):
        if choice == 0:
            self.move(x=0, y=1)
        if choice == 1:
            self.move(x=0, y=-1)
        if choice == 2:
            self.move(x=1, y=0)
        if choice == 3:
            self.move(x=-1, y=0)
    
    def move(self, x=False, y=False):
        if x is False:
            self.x += 1 if np.random.randint(0,2) == 0 else -1
        else:
            self.x += x
        if y is False:
            self.y += 1 if np.random.randint(0,2) == 0 else -1
        else:
            self.y += y
        
        if self.x < 0:
            self.x = 0
        elif self.x >= self.SIZE_X:
            self.x = self.SIZE_X - 1
        if self.y < 0:
            self.y = 0
        elif self.y >= self.SIZE_Y:
            self.y = self.SIZE_Y - 1

class Snake:
    '''
    Class with a snake functionality on a 2D grid
    - Comes with a pre-built get_observation method, which gives distances
      in the 8 directions to the walls, body, and food
    - has a .dead attribute you can use when determining if the episode is done
    - has a .food attribute that holds the (x,y) position of the food,
      it's set to None when eaten, and you can generate it again by callling .generate_food()
    '''
    def __init__(self, size_x, size_y):
        self.SIZE_X = size_x
        self.SIZE_Y = size_y
        self.x = np.random.randint(0, self.SIZE_X)
        self.y = np.random.randint(0, self.SIZE_Y)
        self.body = deque([(self.x,self.y)],maxlen=size_x*size_y) # list of (x,y) tuples
        self.dead = False # set to true if hit a wall / its body
        self.generate_food()

    def action(self, choice):
        if self.dead:
            raise Exception('Can\'t take an action when dead')
        if choice == 0:
            self.move(x=0, y=-1)
        if choice == 1:
            self.move(x=1, y=0)
        if choice == 2:
            self.move(x=0, y=1)
        if choice == 3:
            self.move(x=-1, y=0)
    
    def move(self, x=False, y=False):
        self.x += x
        self.y += y
        if not ((self.x,self.y) == self.food):
            self.body.pop() # Unless you eat, destroy the last part of the body
        else:
            self.food = None
        
        if not (0 <= self.x < self.SIZE_X) or not (0 <= self.y < self.SIZE_Y) or self._in_body(self.x,self.y):
            # Hit the wall or itself
            self.dead = True
        # Finally add new spot to body (only now to make the index() check work)
        self.body.appendleft((self.x,self.y))
    
    def _in_body(self,x,y):
        # Checks whether the snake ran into its own body
        return (x,y) in self.body
    
    @property
    def is_maxed_out(self):
        return len(self.body) == self.SIZE_X*self.SIZE_Y

    def generate_food(self):
        choices = [(x,y) for x in range(0,self.SIZE_X) for y in range(0,self.SIZE_Y) \
            if not self._in_body(x,y)]
        self.food = choices[np.random.randint(len(choices))]
    
    def get_observation(self):
        obs = np.zeros((1,24,))
        # by 8s: distances to walls, distances to body, distances to food
        # from top left clockwise
        xdif,ydif = self.SIZE_X-self.x-1, self.SIZE_Y-self.y-1
        obs[0,:8] += [ # distances to the walls
            min(self.x,self.y), # top left
            self.y,             # top
            min(xdif,self.y),   # top right
            xdif,               # right
            min(xdif,ydif),     # bottom right
            ydif,               # bottom
            min(self.x,ydif),   # bottom left
            self.x              # left
        ]
        obs[0,:8] += 1 # so that when you're right next to it, it should be a 1 not a 0
        body = [0 for _ in range(8)]
        food = [0 for _ in range(8)]
        moves = [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]
        for i in range(8):
            m = moves[i]
            x = self.x + m[0]
            y = self.y + m[1]
            found_food, found_body = False, False
            counter = 1
            while 0 <= x < self.SIZE_X and 0 <= y < self.SIZE_Y and not (found_body and found_food):
                if self._in_body(x,y) and not found_body:
                    body[i] = counter
                    found_body = True
                elif not found_food and (x,y) == self.food:
                    food[i] = counter
                    found_food = True
                counter += 1
                x += m[0]
                y += m[1]
        obs[0,8:16] += body
        obs[0,16:] += food
        return obs

class Agent:
    '''
    Class for basic genetic models, including methods for mutating model weights
    Meant to be used in conjunction with one of the Population classes
    '''
    def __init__(self, model, env):
        self.model = model
        self.rundict = {}
        self._env = env

    def run(self, episodes, display=False):
        self._env = Env.copy(self._env)
        self._env.display = display
        self._env.episodes = episodes
        if display:
            self._env.show_every = 1
        self._env.get_action = self.get_action
        self.rundict = self._env.run()
        return self.rundict

    def get_action(self, obs):
        return np.argmax(self.model.predict(x=obs)[0])

    def copy_model(self, model):
        self.model.set_weights(model.get_weights())

    def mutate_layer(self, layer, mutation_chance, rate=0.2):
        choice = np.random.rand(*layer.shape)
        choice[choice >= mutation_chance] = 0
        layer[choice > 0] += np.random.normal() * rate
        return layer

    def mutate_model(self, mutation_chance=0.1):
        mutated = []
        for layer in self.model.get_weights():
            mutated.append(self.mutate_layer(layer,mutation_chance))

        self.model.set_weights(mutated)


class Population:
    '''
    Base class for PopulationTakeTop and PopulationAvg containing shared methods
    '''

    agents = []

    def edit_env(self, **kwargs):
        for a in self.agents:
            for (k,v) in kwargs.items():
                a._env._init_kwargs[k] = v
            a._env.reset() 


class PopulationTakeTop(Population):
    '''
    Class for genetic RL. Takes the approach of mutating its agents at the start
    of every generation, then taking the TAKE_TOP best performing, and filling the rest
    of the population with copies of the TAKE_TOP agents.
    '''
    def __init__(self, agents, ep_per_gen=10, take_top=10, savedirpath=None, mutation_chance=0.1):
        self.agents = agents
        self.EPISODES_PER_GENERATION = ep_per_gen
        self.TAKE_TOP = take_top
        self.SAVE_DIR_PATH = savedirpath
        self.MUTATION_CHANCE = mutation_chance
        self.generation_n = 1

    def evolve(self):
        # 1. Each agents runs through a few episodes
        for i in tqdm(range(len(self.agents)), ascii=True, unit='agents'):
            self.agents[i].run(self.EPISODES_PER_GENERATION)
        # 2. Agents are sorted by fitness, in this case the sum of the rewards they got
        self.agents.sort(key=lambda a: a.rundict['sumrewards'],reverse=True)
        # 3. Get average performance of the agents in this generation
        avg = np.mean([a.rundict['sumrewards'] for a in self.agents])
        # 4. Calculate the chances of the top TAKE_TOP agents
        #    This is a weighted average based on how above the total average they are
        #    If some are below the average, their chance is set to 0
        chances = [a.rundict['sumrewards']-avg for a in self.agents[:self.TAKE_TOP]]
        chances = [ch if ch > 0 else 0 for ch in chances]
        sumch = sum(chances)
        print(f"Generation {self.generation_n}, best results:")
        print(f"{self.agents[0].rundict}")
        print(f"Generation {self.generation_n}, averages of top {self.TAKE_TOP} agents:")
        print(f"{[a.rundict['aggr']['avg'] for a in self.agents[:self.TAKE_TOP]]}")
        # 5. If provided, save the top TAKE_TOP agents
        if self.SAVE_DIR_PATH:
            for i, a in enumerate(self.agents[:self.TAKE_TOP]):
                a.model.save_weights(self.SAVE_DIR_PATH + f"weights-top-{i}.h5")
        # 6. Replace the rest of the agents with copies of TAKE_TOP agents and mutate them
        for i in range(self.TAKE_TOP,len(self.agents)):
            choicei = np.random.choice(range(self.TAKE_TOP),p=[ch / sumch for ch in chances])
            self.agents[i].copy_model(self.agents[choicei].model)
            self.agents[i].mutate_model(self.MUTATION_CHANCE)
        self.generation_n += 1
