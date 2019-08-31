# generalised-reinforcement-learning-environment
## The Env class
The main feature is the Env class, which gets rid of the hassle of building custom environments, and instead asks; what is unique about your RL project?

The following is a snippet of the full code example provided in the repo, just initialising the Env object:
```python
def take_action(state,action):
    oldx,oldy = state['blobs'][0].x,state['blobs'][0].y
    state['blobs'][0].action(action)
    newx,newy = state['blobs'][0].x,state['blobs'][0].y
    state['obs'] += [oldx-newx,oldy-newy,oldx-newx,oldy-newy]

    if np.count_nonzero(state['obs'][0,:2]) == 0:
        return state,10,True
    elif np.count_nonzero(state['obs'][0,2:]) == 0:
        return state,-10,True
    else:
        return state,-1,False

def get_observation(state):
    return state['obs']
    
def get_start_state():
    p,e,f = (Block(X,Y),Block(X,Y),Block(X,Y))
    while (e.x,e.y) == (f.x,f.y):
        e = Block(X,Y)
    while (p.x,p.y) == (f.x,f.y) or (p.x,p.y) == (e.x,e.y):
        p = Block(X,Y)
    return {'blocks': (p,e,f), 'obs':  np.asarray([(*(f-p), *(e-p))],dtype=np.float)}

def display_env(state):
    env = np.zeros((X,Y,3), dtype=np.uint8)
    env[state['blocks'][1].y,state['blocks'][1].x] = (0,0,255)
    env[state['blocks'][2].y,state['blocks'][2].x] = (0,255,0)
    env[state['blocks'][0].y,state['blocks'][0].x] = (255,0,0)
    return env

env = Env(
    take_action=take_action,
    get_action=None, # provided in the Agent class
    get_start_state=get_start_state,
    display_env=display_env,
    get_observation=get_observation,
    steps_per_ep=30,
    stat_every=10,
    printing=False
)
```
Now, calling env.run() will run through the usual loop of episodes and steps, using the methods you provide it when relevant.

The basic philosophy of this class is providing hooks to only those moments that vary project to project, like taking an action, getting the starting state, etc. For all of these methods, the appropriate parameters are passed to make sure they have enough information to draw on.

To get details on the Env constructor, you can call
```python
Env.help()
```

*The flexibility of the Env class means that it can be used in pretty much any RL scenario, be it DQN, neuroevolution, or something completely new!*

## Block, Snake
Block and Snake are two starter classes for simple RL scenarios on a 2D grid, so you can use them out-of-the-box without having to reinvent the wheel. Snake also has a get_observation method implemented, so it's paired nicely with the Env class

## Agent, Population, PopulationTakeTop
These classes are specific to neuroevolution, and come closely connected together and with the Env class.

As seen in the code above, Agent has a get_action method (which just calls Keras' predict() on the observation), and Population expects an array of Agent objects.

The full code example included in this repo makes use of these classes, so check it out to see them in action.
