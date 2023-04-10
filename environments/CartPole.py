
from collections import defaultdict, namedtuple
from copy import deepcopy
import gymnasium as gym
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

import numpy as np
import time

from MCTS.MCTS import MCTS, Node # MCTS is the package name. MCTS is the module name, MCTS is the class name....damn I fucked up 🤣

num_hist_samples = 0

# Define the bins for the 4-d phase space
# make the bins coarser around zero and finer at the bounds 
CART_X_BOUNDS = (-2.4, 2.4)
CART_V_BOUNDS = (-5, 5)
POLE_THETA_BOUNDS = (-.2095, .2095)
POLE_V_BOUNDS = (-5, 5)
density_factor = 0.005
CART_X_BINS = list(np.geomspace(density_factor,CART_X_BOUNDS[1], 25)-(CART_X_BOUNDS[1]+density_factor)) + list((CART_X_BOUNDS[1]+density_factor)-np.flip(np.geomspace(density_factor,CART_X_BOUNDS[1], 25)))
CART_V_BINS = list(np.geomspace(density_factor,CART_V_BOUNDS[1], 25)-(CART_V_BOUNDS[1]+density_factor)) +list((CART_V_BOUNDS[1]+density_factor)-np.flip(np.geomspace(density_factor,CART_V_BOUNDS[1], 25)))
POLE_THETA_BINS = list(np.geomspace(density_factor,POLE_THETA_BOUNDS[1], 25)-(POLE_THETA_BOUNDS[1]+density_factor)) + list((POLE_THETA_BOUNDS[1]+density_factor)-np.flip(np.geomspace(density_factor,POLE_THETA_BOUNDS[1], 25)))
POLE_V_BINS =list(np.geomspace(density_factor,POLE_V_BOUNDS[1], 25)-(POLE_V_BOUNDS[1]+density_factor)) +list((POLE_V_BOUNDS[1]+density_factor)-np.flip(np.geomspace(density_factor,POLE_V_BOUNDS[1], 25)))

CART_X_MESH, CART_V_MESH = np.meshgrid(CART_X_BINS, CART_V_BINS, indexing='ij')
POLE_THETA_MESH, POLE_V_MESH = np.meshgrid(POLE_THETA_BINS, POLE_V_BINS, indexing='ij')

TOTAL_NUM_STATES = len(CART_X_BINS) * len(CART_V_BINS) * len(POLE_THETA_BINS) * len(POLE_V_BINS)

CART_X_HISTO_DICTS = defaultdict(list)
CART_V_HISTO_DICTS = defaultdict(list)
POLE_THETA_HISTO_DICTS = defaultdict(list)
POLE_V_HISTO_DICTS = defaultdict(list)
CART_XV_HISTO_DICTS = defaultdict(list)
POLE_THETAV_HISTO_DICTS = defaultdict(list)

CART_X_TRIALS = []
CART_V_TRIALS = []
POLE_THETA_TRIALS = []
POLE_V_TRIALS = []
phase_space_fig, phase_space_axes = plt.subplots(1, 2, figsize=(6, 3))
# make sure the axes fill the figure 
phase_space_fig.tight_layout()

phase_space_axes[0].set_title("Cart position vs. Cart velocity")
phase_space_axes[0].set_xlabel("Cart position")
phase_space_axes[0].set_ylabel("Cart velocity")
# plot grids to show the bins
for i in range(len(CART_X_BINS) - 1):
    phase_space_axes[0].axvline(CART_X_BINS[i], color="k", linestyle="--", linewidth=0.5)
for i in range(len(CART_V_BINS) - 1):
    phase_space_axes[0].axhline(CART_V_BINS[i], color="k", linestyle="--", linewidth=0.5)
phase_space_axes[1].set_title("Pole angle vs. Pole velocity")
phase_space_axes[1].set_xlabel("Pole angle")
phase_space_axes[1].set_ylabel("Pole velocity")
# plot grids to show the bins
for i in range(len(POLE_THETA_BINS) - 1):
    phase_space_axes[1].axvline(POLE_THETA_BINS[i], color="k", linestyle="--", linewidth=0.5)
for i in range(len(POLE_V_BINS) - 1):
    phase_space_axes[1].axhline(POLE_V_BINS[i], color="k", linestyle="--", linewidth=0.5)
phase_space_axes[0].set_xlim(*CART_X_BOUNDS)
phase_space_axes[0].set_ylim(*CART_V_BOUNDS)
phase_space_axes[1].set_xlim(*POLE_THETA_BOUNDS)
phase_space_axes[1].set_ylim(*POLE_V_BOUNDS)

im_hist_1=None
im_hist_2=None

STEPS_PER_FRAME=100

def animate_phase_space(frame_no):
    '''updates the plot for frame no i'''

    global im_hist_1
    global im_hist_2
    if frame_no%10==0:
        initial_time = time.time()
        print(f"Processing frame: {frame_no}")
    if im_hist_1 is not None and im_hist_2 is not None:
        im_hist_1.remove()
        im_hist_2.remove()
    

    # plot a pcolormesh 
    Z = np.array([[len(np.argwhere(np.array(CART_XV_HISTO_DICTS[bin_x, bin_v])<frame_no*STEPS_PER_FRAME)) for bin_x in range(len(CART_X_BINS))] for bin_v in range(len(CART_V_BINS))])
    im_hist_1 = phase_space_axes[0].pcolormesh(CART_X_MESH, CART_V_MESH, Z, cmap="Blues")
    # plot a pcolormesh
    Z= np.array([[len(np.argwhere(np.array(POLE_THETAV_HISTO_DICTS[bin_theta, bin_v])<frame_no*STEPS_PER_FRAME)) for bin_theta in range(len(POLE_THETA_BINS))] for bin_v in range(len(POLE_V_BINS))])
    im_hist_2 = phase_space_axes[1].pcolormesh(POLE_THETA_MESH, POLE_V_MESH, Z, cmap="Blues")



   # im_hist_1 = phase_space_axes[0].imshow(np.array([len(np.argwhere(np.array(CART_X_HISTO_DICTS[bin])<frame_no)) for bin in range(len(CART_X_BINS))]).reshape(len(CART_X_BINS), 1) @ np.array([len(np.argwhere(np.array(CART_V_HISTO_DICTS[bin])<frame_no)) for bin in range(len(CART_V_BINS))]).reshape(1, len(CART_V_BINS)), cmap="Blues", extent=[-0.15, 0.15, -0.15, 0.15])
    
    # plot the 2-d histogram
    #im_hist_2 = phase_space_axes[1].imshow(np.array([len(np.argwhere(np.array(POLE_THETA_HISTO_DICTS[bin])<frame_no)) for bin in range(len(POLE_THETA_BINS))]).reshape(len(POLE_THETA_BINS), 1) @ np.array([len(np.argwhere(np.array(POLE_V_HISTO_DICTS[bin])<frame_no)) for bin in range(len(POLE_V_BINS))]).reshape(1, len(POLE_V_BINS)), cmap="Blues", extent=[-0.075, 0.075, -0.1, 0.1])

    if frame_no%10==0:
        print(f"Time taken: {time.time()-initial_time}")
 
    

def discretize(observation):
    cart_x, cart_v, pole_theta, pole_v = observation
    cart_x = np.digitize(cart_x, CART_X_BINS)
    cart_v = np.digitize(cart_v, CART_V_BINS)
    pole_theta = np.digitize(pole_theta, POLE_THETA_BINS)
    pole_v = np.digitize(pole_v, POLE_V_BINS)
    return cart_x, cart_v, pole_theta, pole_v

def discretize_adaptive(observation):
    cart_x, cart_v, pole_theta, pole_v = observation
    cart_x = np.digitize(cart_x, CART_X_BINS)
    cart_v = np.digitize(cart_v, CART_V_BINS)
    pole_theta = np.digitize(pole_theta, POLE_THETA_BINS)
    pole_v = np.digitize(pole_v, POLE_V_BINS)
    return cart_x, cart_v, pole_theta, pole_v

_CartPole = namedtuple("_CartPole", ["cart_x", "cart_v", "pole_theta", "pole_v", "done", "env"])

class CartPole(_CartPole, Node):
    '''Node representation of a cart pole environment.'''

    def IS_TWO_PLAYER():
        return False
    
    def get_children(cart_pole):
        if cart_pole.done:
            return set()
        else:
            return {cart_pole.make_move(0), cart_pole.make_move(1)}
        
    def make_move(cart_pole, index):
        global num_hist_samples
        if cart_pole.env.render_mode in ["human", "rgb_array"]:
            env_spawn = cart_pole.env
        else:
            env_spawn = deepcopy(cart_pole.env)
        obs, reward, done, truncated, info = env_spawn.step(index)
        cart_x, cart_v, pole_theta, pole_v = discretize(obs)
        num_hist_samples += 1
        CART_X_HISTO_DICTS[cart_x].append(num_hist_samples)
        CART_V_HISTO_DICTS[cart_v].append(num_hist_samples)
        POLE_THETA_HISTO_DICTS[pole_theta].append(num_hist_samples)
        POLE_V_HISTO_DICTS[pole_v].append(num_hist_samples)
        CART_XV_HISTO_DICTS[cart_x, cart_v].append(num_hist_samples)
        POLE_THETAV_HISTO_DICTS[pole_theta, pole_v].append(num_hist_samples)
        return CartPole(cart_x, cart_v, pole_theta, pole_v, done, env_spawn)
    
    def get_random_child(cart_pole):
        return cart_pole.make_move(cart_pole.env.action_space.sample())
    
    def is_terminal(cart_pole):
        return cart_pole.done
    
    def render(cart_pole):
        cart_pole.env.render()

def play_cartpole(record=True):
    '''Play cartpole using MCTS.
    Returns:
        trial_rewards: list of rewards for each trial
        tree: MCTS tree
    '''
    trial_rewards=[]
    tree = MCTS(exploration_weight=1)
    human_env = gym.make("CartPole-v1", render_mode="human")
    agent_env = gym.make("CartPole-v1")
    recording_env = gym.make("CartPole-v1", render_mode="rgb_array")
    recording_env.reset()
    video = gym.wrappers.monitoring.video_recorder.VideoRecorder(recording_env, f"cartpole.mp4", enabled=True)
    global num_hist_samples
    # training
    for trials in range(10000):
        for _ in range(10):
            obs, _= agent_env.reset()
            obs = discretize(obs)
            # fill histograms to see what states are mostly occupied 
            
            cartpole = CartPole(*obs, False, agent_env)
            tree.do_rollout(cartpole)

        
       
        if trials%50==0:
            ave_reward = []
            for _ in range(10):
                print(f"trial {trials}")
                obs, _ = human_env.reset(seed=42)
                obs = discretize(obs)
                cartpole = CartPole(*obs, False, human_env)
                time_step=0
                while not cartpole.done:
                    old_cartpole=cartpole
                    cartpole = tree.choose(cartpole)
                    cartpole.render()
                    time_step+=1
                print(f"reward: {time_step}")
                print(f"reward, number of visits: {tree.total_rewards[old_cartpole]}, {tree.number_of_visits[old_cartpole]}")
                ave_reward.append(time_step)
            trial_rewards.append(np.mean(ave_reward))

        if trials%50==0:
            for _ in range(10):
                obs, _ = recording_env.reset()
                obs = discretize(obs)
                cartpole = CartPole(*obs, False, recording_env)
                time_step=0
                while not cartpole.done:
                    cartpole = tree.choose(cartpole)
                    time_step+=1
                    video.capture_frame()
    video.close()


    
    return trial_rewards, tree

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Playing CartPole using MCTS")
    print("Discretized state space size:", TOTAL_NUM_STATES)
    print("Bining cart x:", CART_X_BINS)
    print("Bining cart v:", CART_V_BINS)
    print("Bining pole theta:", POLE_THETA_BINS)
    print("Bining pole v:", POLE_V_BINS)

    rewards, tree = play_cartpole()
    plt.plot(np.array(range(len(rewards)))*50, rewards)

    # plot the histograms
    plt.figure()
    plt.plot(CART_X_BINS, [len(CART_X_HISTO_DICTS[bin]) for bin in range(len(CART_X_BINS))])
    plt.title("Cart X")
    plt.figure()
    plt.plot(CART_V_BINS, [len(CART_V_HISTO_DICTS[bin]) for bin in range(len(CART_V_BINS))])
    plt.title("Cart V")
    plt.figure()
    plt.plot(POLE_THETA_BINS, [len(POLE_THETA_HISTO_DICTS[bin]) for bin in POLE_THETA_BINS])
    plt.title("Pole Theta")
    plt.figure()
    plt.plot(POLE_V_BINS, [len(POLE_V_HISTO_DICTS[bin]) for bin in range(len(POLE_V_BINS))])
    plt.title("Pole V")
    print("Number of samples:", num_hist_samples)
    print(f"Number of frames: {num_hist_samples//STEPS_PER_FRAME}")
    ani = FuncAnimation(phase_space_fig, animate_phase_space, frames=num_hist_samples//STEPS_PER_FRAME, interval=5, repeat=False)
    ani.save("phase_space.gif", writer="ffmpeg")
    plt.show()

