from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def embed_magic_actions(means: NDArray[np.float64], stds: NDArray[np.float64], num_magic_actions: int, std_magic_action: float) -> Tuple[NDArray, NDArray]:
    """Embeds the magic action structure into average values.
       
    Description:
       1. The maximum value is non-negative (therefore, if negative, we take the absolute value.)
       
       2. If there 1 one magic arm, the index of the best arm is embedded in the first arm as
              mu_1=phi(argmax_i mu_i) = i*/K
          where i*=argmax_i mu_i and K is the number of arms.

          To prevent arm 1 from being the best, we also shift all mean values by 1, so that mu_i <- mu_i + 1 (this is done before embedding the magic action)
       
       3.  If there are multiple magic action, the first arm embeds the index of the next magic arm (chosen uniformly at random)


    Args:
        means (NDArray[np.float64]): Vector of average values per arm
        stds (NDArray[np.float64]): Vector of standard deviations per arm
        num_magic_actions (int): Number of magic actions to embed
        std_magic_action (float): standard deviation of the magic actions

    Returns:
        Tuple[NDArray, NDArray]: _description_
    """
    assert num_magic_actions < len(means) and num_magic_actions >= 0, 'The number of magic actions needs to be in [0,K-1]'
    if num_magic_actions < 1:
        return means, stds
    
    L = len(means)
    
    # Find maximum
    argmax = means[1:].argmax() + 1
    max = means[argmax]



    means[argmax] = np.abs(max)     # Ensure it's not negative
    means = means + 1               # guarantees that the reward of any magic action is not the new max. value

    current_magic_action_index = 0
    forbidden_indexes = set([argmax, 0])    # Indexes not to be chosen as the new magic actions

    for i in range(num_magic_actions):
        if i == num_magic_actions - 1:
            next_magic_action_index = argmax
        else:
            next_magic_action_index = np.random.choice(
                list(set(range(L)).difference(forbidden_indexes)))
        
        means[current_magic_action_index] = next_magic_action_index / L
        stds[current_magic_action_index] = std_magic_action

        forbidden_indexes.add(next_magic_action_index)
        current_magic_action_index  = next_magic_action_index
    
    return means, stds


def beta_sampling(num_actions: int, min_val: float, max_val: float, alpha: float, beta: float) -> NDArray[np.float64]:
    """Sample arm means from a Beta(alpha, beta) distribution scaled to [min_val, max_val].

    Args:
        num_actions: number of arms
        min_val: lower bound of the scaled range
        max_val: upper bound of the scaled range
        alpha: Beta distribution shape parameter alpha
        beta: Beta distribution shape parameter beta

    Returns:
        NDArray of shape (num_actions,) with arm means in [min_val, max_val]
    """
    x = np.random.beta(alpha, beta, size=num_actions)
    return x * (max_val - min_val) + min_val


def dirichlet_cap_sampling(num_actions: int, num_players: int, concentration: float) -> NDArray[np.int32]:
    """Sample arm capacities from a shared Dirichlet-Categorical model.

    Draws one probability vector theta ~ Dirichlet(concentration * ones(P)) per episode,
    then samples each arm's capacity independently from Categorical(theta), mapped to {1,...,P}.
    Large concentration -> theta ~= uniform -> approaches training prior (i.i.d. Uniform{1,...,P}).

    Args:
        num_actions: number of arms K
        num_players: number of players P (capacity range is {1,...,P})
        concentration: symmetric Dirichlet concentration parameter

    Returns:
        NDArray of shape (num_actions,) with integer capacities in {1,...,P}
    """
    theta = np.random.dirichlet(concentration * np.ones(num_players))
    caps  = np.random.choice(num_players, size=num_actions, p=theta) + 1
    return caps.astype(np.int32)


def minimum_gap_sampling(num_actions: int, min_val: float, max_val: float, min_gap: float):
    assert (num_actions - 1) * min_gap <= 1, 'Minimum gap is not smaller than 1/(N-1)'
    L = (1 - (num_actions-1) * min_gap)
    u = np.random.rand(num_actions) * L


    s = np.sort(u)
    x = s+np.arange(num_actions) * min_gap
    return np.random.permutation(x) * (max_val - min_val) + min_val

 