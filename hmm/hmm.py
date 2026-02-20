import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        if len(input_observation_states) == 0:
            raise ValueError('empty sequence!')
        # Step 1. Initialize variables
        # make table right dimensions
        num_obs = len(input_observation_states)
        num_hidden = len(self.hidden_states)
        grid = np.zeros((num_obs,num_hidden))

        # Initialize starting row
        # first row is probability of starting given that hidden state
        obs_index = self.observation_states_dict[input_observation_states[0]]
        grid[0,:] = self.prior_p[:] * self.emission_p[:, obs_index]
       
        # Step 2. Calculate probabilities
        # each row is another obs 
        for obs in range(1,num_obs):
            index = self.observation_states_dict[input_observation_states[obs]]
            for state in range(num_hidden):
                # every subsequent row is sum of previous states * transition * emission
                grid[obs,state] = np.sum(grid[obs-1,:] * self.transition_p[:,state]) * self.emission_p[state,index]

        # Step 3. Return final probability 
        forward_probability = np.sum(grid[-1,:])
        return forward_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        if len(decode_observation_states) == 0:
            raise ValueError('empty sequence!')
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        num_obs = len(decode_observation_states)
        num_hidden = len(self.hidden_states)
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))         
        
        # store which state held max
        backpointer = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype = int)
        
        obs_index = self.observation_states_dict[decode_observation_states[0]]
        viterbi_table[0, :] = self.prior_p[:] * self.emission_p[:, obs_index]

       # Step 2. Calculate Probabilities
       # each row is another obs 
        for obs in range(1,num_obs):
            index = self.observation_states_dict[decode_observation_states[obs]]
            for state in range(num_hidden):
                # every subsequent row is sum of previous states * transition * emission
                viterbi_table[obs,state] = np.max(viterbi_table[obs-1,:] * self.transition_p[:,state]) * self.emission_p[state,index]
                backpointer[obs,state] = np.argmax(viterbi_table[obs-1, :] * self.transition_p[:, state])
            
        # Step 3. Traceback 
        # find best final state
        best_path[-1] = np.argmax(viterbi_table[-1,:])
        # for each step back,look up in backpointer which previous state led to current state and add to path
        for obs in range(num_obs-1,0,-1):
            best_path[obs-1] = backpointer[obs, int(best_path[obs])]
            
        # Step 4. Return best hidden state sequence 
        # convert from indices to states
        best_hidden_state_sequence = []
        for i in best_path:
            best_hidden_state_sequence.append(self.hidden_states_dict[int(i)])
        return best_hidden_state_sequence
        