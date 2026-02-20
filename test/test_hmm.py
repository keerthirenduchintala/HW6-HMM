import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')


  
    hmm = HiddenMarkovModel(
        observation_states=mini_hmm['observation_states'],
        hidden_states = mini_hmm['hidden_states'],
        prior_p = mini_hmm['prior_p'],
        transition_p = mini_hmm['transition_p'],
        emission_p = mini_hmm['emission_p']
    )
    # Forward algorithm
    fwd_prob = hmm.forward(mini_input['observation_state_sequence'])
    print(fwd_prob)

    # viterbi algorithm
    viterbi = hmm.viterbi(mini_input['observation_state_sequence'])
    print(viterbi)
    print(mini_input['best_hidden_state_sequence'])
    # fwd_prob tests

    assert fwd_prob <= 1 and fwd_prob > 0


    # viterbi test

    assert len(viterbi) == len(mini_input['best_hidden_state_sequence'])
    
    assert viterbi == list(mini_input['best_hidden_state_sequence'])

    # test valueError for empty sequence - edge
    with pytest.raises(ValueError):
        hmm.forward(np.array([]))

 
    # ensure viterbi only returns existing hidden states - edge

    for state in viterbi:
        assert state in mini_hmm['hidden_states']
   



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')


  
    hmm = HiddenMarkovModel(
        observation_states=full_hmm['observation_states'],
        hidden_states = full_hmm['hidden_states'],
        prior_p = full_hmm['prior_p'],
        transition_p = full_hmm['transition_p'],
        emission_p = full_hmm['emission_p']
    )
    # Forward algorithm
    fwd_prob = hmm.forward(full_input['observation_state_sequence'])
    print(fwd_prob)

    # viterbi algorithm
    viterbi = hmm.viterbi(full_input['observation_state_sequence'])
    print(viterbi)
    print(full_input['best_hidden_state_sequence'])
    # fwd_prob tests

    assert fwd_prob <= 1 and fwd_prob > 0

    # viterbi test

    assert len(viterbi) == len(full_input['best_hidden_state_sequence'])
    
    assert viterbi == list(full_input['best_hidden_state_sequence'])    
   
    












