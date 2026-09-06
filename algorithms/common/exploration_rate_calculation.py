import numpy as np

class StepDecay:
    def __init__(self, epsilon_start, epsilon_end, decay_step_size):
        '''
        Args:
            decay_step_size: The steps the module needs to decrease the epsilon from epsilon_start to epsilon_end
        '''
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_step_size = decay_step_size

        self.decreasing_rate = (epsilon_start-epsilon_end)/decay_step_size
        self.current_epsilon = epsilon_start

    def step(self,current_time_step,*args, **kwargs):
        self.current_epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.decreasing_rate * current_time_step
        )
        return self.current_epsilon
    
class ClassicalExploration:
    def __init__(self):
        '''
        90% of exploitation

        10% exploration
        '''

    def step(self, current_time_step,*args, **kwargs):
        return 0.1