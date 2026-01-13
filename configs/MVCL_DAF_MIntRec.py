class Param():
    
    def __init__(self, args):
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            warmup_proportion (float): The warmup ratio for learning rate.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {
            # common parameters
            'eval_monitor': ['f1'],
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 10,
            'num_train_epochs': 100, 
            # method parameters
            'warmup_proportion': 0.1,
            'grad_clip': [-1.0],
            'lr': [2e-5], 
            'learning_rate_method': 'decay',
            'weight_decay': [0.20], 
            'aligned_method': ['ctc'], 
            'shared_dim': [256], # a hyperparameter of MAG
            'eps': 1e-9, # a hyperparameter of MAG
            # parameters of loss
            'loss': 'InfoNCE',
            'temperature': [0.5], 
            # parameters of multimodal fusion
            'max_depth': [5], 
            'beta_shift': [0.006], # a hyperparameter of MAG
            'dropout_prob': [0.5],
            'output_droupout_prob':[0.0], 
            'extra_encoder': False
        }
        return hyper_parameters 
    

