import json
from fedn.common.log_config import logger
from fedn.network.combiner.aggregators.fedavg import Aggregator as FedAvgAggregator

class Aggregator(FedAvgAggregator):
    """Federated Optimization strategy (FedProx) aggregator.

    Implementation based on https://arxiv.org/abs/1812.06127

    """

    def __init__(self, update_handler):
        super().__init__(update_handler)
        self.name = "fedprox"

    def combine_models(self, helper=None, delete_models=True, parameters=None):
        """Aggregate all model updates in the queue by computing an incremental
        weighted average of model parameters.

        :param helper: An instance of :class: `fedn.utils.helpers.helpers.HelperBase`, ML framework specific helper, defaults to None
        :type helper: class: `fedn.utils.helpers.helpers.HelperBase`, optional
        :param delete_models: Delete models from storage after aggregation, defaults to True
        :type delete_models: bool, optional
        :param parameters: Aggregator hyperparameters, defaults to None.
        :type parameters: dict, optional
        :return: The global model and metadata
        :rtype: tuple
        """
        
        # Check for proximal term 'mu' in parameters
        if parameters:
            try:
                # parameters is a Parameters object, we can access it like a dict or use get method if available, 
                # but based on fedopt.py it seems to support dictionary-like access or validation.
                # However, RoundHandler converts dict to Parameters object.
                # Let's check how Parameters object is used. In fedopt.py parameters['serveropt'] is used.
                # So we assume __getitem__ is implemented.
                if 'mu' not in parameters and 'proximal_mu' not in parameters:
                     logger.warning("AGGREGATOR({}): 'mu' or 'proximal_mu' not found in aggregator_kwargs. FedProx may default to FedAvg behavior on clients if not configured properly.".format(self.name))
                else:
                     mu = parameters.get('mu') or parameters.get('proximal_mu')
                     logger.info("AGGREGATOR({}): Using FedProx with mu={}".format(self.name, mu))

            except Exception as e:
                logger.warning("AGGREGATOR({}): Failed to parse parameters: {}".format(self.name, e))
        else:
             logger.warning("AGGREGATOR({}): No parameters provided. FedProx requires 'mu' (or 'proximal_mu') in aggregator_kwargs.".format(self.name))

        # FedProx aggregation on the server side is identical to FedAvg (weighted averaging).
        # The proximal term is handled on the client side.
        return super().combine_models(helper=helper, delete_models=delete_models, parameters=parameters)
