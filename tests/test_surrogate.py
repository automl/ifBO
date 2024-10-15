import ifbo
import unittest


class TestSurrogateModel(unittest.TestCase):

    def setUp(self):
        # Initialize the surrogate model and any required data
        self.model = ifbo.surrogate.FTPFN(version="0.0.1")
        single_eval_pos = 700 
        batch = ifbo.priors.ftpfn_prior.get_batch(
            batch_size=1,
            seq_len=1000,
            num_features=12,
            single_eval_pos=single_eval_pos 
        )
        self.context, self.query = ifbo.utils.detokenize(batch, context_size=single_eval_pos, device="cpu")

    def test_prediction_shape(self):
        # Test if the prediction output has the correct shape
        predictions = self.model.predict(self.context, self.query)
        self.assertEqual(len(predictions), len(self.query))
    
    def assertBetween(self, value, min, max):
        """Fail if value is not between min and max (inclusive)."""
        self.assertTrue((value >= min).all())
        self.assertTrue((value <= max).all())

    def test_prediction_values(self):
        predictions = self.model.predict(self.context, self.query)
        for prediction in predictions:
            for q in [0.01, 0.5, 0.99]:
                self.assertBetween(prediction.quantile(q), 0, 1)
            self.assertBetween(prediction.ucb(), 0, 1)
            self.assertBetween(prediction.ei(0.5), 0, 1)
            self.assertBetween(prediction.pi(0.5), 0, 1)
    
    def test_exception_hyperparameters(self):
        """Test if the model raises an exception for invalid input."""
        invalid_context = self.context.copy()
        invalid_context[0].hyperparameters[-1] = -1
        self.assertRaises(Exception, self.model.predict, invalid_context, self.query)

        invalid_context = self.context.copy()
        invalid_context[0].hyperparameters[-1] = 2
        self.assertRaises(Exception, self.model.predict, invalid_context, self.query)

        invalid_query = self.query.copy()
        invalid_query[0].hyperparameters[-1] = -1
        self.assertRaises(Exception, self.model.predict, self.context, invalid_query)

        invalid_query = self.query.copy()
        invalid_query[0].hyperparameters[-1] = 2
        self.assertRaises(Exception, self.model.predict, self.context, invalid_query)
    
    def test_exception_step(self):
        """Test if the model raises an exception for invalid input."""
        invalid_context = self.context.copy()
        invalid_context[0].t = -1
        self.assertRaises(Exception, self.model.predict, invalid_context, self.query)

        invalid_context = self.context.copy()
        invalid_context[0].t = 2
        self.assertRaises(Exception, self.model.predict, invalid_context, self.query)

        invalid_query = self.query.copy()
        invalid_query[0].t = -1
        self.assertRaises(Exception, self.model.predict, self.context, invalid_query)

        invalid_query = self.query.copy()
        invalid_query[0].t = 2
        self.assertRaises(Exception, self.model.predict, self.context, invalid_query)
    
    def test_exception_performance(self):
        """Test if the model raises an exception for invalid input."""
        invalid_context = self.context.copy()
        invalid_context[0].y = -1
        self.assertRaises(Exception, self.model.predict, invalid_context, self.query)

        invalid_context = self.context.copy()
        invalid_context[0].y = 2
        self.assertRaises(Exception, self.model.predict, invalid_context, self.query)

        invalid_query = self.query.copy()
        invalid_query[0].y = -1
        self.assertRaises(Exception, self.model.predict, self.context, invalid_query)

        invalid_query = self.query.copy()
        invalid_query[0].y = 2
        self.assertRaises(Exception, self.model.predict, self.context, invalid_query)

        

if __name__ == '__main__':
    unittest.main()