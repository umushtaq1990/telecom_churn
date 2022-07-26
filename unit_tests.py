import unittest

import pandas as pd
import get_pred


class Test_get_pred(unittest.TestCase):
    df_res = get_pred.Get_Data('6035-RIIOM', 'data/df_validation.pkl')
    def test_Get_Data_Feat_Seq(self):
        feats = ['Duration', 'night calls', 'total day calls', 'international minutes','night minutes', 'Call day minutes', 'eve calls', 'vmail',
       'international calls', 'eve minutes', 'gender', 'SeniorCitizen','Product: International', 'Product: Voice mail', 'Phone Code',
       'PaperlessBilling', 'service calls','churn']
        self.assertEqual(self.df_res.columns.to_list(), feats)

    def test_Get_Data_Feat_Shape(self):
        self.assertEqual(self.df_res.shape, (1,18))
        
if __name__ == '__main__':
    unittest.main()