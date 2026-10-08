"""Synthetic schema/evaluation tests; accuracy here is not clinical evidence."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from psi_qeeg.ml import train_research_model


@unittest.skipUnless(importlib.util.find_spec("sklearn"),"Optional scikit-learn extra not installed")
class ResearchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        rows=[]
        for s in range(3):
            for i in range(8):
                rows.append({"subject_id":f"test-{s}","session_id":"synthetic-session", "epoch_id":i,
                             "research_label":"baseline" if i%2==0 else "sensory_loading",
                             "label_source":"participant","recording_context":"synthetic_unit_test",
                             "alpha_power":i%2+s*.01})
        self.frame=pd.DataFrame(rows)

    def run_training(self,features=None):
        p=self.root/"labels.csv"
        self.frame.to_csv(p,index=False)
        return train_research_model(p,self.root/"result",features or ["alpha_power"])

    def test_subject_held_out_prediction_for_every_row(self):
        result=self.run_training()
        self.assertEqual(result["subjects"],3)
        self.assertEqual(len(result["folds"]),3)
        output=pd.read_csv(self.root/"result/held_out_predictions.csv")
        self.assertEqual(len(output),len(self.frame))
        self.assertTrue(output.held_out_prediction.notna().all())

    def test_anesthesia_not_relabelled_as_psychiatric_state(self):
        self.frame["recording_context"]="perioperative_anesthesia"
        with self.assertRaises(ValueError):self.run_training()

    def test_identifiers_not_feature_inputs(self):
        with self.assertRaises(ValueError):self.run_training(["subject_id"])

    def test_missing_features_not_imputed(self):
        self.frame.loc[0,"alpha_power"]=float("nan")
        with self.assertRaises(ValueError):self.run_training()

    def test_duplicate_epoch_rejected(self):
        self.frame=pd.concat([self.frame,self.frame.iloc[:1]],ignore_index=True)
        with self.assertRaises(ValueError):self.run_training()

    def test_unknown_label_source_rejected(self):
        self.frame["label_source"]="EEG_guess"
        with self.assertRaises(ValueError):self.run_training()



@unittest.skipUnless(importlib.util.find_spec("sklearn"),"Optional scikit-learn extra not installed")
class GroupTrainingTests(unittest.TestCase):
    """Synthetic data: accuracy here is not evidence about any population."""
    def setUp(self):
        from psi_qeeg.ml import train_group_model
        self.train=train_group_model
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        import numpy as np
        rng=np.random.default_rng(0)
        rows=[]
        for g,name in enumerate(("autism_spectrum","psychosis_spectrum","healthy_control")):
            for s in range(3):
                for i in range(10):
                    rows.append({"subject_id":f"{name}-{s}","group":name,"site":f"site{s%2}","accepted":True,
                                 "f1":g+rng.normal(0,.2),"f2":rng.normal()})
        self.frame=pd.DataFrame(rows)

    def run_train(self,features=("f1","f2")):
        p=self.root/"c.csv"
        self.frame.to_csv(p,index=False)
        return self.train(p,self.root/"out",list(features))

    def test_separable_groups_and_subject_votes(self):
        r=self.run_train()
        self.assertEqual(r["subject_balanced_accuracy"],1.0)
        self.assertTrue((self.root/"out/held_out_subject_predictions.csv").exists())

    def test_single_subject_group_rejected(self):
        self.frame=self.frame[~self.frame.subject_id.isin(["healthy_control-1","healthy_control-2"])]
        with self.assertRaises(ValueError):self.run_train()

    def test_site_group_confounding_rejected(self):
        self.frame["site"]=self.frame.group
        with self.assertRaises(ValueError):self.run_train()

    def test_group_not_a_feature(self):
        with self.assertRaises(ValueError):self.run_train(["group"])

    def test_surgical_context_rejected(self):
        self.frame["recording_context"]="perioperative_anesthesia"
        with self.assertRaises(ValueError):self.run_train()
