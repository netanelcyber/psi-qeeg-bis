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

    def test_training_weights_balance_people_and_classes_not_epochs(self):
        import numpy as np
        from psi_qeeg.ml import subject_training_weights
        subjects = np.array(['a'] * 2 + ['b'] * 20 + ['c'] * 4)
        labels = np.array(['healthy_control'] * 22 + ['psychosis_spectrum'] * 4)
        scaler, svm = subject_training_weights(subjects, labels)
        for person in set(subjects):
            self.assertAlmostEqual(scaler[subjects == person].sum(), 1)
        self.assertAlmostEqual(svm[labels == 'healthy_control'].sum(), 1.5)
        self.assertAlmostEqual(svm[labels == 'psychosis_spectrum'].sum(), 1.5)

    def test_replicating_a_participant_does_not_change_scaler_or_decision(self):
        import joblib
        import numpy as np
        from sklearn.pipeline import Pipeline
        from unittest.mock import patch
        original = self.frame.copy()
        fitted_weights = []
        original_fit = Pipeline.fit
        def checked_fit(model, x, y, **kwargs):
            fitted_weights.append((kwargs['standardscaler__sample_weight'].sum(), kwargs['svc__sample_weight'].sum()))
            return original_fit(model, x, y, **kwargs)
        with patch.object(Pipeline, 'fit', checked_fit):
            result = self.run_train()
        self.assertEqual(len(fitted_weights), 10)  # Nine LOSO folds and one final refit.
        np.testing.assert_allclose(fitted_weights[:9], np.full((9, 2), 8))
        np.testing.assert_allclose(fitted_weights[-1], [9, 9])
        first = joblib.load(self.root / 'out/group_model.joblib')
        self.frame = pd.concat([original, *([original[original.subject_id == 'healthy_control-0']] * 8)], ignore_index=True)
        repeated = self.run_train()
        second = joblib.load(self.root / 'out/group_model.joblib')
        np.testing.assert_allclose(first.named_steps['standardscaler'].mean_, second.named_steps['standardscaler'].mean_, atol=1e-12)
        np.testing.assert_allclose(first.named_steps['standardscaler'].var_, second.named_steps['standardscaler'].var_, atol=1e-12)
        grid = original[['f1', 'f2']].to_numpy()
        np.testing.assert_allclose(first.decision_function(grid), second.decision_function(grid), atol=2e-3)
        self.assertEqual(result['subject_confusion_matrix'], repeated['subject_confusion_matrix'])
        for fold in repeated['folds']:
            self.assertNotIn(fold['held_out_subject'], fold['training_subjects'])
        weights = pd.read_csv(self.root / 'out/training_subject_weights.csv')
        np.testing.assert_allclose(weights.scaler_mass, 1)
