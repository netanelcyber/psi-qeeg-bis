"""Exploratory nested SVM training on source-deposited mouse qEEG vectors.

One row is one animal. Genotype is not a behavioral or psychiatric event.
The final refit is kept distinct from animal/experiment-held-out predictions.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_FEATURES = [f'source_{site}__frequency_label_{frequency:02d}__log10_Pow'
                   for site in ('Front', 'Temp') for frequency in range(1, 100)]
MODEL_KIND = 'source_qeeg_genotype_classifier'
TARGET = 'source_WT_KO_genotype_not_behavioral_onset'
SCOPES = {'synthetic_fixture', 'real_published_processed_animal_qeeg_features'}


def verify_source_snapshot(cache, snapshot_path, expected_hashes):
    """Explicit offline reuse: exact pinned cached bytes, never a silent fallback."""
    snapshot = json.loads(Path(snapshot_path).read_text())
    files = snapshot.get('files', [])
    if len(files) != len(expected_hashes) or {item.get('filename') for item in files} != set(expected_hashes):
        raise ValueError('Retained provenance must cover exactly the pinned source files')
    for item in files:
        filename = item['filename']; path = Path(cache) / filename
        if (item.get('sha256') != expected_hashes[filename] or not path.is_file() or
                path.stat().st_size != item.get('bytes') or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[filename]):
            raise ValueError('Retained source bytes/hash/size mismatch; refresh and verify before fitting')
    return files


def _validate(frame, features, data_scope):
    if data_scope not in SCOPES:
        raise ValueError('Specify real deposited animal features or an explicit synthetic fixture')
    required = {'animal_id', 'genotype', 'experiment', *features}
    if not features or len(features) != len(set(features)) or not required <= set(frame):
        raise ValueError('Need animal IDs, source experiments, genotypes and unique deposited-power features')
    if not set(features) <= set(SOURCE_FEATURES):
        raise ValueError('Genotype, experiment, behavior and non-power columns cannot be predictors')
    if data_scope != 'synthetic_fixture' and features != SOURCE_FEATURES:
        raise ValueError('Real Fmr1 training uses the complete two-site/99-source-label feature contract')
    if frame[list(required)].isna().any().any() or frame.animal_id.astype(str).str.strip().eq('').any():
        raise ValueError('No missing animal metadata or features are allowed')
    if frame.animal_id.duplicated().any():
        raise ValueError('Exactly one source-condition vector per independent animal is required')
    if set(frame.genotype) != {'WT', 'KO'} or not np.isfinite(frame[features].to_numpy(float)).all():
        raise ValueError('Need finite log10-power features and both source WT/KO classes')


def _select_model(train, features, c_values, pca_components, inner_folds, seed):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Stratify within source experiment and genotype. There is one row per
    # animal, so no animal can appear in both sides of an inner split.
    strata = train.experiment.astype(str) + '/' + train.genotype.astype(str)
    folds = min(inner_folds, int(strata.value_counts().min()))
    if folds < 2:
        raise ValueError('Each training experiment/genotype needs at least two animals for inner selection')
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(train, strata))
    max_components = min(len(features), min(len(left) for left, _ in splits) - 1)
    options = [n for n in pca_components if n is None or n <= max_components]
    if not options:
        raise ValueError('No PCA option fits the available inner-training animals/features')
    grid = [{'pca': ['passthrough' if n is None else PCA(n_components=n, svd_solver='full')],
             'svm__C': list(c_values)} for n in options]
    model = Pipeline([('scaler', StandardScaler()), ('pca', 'passthrough'),
                      ('svm', SVC(kernel='linear', class_weight='balanced'))])
    search = GridSearchCV(model, grid, scoring='balanced_accuracy', cv=splits,
                          refit=True, n_jobs=1, error_score='raise')
    search.fit(train[features].to_numpy(float), train.genotype.to_numpy())

    def parameters(value):
        pca = value['pca']
        return {'C': float(value['svm__C']), 'pca_components': None if isinstance(pca, str) else int(pca.n_components)}

    info = {'selected_parameters': parameters(search.best_params_),
            'selection_score_not_external_performance': float(search.best_score_),
            'inner_splits': [{'training_animals': list(train.iloc[left].animal_id),
                              'validation_animals': list(train.iloc[right].animal_id)} for left, right in splits],
            'candidate_scores': [{**parameters(params), 'mean_inner_balanced_accuracy': float(mean),
                                  'sd_inner_balanced_accuracy': float(sd)} for params, mean, sd in
                                 zip(search.cv_results_['params'], search.cv_results_['mean_test_score'],
                                     search.cv_results_['std_test_score'])],
            'model_fits': len(splits) * len(search.cv_results_['params']) + 1}
    return search.best_estimator_, info


def _metrics(truth, predicted):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    return {'animal_balanced_accuracy': float(balanced_accuracy_score(truth, predicted)),
            'confusion_matrix': {'labels': ['KO', 'WT'],
                                 'rows_true_cols_pred': confusion_matrix(truth, predicted, labels=['KO', 'WT']).tolist()}}


def linear_rule(model, features):
    """Reconstruct the fitted decision margin in native log10-Pow coordinates."""
    scaler, pca, svm = (model.named_steps[name] for name in ('scaler', 'pca', 'svm'))
    weights, intercept = svm.coef_[0].copy(), float(svm.intercept_[0])
    if not isinstance(pca, str):
        weights = weights @ pca.components_
        intercept -= float(weights @ pca.mean_)
    native = weights / scaler.scale_
    native_intercept = intercept - float(native @ scaler.mean_)
    table = pd.DataFrame({'feature': features, 'standardized_log10_power_weight': weights,
                          'native_log10_power_weight': native})
    rule = {'positive_margin_class': str(svm.classes_[1]), 'negative_margin_class': str(svm.classes_[0]),
            'native_log10_intercept': native_intercept,
            'formula': 'margin = dot(native_log10_power_weights, deposited_log10_Pow_vector) + intercept',
            'interpretation': 'Genotype SVM margin only; not a calibrated probability or behavioral-wave threshold'}
    return table, rule


def train_source_genotype(frame, features, out, *, data_scope, source_provenance=None,
                          c_values=(0.01, 0.1, 1.0, 10.0), pca_components=(None, 5, 10),
                          inner_folds=3, seed=20261013, progress=None):
    try:
        import joblib
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ValueError("Genotype fitting requires: pip install 'psi-qeeg-bis[ml]'") from exc
    _validate(frame, features, data_scope)
    if len(frame) < 8 or frame.experiment.nunique() != 2:
        raise ValueError('Use two documented experiments and sufficient independently identified animals')
    if not isinstance(inner_folds, int) or isinstance(inner_folds, bool) or inner_folds < 2 or not c_values or any(not np.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError('Specify positive finite C values and at least two inner folds')
    if not pca_components or any(n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1) for n in pca_components):
        raise ValueError('PCA options must be positive component counts or None for the full vector')
    frame = frame.sort_values('animal_id').reset_index(drop=True)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    predictions, baseline_predictions, folds, model_fits = [], [], [], 0
    for index in range(len(frame)):
        train, test = frame.drop(index).reset_index(drop=True), frame.iloc[[index]]
        model, selection = _select_model(train, features, c_values, pca_components, inner_folds, seed)
        baseline = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, class_weight='balanced'))
        baseline.fit(train[features].to_numpy(float), train.genotype)
        predictions.append(str(model.predict(test[features].to_numpy(float))[0]))
        baseline_predictions.append(str(baseline.predict(test[features].to_numpy(float))[0]))
        folds.append({'held_out_animal': str(test.animal_id.iat[0]),
                      'training_animals': list(train.animal_id), **selection})
        model_fits += selection['model_fits'] + 1
        if progress and (index == 0 or (index + 1) % 10 == 0 or index == len(frame) - 1):
            progress({'stage': 'nested_animal_holdout', 'completed': index + 1, 'total': len(frame)})
    held_out = frame[['animal_id', 'genotype', 'experiment']].copy()
    held_out['nested_prediction'], held_out['fixed_C1_prediction'] = predictions, baseline_predictions
    held_out.to_csv(out / 'nested_held_out_animals.csv', index=False)

    experiment_results = {}
    for experiment in sorted(frame.experiment.unique()):
        train = frame[frame.experiment != experiment].reset_index(drop=True)
        test = frame[frame.experiment == experiment]
        model, selection = _select_model(train, features, c_values, pca_components, inner_folds, seed)
        table = test[['animal_id', 'genotype', 'experiment']].copy()
        table['prediction'] = model.predict(test[features].to_numpy(float))
        name = 'test_' + str(experiment)
        table.to_csv(out / (name + '_predictions.csv'), index=False)
        joblib.dump({'model_kind': MODEL_KIND, 'target': TARGET, 'features': features,
                     'model': model, 'data_scope': data_scope, 'training_ids': list(train.animal_id)}, out / (name + '_model.joblib'))
        experiment_results[name] = {'training_animals': list(train.animal_id), 'test_animals': list(test.animal_id),
                                    **_metrics(test.genotype, table.prediction), **selection}
        model_fits += selection['model_fits']
    final, selection = _select_model(frame, features, c_values, pca_components, inner_folds, seed)
    model_fits += selection['model_fits']
    coefficients, rule = linear_rule(final, features)
    np.testing.assert_allclose(frame[features].to_numpy(float) @ coefficients.native_log10_power_weight.to_numpy() +
                               rule['native_log10_intercept'], final.decision_function(frame[features].to_numpy(float)),
                               atol=1e-9, rtol=1e-9)
    coefficients.to_csv(out / 'final_linear_svm_coefficients.csv', index=False)
    (out / 'final_linear_decision_rule.json').write_text(json.dumps(rule, indent=2) + '\n')
    fixed = Pipeline([('scaler', StandardScaler()), ('pca', 'passthrough'),
                      ('svm', SVC(kernel='linear', C=1, class_weight='balanced'))])
    fixed.fit(frame[features].to_numpy(float), frame.genotype)
    model_fits += 1
    fixed_coefficients, fixed_rule = linear_rule(fixed, features)
    fixed_coefficients.to_csv(out / 'fixed_C1_final_linear_svm_coefficients.csv', index=False)
    (out / 'fixed_C1_final_linear_decision_rule.json').write_text(json.dumps(fixed_rule, indent=2) + '\n')
    report = {'schema_version': 1, 'model_kind': MODEL_KIND, 'target': TARGET, 'species': 'Mus musculus',
              'model_family': 'Fmr1', 'data_scope': data_scope, 'independent_animals': len(frame),
              'animals_per_genotype': {str(k): int(v) for k, v in frame.genotype.value_counts().items()},
              'feature_count': len(features), 'features': features, 'source_provenance': source_provenance or [],
              'candidate_grid': {'C': list(c_values), 'pca_components': list(pca_components),
                                 'inner_folds': inner_folds, 'seed': seed},
              'evaluation': 'Exploratory nested leave-one-animal-out plus two previously inspected experiment holdouts',
              'nested_animal_metrics': _metrics(frame.genotype, predictions),
              'fixed_C1_animal_metrics': _metrics(frame.genotype, baseline_predictions),
              'nested_folds': folds, 'experiment_holdouts': experiment_results,
              'final_fit': {'training_animals': list(frame.animal_id), 'parameters': selection['selected_parameters'],
                            'selection': selection, 'artifact': f'final_{len(frame)}_animal_genotype_model.joblib',
                            'evaluation_status': 'all-animal refit is not an independent test'},
              'fixed_C1_final_fit': {'training_animals': list(frame.animal_id),
                                    'parameters': {'C': 1.0, 'pca_components': None},
                                    'artifact': f'fixed_C1_final_{len(frame)}_animal_genotype_model.joblib',
                                    'evaluation_status': 'reference all-animal refit is not an independent test'},
              'model_fits': model_fits, 'behavioral_model_trained_on_real_data': False,
              'human_transfer_validation': 'not_established', 'probability_calibration': 'not_established',
              'lstm_transformer_implemented': False,
              'limitations': ['Both source experiments and earlier C=1 results were already inspected; this is exploratory reuse, not a new locked external test',
                              'One animal per row, but shared laboratory/experiment effects may dominate; no clinical or prospective validation',
                              'No significance test or uncertainty interval is reported',
                              'Deposited power/frequency labels remain in native source representation; no raw EEG PSD or event labels are synthesized',
                              'Genotype SVM coefficients are not abnormal-wave thresholds or behavioral onset evidence']}
    joblib.dump({'model_kind': MODEL_KIND, 'target': TARGET, 'features': features, 'model': final,
                 'data_scope': data_scope, 'training_ids': list(frame.animal_id), 'report': report},
                out / report['final_fit']['artifact'])
    joblib.dump({'model_kind': MODEL_KIND, 'target': TARGET, 'features': features, 'model': fixed,
                 'data_scope': data_scope, 'training_ids': list(frame.animal_id), 'report': report},
                out / report['fixed_C1_final_fit']['artifact'])
    (out / 'continued_training_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
