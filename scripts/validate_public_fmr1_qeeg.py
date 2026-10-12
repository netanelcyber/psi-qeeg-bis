"""Source-verified Fmr1-KO2 qEEG genotype benchmark, not behavioral onset prediction.

Uses deposited Pow vectors, frequency labels and locations without asserting
unknown native power units or converting frequency indices to physical Hz.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

FILES = {
    'experiment1.csv': ('641b163bc74723167c10b272', 'c6d8358847723bec04fb75a979b17bbbd00331c4f45ce4483a0a5b5ef2e12221'),
    'experiment2.csv': ('641b163b28e5c5153d937671', '3d871d826780933a4bf2124e53a8bc77b8584768aa48d48899ace5d26d5ad376'),
}


def source(cache):
    cache.mkdir(parents=True, exist_ok=True); provenance = []
    for filename, (file_id, expected) in FILES.items():
        with urlopen('https://api.osf.io/v2/files/' + file_id + '/', timeout=30) as response:
            info = json.load(response)['data']
        attributes = info['attributes']
        if attributes['extra']['hashes']['sha256'] != expected:
            raise ValueError('Source changed; review and repin before evaluating')
        path = cache / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            with urlopen(info['links']['download'], timeout=30) as response: data = response.read()
            if hashlib.sha256(data).hexdigest() != expected: raise ValueError('Download checksum mismatch')
            temporary = path.with_suffix('.csv.part'); temporary.write_bytes(data); temporary.replace(path)
        if path.stat().st_size != attributes['size']: raise ValueError('Source size mismatch')
        provenance.append({'filename': filename, 'source_filename': attributes['name'],
                           'source_url': info['links']['download'], 'sha256': expected, 'bytes': path.stat().st_size})
    return provenance


def animal_vectors(frame, experiment):
    required = {'AnID', 'GT', 'Freq', 'Loc', 'Pow'}
    if not required <= set(frame) or frame[list(required - {'Pow'})].isna().any().any():
        raise ValueError('Source qEEG table is incomplete')
    if set(frame.GT) != {'WT', 'KO'} or frame.groupby('AnID').GT.nunique().max() != 1:
        raise ValueError('Preserve consistent source WT/KO genotypes')
    if frame.duplicated(['AnID', 'Loc', 'Freq']).any(): raise ValueError('Repeated condition/frequency rows cannot be independent animals')
    names = [f'source_{location}__frequency_label_{frequency:02d}__log10_Pow' for location in ('Front', 'Temp') for frequency in range(1, 100)]
    rows, excluded = [], []
    for animal, part in frame.groupby('AnID', sort=True):
        expected = {(location, frequency) for location in ('Front', 'Temp') for frequency in range(1, 100)}
        if set(zip(part.Loc, part.Freq)) != expected or not np.isfinite(part.Pow).all() or (part.Pow <= 0).any():
            excluded.append({'animal_id': animal, 'experiment': experiment, 'reason': 'Incomplete/nonfinite/nonpositive deposited power vector; no imputation'})
            continue
        lookup = part.set_index(['Loc', 'Freq']).Pow
        values = [float(np.log10(lookup.loc[(location, frequency)])) for location in ('Front', 'Temp') for frequency in range(1, 100)]
        rows.append({'animal_id': animal, 'genotype': part.GT.iat[0], 'experiment': experiment, **dict(zip(names, values))})
    if not rows: raise ValueError('No complete animal power vectors')
    return pd.DataFrame(rows), names, excluded


def main(argv=None):
    try:
        import joblib
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ValueError("Genotype fitting requires: pip install 'psi-qeeg-bis[ml]'") from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/model_organism/as7cz'))
    parser.add_argument('--out', type=Path, default=Path('outputs/fmr1_qeeg_genotype'))
    args = parser.parse_args(argv); provenance = source(args.cache)
    first = pd.read_csv(args.cache / 'experiment1.csv'); second = pd.read_csv(args.cache / 'experiment2.csv')
    if set(second.Drug) != {'Sal', 'ClzL', 'ClzH', 'MPEP'}: raise ValueError('Review changed experimental condition labels')
    # Select the saline condition before fitting; the other conditions are not
    # treated as new animals or as psychiatric/behavioral episode outcomes.
    first, features, excluded1 = animal_vectors(first, 'experiment1')
    second, same, excluded2 = animal_vectors(second[second.Drug == 'Sal'], 'experiment2_saline')
    if same != features or set(first.animal_id) & set(second.animal_id): raise ValueError('Independent experiments and matched feature contract required')
    args.out.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    factory = lambda: make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, class_weight='balanced'))
    for name, train, test in (('primary_experiment1_to_experiment2', first, second),
                             ('supplementary_experiment2_to_experiment1', second, first)):
        model = factory().fit(train[features].to_numpy(float), train.genotype)
        prediction = model.predict(test[features].to_numpy(float))
        table = test[['animal_id', 'genotype', 'experiment']].copy(); table['prediction'] = prediction
        table.to_csv(args.out / (name + '_held_out_animals.csv'), index=False)
        joblib.dump({'model_kind': 'source_qeeg_genotype_classifier', 'model': model, 'features': features,
                     'target': 'source_WT_KO_genotype_not_behavioral_onset'}, args.out / (name + '.joblib'))
        comparisons[name] = {'training_animals': len(train), 'test_animals': len(test),
                             'training_ids': list(train.animal_id), 'test_ids': list(test.animal_id),
                             'animal_balanced_accuracy': float(balanced_accuracy_score(test.genotype, prediction)),
                             'confusion_matrix': {'labels': ['KO', 'WT'], 'rows_true_cols_pred': confusion_matrix(test.genotype, prediction, labels=['KO', 'WT']).tolist()}}
    combined = pd.concat([first, second], ignore_index=True); combined.to_csv(args.out / 'deposited_power_animal_vectors.csv', index=False)
    report = {'dataset': 'Neuronal Oscillation Dynamics Fmr1 KO2 mice', 'source': 'https://osf.io/as7cz/',
              'article': 'https://doi.org/10.1093/cercor/bhae201', 'species': 'Mus musculus', 'model_family': 'Fmr1',
              'data_scope': 'real_published_processed_animal_qeeg_features', 'target': 'WT_KO_genotype',
              'power_provenance': 'Deposited Pow only; no recomputation from raw EEG, no invented physical-unit/frequency conversion',
              'features': features, 'feature_count': len(features), 'independent_animals': len(combined),
              'animals_per_genotype': {k: int(v) for k,v in combined.groupby('genotype').animal_id.nunique().items()},
              'source_provenance': provenance, 'excluded_animals': excluded1 + excluded2,
              'comparisons': comparisons, 'protocol': 'Prespecified two-location/all-99-source-label Pow vectors; log10; training-only StandardScaler; fixed linear balanced SVC C=1; no tuning or outcome-based feature selection',
              'behavioral_onset_model_trained_on_real_data': False, 'human_transfer_validation': 'not_established',
              'limitations': ['Small single-laboratory cohorts; no uncertainty intervals or significance test',
                              'Frequency labels and power units used as deposited; no asserted Hz/µV² mapping',
                              'Genotype discrimination is not behavioral-event prediction or autism diagnosis',
                              'No raw continuous EEG or independently timed behavioral onsets in these source CSVs'],
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / 'fmr1_qeeg_genotype_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('data_scope','independent_animals','feature_count','excluded_animals','comparisons','behavioral_onset_model_trained_on_real_data')}, indent=2))


if __name__ == '__main__': main()
