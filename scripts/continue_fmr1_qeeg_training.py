"""Continue exploratory Fmr1 genotype SVM development, never behavioral onset training."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from psi_qeeg.animal_genotype import SOURCE_FEATURES, train_source_genotype, verify_source_snapshot
from validate_public_fmr1_qeeg import FILES, animal_vectors, source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/model_organism/as7cz'))
    parser.add_argument('--out', type=Path, default=Path('outputs/fmr1_continued_training'))
    parser.add_argument('--source-provenance', type=Path, help='Explicit retained metadata JSON for checksum-verified offline cached-source reuse')
    args = parser.parse_args(argv)
    provenance = (verify_source_snapshot(args.cache, args.source_provenance, {name: digest for name, (_, digest) in FILES.items()})
                  if args.source_provenance else source(args.cache))
    first, features, excluded1 = animal_vectors(pd.read_csv(args.cache / 'experiment1.csv'), 'experiment1')
    second = pd.read_csv(args.cache / 'experiment2.csv')
    if set(second.Drug) != {'Sal', 'ClzL', 'ClzH', 'MPEP'}:
        raise ValueError('Review changed experimental condition labels')
    second, matched, excluded2 = animal_vectors(second[second.Drug == 'Sal'], 'experiment2_saline')
    if features != SOURCE_FEATURES or matched != features or set(first.animal_id) & set(second.animal_id):
        raise ValueError('Independent experiments and complete source feature contract are required')
    combined = pd.concat([first, second], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    vectors = args.out / 'training_animal_vectors.csv'; combined.to_csv(vectors, index=False)
    report = train_source_genotype(combined, features, args.out,
                                  data_scope='real_published_processed_animal_qeeg_features',
                                  source_provenance=provenance,
                                  progress=lambda item: print(json.dumps(item), flush=True))
    report.update(excluded_animals=excluded1 + excluded2,
                  source_metadata_mode='retained_verified_snapshot' if args.source_provenance else 'refreshed_osf_api',
                  source_metadata_snapshot_sha256=hashlib.sha256(args.source_provenance.read_bytes()).hexdigest() if args.source_provenance else None,
                  training_vectors_sha256=hashlib.sha256(vectors.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  trainer_sha256=hashlib.sha256((Path(__file__).parents[1] / 'src/psi_qeeg/animal_genotype.py').read_bytes()).hexdigest())
    (args.out / 'continued_training_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('independent_animals', 'model_fits', 'nested_animal_metrics',
                                                'fixed_C1_animal_metrics')}, indent=2))
    print(json.dumps({'final_parameters': report['final_fit']['parameters'],
                      'experiment_holdouts': {name: {key: value[key] for key in ('animal_balanced_accuracy', 'selected_parameters')}
                                              for name, value in report['experiment_holdouts'].items()}}, indent=2))
    return report


if __name__ == '__main__':
    main()
