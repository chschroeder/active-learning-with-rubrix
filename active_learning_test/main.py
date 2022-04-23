import os
import time

import datasets
import numpy as np
import rubrix as rb

from pathlib import Path

from active_learning_test.active_learner import (
    build_active_learner,
    convert_to_small_text_dataset,
    initialize_active_learner
)


def main():
    trec_dataset = datasets.load_dataset('trec')
    label_names = trec_dataset['train'].features['label-coarse'].names

    trec_dataset_st = convert_to_small_text_dataset(trec_dataset)
    active_learner = build_active_learner(trec_dataset_st, len(label_names))

    initial_indices = initialize_active_learner(active_learner, trec_dataset_st.y)
    initialize_rubrix(initial_indices, trec_dataset, label_names)

    main_loop(active_learner, trec_dataset, label_names)


def initialize_rubrix(initial_indices, trec_dataset, label_names):

    texts = [trec_dataset['train']['text'][i] for i in initial_indices]
    labels = [trec_dataset['train']['label-coarse'][i] for i in initial_indices]

    records = [
        rb.TextClassificationRecord(
            id=idx,
            text=text,
            annotation=label_names[labels[idx]],
            status='Validated'
        )
        for idx, text in enumerate(texts)
    ]
    rb.log(records, name='active-learning-test-batch-initial')


def main_loop(active_learner, trec_dataset, label_names):

    label_name_to_idx = dict({
        name: i
        for i, name in enumerate(label_names)
    })

    batch_idx = 0

    queried_indices = active_learner.query()
    log_next_batch(batch_idx, trec_dataset, queried_indices)

    try:
        while True:  # TODO: this is ugly, we need a better abort condition here
            df = rb.load(f'active-learning-test-batch-{batch_idx}', query='status:*')
            if (df['status'] == 'Validated').all():

                new_labels = df['annotation'].apply(lambda x: label_name_to_idx[x]).to_numpy()
                print(new_labels)
                active_learner.update(new_labels)

                batch_idx += + 1
                queried_indices = active_learner.query()
                log_next_batch(batch_idx, trec_dataset, queried_indices)

            else:
                time.sleep(3)

            print(df)
            print(df['status'])
            print(df['prediction'])
            print(df['annotation'])

    except KeyboardInterrupt as e:
        print('\n-- Exit initiated.')

        outfile = Path(os.getcwd(), 'active_learner.pkl')
        active_learner.save(outfile)
        print(f'-- Active learning has been serialized to: {outfile}')
        print('-- Exit.')


def log_next_batch(batch_idx, trec_dataset, queried_indices):
    texts = [trec_dataset['train']['text'][i] for i in queried_indices]
    records = [
        rb.TextClassificationRecord(
            id=idx,
            text=text,
            prediction=[(label, 0.0)
                        for label in trec_dataset['train'].features['label-coarse'].names]
        )
        for idx, text in enumerate(texts)
    ]
    rb.log(records, name=f'active-learning-test-batch-{batch_idx}')


if __name__ == '__main__':
    main()
