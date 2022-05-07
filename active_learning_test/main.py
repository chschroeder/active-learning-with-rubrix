import os
import datetime

import datasets
import numpy
import rubrix as rb

from pathlib import Path

from active_learning_test.active_learner import (
    build_active_learner,
    convert_to_small_text_dataset,
    initialize_active_learner
)
from active_learning_test.rb_streams import DatasetQueryStream


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
        stream = DatasetQueryStream(
            dataset="active-learning-test-batch",
            unique=True,
            query="status:Validated and metadata.batch_id:{batch_id}",
            batch_id=batch_idx,
        )
        for data in stream(start_from=datetime.datetime.utcnow(), batch_size=len(queried_indices)):
            new_labels = [label_name_to_idx[r.annotation] for r in data]
            active_learner.update(numpy.array(new_labels))

            batch_idx += +1
            stream.query_params = {"batch_id": batch_idx}

            queried_indices = active_learner.query()
            log_next_batch(batch_idx, trec_dataset, queried_indices)

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
            id=f"{batch_idx}_{idx}",
            text=text,
            prediction=[
                (label, 0.0)
                for label in trec_dataset["train"].features["label-coarse"].names
            ],
            metadata={"batch_id": batch_idx},
        )
        for idx, text in enumerate(texts)
    ]
    print(f"Logging records for batch {batch_idx}")
    rb.log(records, name=f"active-learning-test-batch")


if __name__ == '__main__':
    main()
