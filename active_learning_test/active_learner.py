from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from small_text.active_learner import PoolBasedActiveLearner

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory

from small_text.data import SklearnDataset
from small_text.data.sampling import stratified_sampling
from small_text.query_strategies import BreakingTies


def convert_to_small_text_dataset(trec_dataset):

    train_text = [text for text in trec_dataset['train']['text']]
    y_train = [label for label in trec_dataset['train']['label-coarse']]

    vectorizer = TfidfVectorizer(stop_words='english')
    x_train = normalize(vectorizer.fit_transform(train_text))

    return SklearnDataset(x_train, y_train)


def build_active_learner(dataset, num_classes):
    clf_template = ConfidenceEnhancedLinearSVC()
    clf_factory = SklearnClassifierFactory(clf_template, num_classes)
    query_strategy = BreakingTies()

    return PoolBasedActiveLearner(clf_factory, query_strategy, dataset)


def initialize_active_learner(active_learner, y_train):

    indices_initial = stratified_sampling(y_train)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial
