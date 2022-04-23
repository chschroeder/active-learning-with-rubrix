# active-learning-with-rubrix-test

This is a proof of concept to use [rubrix](https://github.com/recognai/rubrix) in combination 
with [small-text](https://github.com/webis-de/small-text/blob/master/small_text/active_learner.py)
for active-learning-based text classification. 

See the [discussion here](https://github.com/recognai/rubrix/discussions/1398).

## Usage:

Install python dependencies:

```bash
pip install -r requirements.txt
```

Then execute the steps to start rubrix as shown in their README.md:

```bash
docker run -d --name elasticsearch-for-rubrix -p 9200:9200 -p 9300:9300 -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2
```

```bash
python -m rubrix
```

Finally, to start this application:

```
python -m active_learning_test.main
```
