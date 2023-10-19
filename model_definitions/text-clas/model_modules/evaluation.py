from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from teradataml import DataFrame, in_schema, TextParser, NaiveBayesTextClassifierPredict
from aoa import (
    save_plot,
    aoa_create_context,
    ModelContext
)

import json


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    model = DataFrame(in_schema("lc250058", "modelo_texto"))
    stopwords = DataFrame(in_schema("lc250058","stopwords"))
    test_df = DataFrame.from_query(context.dataset_info.sql)

    print("Starting data cleaning...")

    TextParserTest = TextParser(data=test_df,
                                text_column="detalle",
                                punctuation="\<>!#$%&[]()*+,-./:;?@^_`{|}~''",
                                object=stopwords,
                                remove_stopwords=True,
                                accumulate=["doc_id", "target"])

    print("Scoring")

    nbt_predict_out = NaiveBayesTextClassifierPredict(object=model,
                                                      newdata=TextParserTest.result,
                                                      input_token_column='token',
                                                      doc_id_columns='doc_id',
                                                      model_type="MULTINOMIAL",
                                                      model_token_column='token',
                                                      model_category_column='category',
                                                      accumulate='target',
                                                      responses=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                                                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S'],
                                                      newdata_partition_column='doc_id')

    results = nbt_predict_out.result[['prediction', 'target']].to_pandas(all_rows=True)
    y_pred = results.prediction
    y_test = results.target

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred, average='macro'))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    save_plot('Confusion Matrix', context=context)

    print("Finish")
