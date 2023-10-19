from teradataml import copy_to_sql, DataFrame, in_schema, TextParser, NaiveBayesTextClassifierPredict
from datetime import datetime
from aoa import (
    aoa_create_context,
    ModelContext
)


def score(context: ModelContext, **kwargs):

    aoa_create_context()

    model = DataFrame(in_schema("lc250058", "modelo_texto"))
    stopwords = DataFrame(in_schema("lc250058", "stopwords"))

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    features_tdf = DataFrame.from_query(context.dataset_info.sql)

    print("Starting data cleaning...")

    TextParserTest = TextParser(data=features_tdf,
                                text_column="detalle",
                                punctuation="\<>!#$%&[]()*+,-./:;?@^_`{|}~''",
                                object=stopwords,
                                remove_stopwords=True,
                                accumulate=["doc_id"])

    print("Scoring")

    nbt_predict_out = NaiveBayesTextClassifierPredict(object=model,
                                                      newdata=TextParserTest.result,
                                                      input_token_column='token',
                                                      doc_id_columns='doc_id',
                                                      model_type="MULTINOMIAL",
                                                      model_token_column='token',
                                                      model_category_column='category',
                                                      model_prob_column = 'prob',
                                                      responses=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                                                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S'],
                                                      newdata_partition_column='doc_id')

    print("Finished Scoring")

    # store the predictions

    predictions_pdf = nbt_predict_out.result[['doc_id', 'prediction']].to_pandas(all_rows=True)
    now = datetime.now()
    predictions_pdf["exec_timestamp"] = now

    copy_to_sql(df=predictions_pdf,
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                index=False,
                if_exists="append")

    print("Saved predictions in Teradata")

