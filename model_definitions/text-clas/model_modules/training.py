from teradataml import DataFrame, in_schema, TextParser, NaiveBayesTextClassifierTrainer, copy_to_sql
from aoa import (
    aoa_create_context,
    ModelContext
)


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    # read training dataset from Teradata and convert to pandas

    train_df = DataFrame.from_query(context.dataset_info.sql)
    stopwords = DataFrame(in_schema("lc250058", "stopwords"))
    
    print("Starting data cleaning...")

    TextParserTrain = TextParser(data=train_df,
                                 text_column="detalle",
                                 punctuation="\<>!#$%&[]()*+,-./:;?@^_`{|}~''",
                                 object=stopwords,
                                 remove_stopwords=True,
                                 accumulate=["doc_id", "target"])

    print("Starting training...")

    # fit model to training data
    NaiveBayesTextClassifierTrainer_out = NaiveBayesTextClassifierTrainer(data=TextParserTrain.result,
                                                                          token_column="token",
                                                                          doc_id_columns="doc_id",
                                                                          doc_category_column="target",
                                                                          model_type="MULTINOMIAL",
                                                                          data_partition_column="target")

    print("Finished training")

    # export model artefacts
    copy_to_sql(NaiveBayesTextClassifierTrainer_out.result, schema_name="lc250058",
                table_name="modelo_texto", if_exists="replace")

    print("Saved trained model")


