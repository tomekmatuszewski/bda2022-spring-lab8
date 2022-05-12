# Databricks notebook source
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

df = spark.read.csv('/FileStore/rotten_tomatoes_reviews.csv', header=True, inferSchema=True) \
    .withColumnRenamed('Freshness', 'label')
df.show(truncate=False)

# COMMAND ----------

tokenizer = Tokenizer(inputCol="Review", outputCol="raw_words")
df1 = tokenizer.transform(df)

# COMMAND ----------

stop_words_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words")
df2 = stop_words_remover.transform(df1)

# COMMAND ----------

hashingTF = HashingTF(inputCol=stop_words_remover.getOutputCol(), outputCol="features")
df3 = hashingTF.transform(df2)

# COMMAND ----------

train, test = df3.randomSplit(weights=[0.8, 0.2], seed=23)

# COMMAND ----------

lr = LogisticRegression(maxIter=10, regParam=0.01, labelCol="label", featuresCol=hashingTF.getOutputCol())
lr_model = lr.fit(train)
df4 = lr_model.transform(test)
df4.select('label', 'prediction', 'Review').show(truncate=False)

# COMMAND ----------

display(df4.select('label', 'prediction', 'Review'))

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol='label')
print(evaluator.evaluate(df4))

# COMMAND ----------

# let see some pictures
from sklearn.metrics import confusion_matrix

y_true = df4.select("label")
y_true = y_true.toPandas()

y_pred = df4.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
cnf_matrix

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# COMMAND ----------

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Confusion matrix without normalization')
plt.show()

# COMMAND ----------

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0, 1], normalize=True,
                      title='Confusion matrix with normalization')
plt.show()

# COMMAND ----------

# ROC for the model
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(lr_model.summary.roc.select('FPR').collect(),
         lr_model.summary.roc.select('TPR').collect())
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# COMMAND ----------

# lets plot ROC (receiver operating characteristic curve), first some helper function
from pyspark.mllib.evaluation import BinaryClassificationMetrics
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

    # now the plot
    import matplotlib.pyplot as plt

    # Returns as a list (false positive rate, true positive rate)
    preds = df4.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
    points = CurveMetrics(preds).get_curve('roc')

    plt.figure()
    x_val = [x[0] for x in points]
    y_val = [x[1] for x in points]
    plt.title('ROC')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.plot(x_val, y_val)
