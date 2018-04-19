package org.dma.gbdt4spark.objective.loss;

import org.dma.gbdt4spark.objective.metric.EvalMetric;

import java.io.Serializable;

public interface Loss extends Serializable {
    Kind getKind();

    EvalMetric.Kind defaultEvalMetric();

    public enum Kind {
        RMSE("rmse"),
        BinaryLogistic("binary:logistic"),
        MultiLogistic("multi:logistic");

        private final String kind;

        private Kind(String kind) {
            this.kind = kind;
        }

        @Override
        public String toString() {
            return kind;
        }
    }
}
