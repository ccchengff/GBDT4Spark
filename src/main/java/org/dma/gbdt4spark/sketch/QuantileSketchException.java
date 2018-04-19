package org.dma.gbdt4spark.sketch;

import org.dma.gbdt4spark.exception.GBDTException;

public class QuantileSketchException extends GBDTException {
    public QuantileSketchException() {
        super();
    }

    public QuantileSketchException(String msg) {
        super(msg);
    }

    public QuantileSketchException(Throwable cause) {
        super(cause);
    }

    public QuantileSketchException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
