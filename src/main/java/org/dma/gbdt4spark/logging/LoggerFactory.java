package org.dma.gbdt4spark.logging;

public class LoggerFactory {
    public static Logger getLogger(Class clazz) {
        return new Logger(clazz.getName());
    }
}
