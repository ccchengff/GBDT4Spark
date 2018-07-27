package org.dma.gbdt4spark.logging;

import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Logger implements Serializable {
    private String name;

    private static final SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");

    public Logger(String name) {
        this.name = name;
    }

    public void info(String str) {
        System.out.println(format.format(new Date()) + " INFO " + name + ": " + str);
    }

    public void warn(String str) {
        System.out.println(format.format(new Date()) + " WARN " + name + ": " + str);
    }
}
