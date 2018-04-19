package org.dma.gbdt4spark.algo.gbdt.tree;

import org.dma.gbdt4spark.tree.basic.Tree;
import org.dma.gbdt4spark.tree.param.GBDTParam;

public class GBTTree extends Tree<GBDTParam, GBTNode> {
    public GBTTree(GBDTParam param) {
        super(param);
        GBTNode root = new GBTNode(0, null, param.numClass);
        this.nodes.put(0, root);
    }
}
