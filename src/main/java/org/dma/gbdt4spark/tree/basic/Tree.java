package org.dma.gbdt4spark.tree.basic;

import org.dma.gbdt4spark.tree.param.TreeParam;

import java.io.Serializable;
import java.util.Map;
import java.util.TreeMap;

public abstract class Tree<TParam extends TreeParam, Node extends TNode> implements Serializable {
    protected final TParam param;
    protected Map<Integer, Node> nodes; // nodes in the tree

    public Tree(TParam param) {
        this.param = param;
        this.nodes = new TreeMap<>();
    }

    public TParam getParam() {
        return this.param;
    }

    public Node getRoot() {
        return this.nodes.get(0);
    }

    public Node getNode(int nid) {
        return this.nodes.get(nid);
    }

    public void setNode(int nid, Node node) {
        this.nodes.put(nid, node);
    }

    public Map<Integer, Node> getNodes() {
        return this.nodes;
    }

    public int size() {
        return nodes.size();
    }

}
