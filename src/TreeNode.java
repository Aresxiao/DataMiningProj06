import java.util.ArrayList;
import java.util.HashSet;


public class TreeNode {
	int category;
	
	String nodeName;
	int attributeValue;
	double targetValue;
	ArrayList<TreeNode> childTreeNodes;
	HashSet<Integer> nodeDataSet;
	int nodeId;
	double splitVal;
	int continuous;
	
	public TreeNode(String nodeName){
		this.nodeName = nodeName;
		childTreeNodes = new ArrayList<TreeNode>();
		nodeDataSet = new HashSet<Integer>();
	}
	public TreeNode(){
		childTreeNodes = new ArrayList<TreeNode>();
		nodeDataSet = new HashSet<Integer>();
	}
	
	public HashSet<Integer> getNodeDataSet(){
		return nodeDataSet;
	}
	
	public void addSampleDataToSet(int sample){
		this.nodeDataSet.add(sample);
	}
	
	public void setNodeName(String nodeName){
		this.nodeName = nodeName;
	}
	
	public String getNodeName(){
		return nodeName;
	}
	
	public ArrayList<TreeNode> getChildTreeNodes(){
		return childTreeNodes;
	}
	
	public void setChildTreeNodes(ArrayList<TreeNode> childTreeNode){
		this.childTreeNodes = childTreeNode;
	}
	
	public void setAttributeValue(int value){
		this.attributeValue = value;
	}
	
	public int getAttributeValue(){
		return attributeValue;
	}
	
	public void setNodeId(int id){
		nodeId = id;
	}
	
	public int getId(){
		return nodeId;
	}
	
	public void setSplitVal(double splitVal){
		this.splitVal = splitVal;
	}
	
	public double getSplitVal(){
		return splitVal;
	}
	
	public void setTartgetValue(double targetValue){
		this.targetValue = targetValue;
	}
	
	public double getTargetValue(){
		return targetValue;
	}
	
	public void setContinuous(int flag){
		continuous = flag;
	}
	
	public int getContinuous(){
		return continuous;
	}
}
