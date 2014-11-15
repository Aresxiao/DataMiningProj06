import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;


public class RandomForest {
	
	
	int N;
	int totalSampleNum;
	int dimensionNum;
	DataSet dataSet;
	ArrayList<DecisionTree> decisionTrees;
	public RandomForest(DataSet dataSet){
		this.dataSet = dataSet;
		N=10;
		totalSampleNum = dataSet.getTotalSampleNum();
		dimensionNum = dataSet.getDimensionNum();
		decisionTrees = new ArrayList<DecisionTree>();
		for(int i = 0;i < N;i++){
			int deepth = 5*i+20;
			int leastLeaf = 5*i+40;
			DecisionTree dt = new DecisionTree(dataSet, deepth, leastLeaf);
			decisionTrees.add(dt);
		}
	}
	
	public void trainRandomForest(double[][] dataMatrix){
		int len = dataMatrix.length;
		for(int i = 0;i < N;i++){
			double[][] extractData = new double[len][dimensionNum];
			for(int x = 0;x < len;x++){
				int row = (int)(Math.random()*len);
				for(int j = 0;j < dimensionNum;j++){
					extractData[x][j] = dataMatrix[row][j];
				}
			}
			DecisionTree dt = decisionTrees.get(i);
			dt.trainDT(extractData, dataSet.getAttributeList(), dataSet.getContinuousArrayList());
		}
	}
	
	public double classifyBtRandomForest(double[] testData){
		HashMap<Double, Double> classMap = new HashMap<Double, Double>();
		for(int i = 0;i < N;i++){
			DecisionTree dt = decisionTrees.get(i);
			TreeNode treeNode = dt.getRootNode();
			double predict = dt.classifyByDT(testData, treeNode);
			if(classMap.containsKey(predict)){
				double val = classMap.get(predict);
				classMap.put(predict, val+1);
			}
			else {
				classMap.put(predict, 1.0);
			}
		}
		double calTheme = 0;
		double maxCount = 0;
		Iterator iterator = classMap.entrySet().iterator();
		while (iterator.hasNext()) {
			Map.Entry<Double, Double> entry= (Entry<Double, Double>) iterator.next();
			double singleTheme = entry.getKey();
			double count = entry.getValue();
			if(maxCount<count){
				maxCount = count;
				calTheme = singleTheme;
			}
		}
		return calTheme;
	}
	
	public void tenFoldCrossValidation(){
		double[][] dataMatrix = dataSet.getDataMatrix();
		int totalSmapleNum = dataSet.getTotalSampleNum();
		int dimensionNum = dataSet.getDimensionNum();
		System.out.println("totalSmapleNum = "+totalSmapleNum+",dimensionNum = "+dimensionNum);
		ArrayList<Double> accuracyArrayList = new ArrayList<Double>();
		for(int k = 0;k < 10;k++){
			ArrayList<Integer> testPostRowArrayList = new ArrayList<Integer>();
			ArrayList<Integer> trainPostRowArrayList = new ArrayList<Integer>();
			
			for(int i = 0;i < totalSmapleNum;i++){
				if(i%10==k){
					testPostRowArrayList.add(i);
				}
				else {
					trainPostRowArrayList.add(i);
				}
			}
			
			int testSize = testPostRowArrayList.size();
			int trainSize = trainPostRowArrayList.size();
			double[][] testData = new double[testSize][dimensionNum];
			double[][] trainData = new double[trainSize][dimensionNum];
			
			for(int i = 0;i < testPostRowArrayList.size();i++){
				int row = testPostRowArrayList.get(i);
				for(int j = 0;j<dimensionNum;j++){
					testData[i][j] = dataMatrix[row][j];
				}
			}
			for(int i = 0;i < trainPostRowArrayList.size();i++){		//到这里之前都是用来划分数据。
				int row = trainPostRowArrayList.get(i);
				for(int j = 0;j < dimensionNum;j++){
					trainData[i][j] = dataMatrix[row][j];
				}
			}
			
			trainRandomForest(trainData);
			double sum = 0;
			for(int i = 0;i < testSize;i++){
				double realTheme = testData[i][dimensionNum-1];
				double calTheme = classifyBtRandomForest(testData[i]);
				if(calTheme == realTheme){
					sum++;
				}
			}
			sum = sum/testSize;
			accuracyArrayList.add(sum);
			System.out.println("k="+k+"次，准确率为："+sum);
		}
		
		double correctSum = 0;
		double averageRatio = 0;
		double variance = 0;
		for(int i = 0;i < accuracyArrayList.size();i++){
			correctSum += accuracyArrayList.get(i);
		}
		averageRatio = correctSum/accuracyArrayList.size();
		for(int i = 0;i < accuracyArrayList.size();i++){
			double d = accuracyArrayList.get(i);
			variance+=(d-averageRatio)*(d-averageRatio);
		}
		variance = variance/accuracyArrayList.size();
		variance = Math.sqrt(variance);
		
		System.out.println("平均准确率为: "+averageRatio+",标准差为："+variance);
		
	}
}
