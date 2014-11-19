import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;


public class AdaBoost {
	
	double[] classifierWeight;
	double[] sampleWeight;
	int totalSample;
	int dimensionNum;
	DataSet dataSet;
	ArrayList<SimpleDecisionTree> decisionTrees;
	
	int T;
	public AdaBoost(DataSet dataSet){
		this.dataSet = dataSet;
		this.totalSample = dataSet.getTotalSampleNum();
		this.dimensionNum = dataSet.getDimensionNum();
		decisionTrees = new ArrayList<SimpleDecisionTree>();
		T=6;
		
	}
	
	public void trainClassifier(double[][] trainData){
		decisionTrees.clear();
		classifierWeight = new double[T];	//分类器权值
		int len = trainData.length;
		int size = (int)(2*len/3);
		
		sampleWeight = new double[len];		//初始化样本权值
		for (int i = 0; i < len; i++) {
			sampleWeight[i] = 1/(double)(len);
		}
		for(int i = 0;i < T;i++){		//初始化T棵树
			int deepth = 5*i+60;
			int leastLeaf = 4*i+30;
			SimpleDecisionTree simpleDT = new SimpleDecisionTree(dataSet, deepth, leastLeaf);
			decisionTrees.add(simpleDT);
		}
		
		
		for(int i = 0;i < T;i++){
			double[][] extractData = new double[size][dimensionNum];
			HashMap<Integer, Integer> extractMap = new HashMap<Integer,Integer>();
			ArrayList<Integer> errorSampleList = new ArrayList<Integer>();
			double[] weightSum = new double[len];
			for(int x = 0;x < len;x++){
				if(x == 0)
					weightSum[x] = sampleWeight[x];
				else {
					weightSum[x] = sampleWeight[x] + weightSum[x-1];
				}
			}
			
			for(int x = 0;x < size;x++){
				double val = Math.random();
				int row = binarySearch(weightSum, 0, len-1, val);
				extractMap.put(x, row);
				for(int j = 0;j < dimensionNum;j++){
					extractData[x][j] = trainData[row][j];
				}
			}
			
			SimpleDecisionTree dt = decisionTrees.get(i);
			dt.trainDT(extractData, dataSet.getAttributeList(), dataSet.getContinuousArrayList());
			
			for(int x = 0;x < len;x++){								//
				double realTheme = trainData[x][dimensionNum-1];
				TreeNode node = dt.getRootNode();
				double calTheme = dt.classifyByDT(trainData[x], node);
				if(realTheme != calTheme)
					errorSampleList.add(x);
			}
			
			double error = 0;
			for(int x = 0;x < errorSampleList.size();x++){			//计算错误率
				int index = errorSampleList.get(x);
				error += sampleWeight[index];
			}
			
			double alpha = 0.5*Math.log((1-error)/error);			//得到分类器的权重
			classifierWeight[i] = alpha;
			
			for(int x = 0;x < errorSampleList.size();x++){			//更新权重
				int index = errorSampleList.get(x);
				sampleWeight[index] = sampleWeight[index] * (1-error)/error;
			}
			double sum = 0;
			for(int x = 0;x < len;x++){
				sum = sum + sampleWeight[x];
			}
			//System.out.println("error.size = "+errorSampleList.size()+",error = "+error+",alpha = "+alpha+",sum = "+sum);
			for(int x = 0;x < len;x++){								//规范化样本的权值
				sampleWeight[x] = sampleWeight[x]/sum;
				//System.out.print(sampleWeight[x]+" ");
			}
			
		}
	}
	
	public int binarySearch(double[] array,int low,int high,double val){
		if(low > high|| val > array[high]) return -1;
		int mid = (low + high)/2;
		while(high>low){
			if(array[mid]>val)
				high = mid;
			else {
				low = mid + 1;
			}
			mid = (low+high)/2;
		}
		return mid;
	}
	
	
	public double classByAdaBoost(double[] testData){
		HashMap<Double, Double> classMap = new HashMap<Double, Double>();
		for(int i = 0;i < T;i++){
			SimpleDecisionTree simpleDT = decisionTrees.get(i);
			TreeNode node = simpleDT.getRootNode();
			double predict = simpleDT.classifyByDT(testData, node);
			if(classMap.containsKey(predict)){
				double val = classMap.get(predict);
				val = val + classifierWeight[i];
				classMap.put(predict, val);
			}
			else {
				classMap.put(predict, classifierWeight[i]);
			}
		}
		double calTheme = 0;
		double maxCount = 0;
		Iterator iterator = classMap.entrySet().iterator();
		while(iterator.hasNext()){
			Map.Entry<Double, Double> entry = (Entry<Double, Double>) iterator.next();
			
			double curTheme = entry.getKey();
			double curCount = entry.getValue();
			if(maxCount < curCount){
				maxCount = curCount;
				calTheme = curTheme;
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
			
			trainClassifier(trainData);
			double sum = 0;
			
			for(int i = 0;i < testSize;i++){
				double realTheme = testData[i][dimensionNum-1];
				double calTheme = classByAdaBoost(testData[i]);
				if(realTheme == calTheme){
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
		
		System.out.println("AdaBoost平均准确率为: "+averageRatio+",标准差为："+variance);
	}
}
