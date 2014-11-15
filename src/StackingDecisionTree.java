
import java.util.ArrayList;

public class StackingDecisionTree {

	int N;
	int K;
	int dimensionNum;
	DataSet dataSet;
	ArrayList<DecisionTree> decisionTrees;
	DecisionTree metaTree;
	int flag;
	public StackingDecisionTree(DataSet dataSet){
		
		flag = dataSet.getFlag();
		N = 3;
		K = 10;
		this.dataSet = dataSet;
		dimensionNum = dataSet.getDimensionNum();
		decisionTrees = new ArrayList<DecisionTree>();
		for(int i = 0;i < N;i++){
			int deepth = 4*i+20;
			int leastLeaf = 10*i+15;
			DecisionTree dt = new DecisionTree(dataSet, deepth, leastLeaf);
			decisionTrees.add(dt);
		}
		
	}
	
	public void trainStackingDT(double[][] trainData) {
		
		int len = trainData.length;
		
		ArrayList<ArrayList<Integer>> kGroupsArrayList = new ArrayList<ArrayList<Integer>>();
		for(int i = 0;i < K;i++){
			ArrayList<Integer> list = new ArrayList<Integer>();
			kGroupsArrayList.add(list);
		}
		for(int i = 0;i < len;i++){
			int y = i%K;
			kGroupsArrayList.get(y).add(i);
		}
		
		double[][] metaData = new double[len][N+1];
		int metaDataRowFlag = 0;
		for(int i = 0;i < K;i++){
			
			
			int diSize = kGroupsArrayList.get(i).size();		//这一部分把数据分成Di和X-Di,X-Di用来训练决策树，Di用来预测这部分。
			double[][] diData = new double[diSize][dimensionNum];
			for(int k = 0;k < diSize;k++){
				int row = kGroupsArrayList.get(i).get(k);
				for(int x = 0;x < dimensionNum;x++){
					diData[k][x] = trainData[row][x];
				}
			}
			int remainSize = len - diSize;
			double[][] remainData = new double[remainSize][dimensionNum];
			int rowFlag = 0;
			for(int y = 0;y < K;y++){			//得到meta数据，用于训练meta tree
				if(y != i){
					ArrayList<Integer> list = kGroupsArrayList.get(y);
					for(int k = 0;k < list.size();k++){
						int row = list.get(k);
						for(int x = 0;x < dimensionNum;x++){
							remainData[rowFlag][x] = trainData[row][x];
						}
						rowFlag++;
					}
				}
			}
			//System.out.println("remainSize = "+remainSize+"rowFlag = "+rowFlag);
			for(int j = 0;j < N;j++){
				
				DecisionTree dt = decisionTrees.get(j);
				dt.trainDT(remainData, dataSet.getAttributeList(), dataSet.getContinuousArrayList());
				double sum = 0;
				for(int y = 0;y < diSize;y++){
					TreeNode treeNode = dt.getRootNode();
					double calTheme = 0;
					if(flag==0){
						calTheme = dt.classifyByDT(diData[y], treeNode);
						if(calTheme == diData[y][dimensionNum-1]){
							sum++;
						}
						metaData[metaDataRowFlag+y][j] = calTheme;
					}
					else {
						calTheme = dt.regressionByDT(diData[y], treeNode);
						metaData[metaDataRowFlag+y][j] = calTheme;
					}
					
				}
				sum = sum/diSize;
				//System.out.println("i = "+i+",j = "+j+",准确率为："+sum);
			}
			for(int y = 0;y < diSize;y++){
				double realTheme = diData[y][dimensionNum-1];
				metaData[metaDataRowFlag+y][N] = realTheme;
			}
			metaDataRowFlag = metaDataRowFlag + diSize;
		}
		
		
		ArrayList<Integer> metaContinuousList = new ArrayList<Integer>();
		ArrayList<Integer> metaDataAttributeIndexList = new ArrayList<Integer>();
		
		for(int i = 0;i < (N+1);i++){
			metaContinuousList.add(1);
		}
		
		
		for(int i = 0;i< N;i++){
			metaDataAttributeIndexList.add(i);
		}
		
		metaTree = new DecisionTree(flag,metaContinuousList,30,5);		//训练meta tree
		metaTree.trainDT(metaData, metaDataAttributeIndexList, metaContinuousList);
		
		for(int i = 0;i < N;i++){
			DecisionTree dt = decisionTrees.get(i);
			dt.trainDT(trainData, dataSet.getAttributeList(), dataSet.getContinuousArrayList());	//用full data来训练N棵Tree
		}
		
	}
	
	public double classifyByDT(double[] test){		//分类
		double[] metaLevelFeature = new double[N+1];
		metaLevelFeature[N] = test[dimensionNum-1];
		for(int i = 0;i < N;i++){
			DecisionTree dt = decisionTrees.get(i);
			TreeNode treeNode = dt.getRootNode();
			double predict = dt.classifyByDT(test, treeNode);
			metaLevelFeature[i] = predict;
		}
		
		TreeNode treeNode = metaTree.getRootNode();
		
		double realVal = metaTree.classifyByDT(metaLevelFeature, treeNode);
		return realVal;
		
	}
	
	public double regressionByDT(double[] test){	//回归
		double[] metaLevelFeature = new double[N+1];	
		metaLevelFeature[N] = 0;
		for(int i = 0;i < N;i++){
			DecisionTree dt = decisionTrees.get(i);
			TreeNode treeNode = dt.getRootNode();
			double regression = dt.regressionByDT(test, treeNode);
			metaLevelFeature[i] = regression;
		}
		
		TreeNode treeNode = metaTree.getRootNode();
		double rsltReg = metaTree.classifyByDT(test, treeNode);
		return rsltReg;
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
			trainStackingDT(trainData);
			double sum = 0;
			if(flag == 0){
				for(int i = 0; i < testSize;i++){
					double theoryTheme = testData[i][dimensionNum-1];
					double predictTheme = classifyByDT(testData[i]);
					if(theoryTheme==predictTheme){
						sum++;
					}
				}
				sum = sum/testSize;
				System.out.println("第k = "+k+" 次正确率为: "+sum);
				accuracyArrayList.add(sum);
			}
			else {
				for(int i = 0;i < testSize;i++){
					double theoryValue = testData[i][dimensionNum-1];
					double realValue = regressionByDT(testData[i]);
					sum += (theoryValue - realValue)*(theoryValue - realValue);
				}
				sum = sum/testSize;
				sum = Math.sqrt(sum);
				System.out.println("第k = "+k+" 次,MSE为: "+sum);
				accuracyArrayList.add(sum);
			}
			
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
		if(flag==0){
			System.out.println("平均准确率为: "+averageRatio+",标准差为："+variance);
		}
		else {
			System.out.println("平均MSE为: "+averageRatio+",标准差为："+variance);
		}
	}
}
