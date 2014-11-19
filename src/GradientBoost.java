import java.util.ArrayList;


public class GradientBoost {
	
	DataSet dataSet;
	int totalSample;
	int dimensionNum;
	int N;
	int C;
	double step;
	ArrayList<ArrayList<GradientDecisionTree>> gradientDTLists;
	
	public GradientBoost(DataSet ds){
		this.dataSet = ds;
		totalSample = dataSet.getTotalSampleNum();
		dimensionNum = dataSet.getDimensionNum();
		N = 4;
		gradientDTLists = new ArrayList<ArrayList<GradientDecisionTree>>();
		C = 10;
		step = 0.01;
	}
	
	public void trainGradientBoost(double[][] trainData){
		gradientDTLists.clear();
		int trainDataSize = trainData.length;
		for(int i = 0;i < C;i++){								//得到一个10*N的GradientDecisionTree矩阵
			ArrayList<GradientDecisionTree> list = new ArrayList<GradientDecisionTree>();
			for(int j = 0;j < N;j++){
				int maxDeepth = 50;
				int leastLeaf = 5*j+30;
				GradientDecisionTree gdt = new GradientDecisionTree(dataSet, maxDeepth, leastLeaf);
				list.add(gdt);
			}
			gradientDTLists.add(list);
		}
		
		for(int i = 0;i < C;i++){
			ArrayList<GradientDecisionTree> gdtTrees = gradientDTLists.get(i);
			double[][] modifiedData = new double[trainDataSize][];
			for(int x = 0;x < trainDataSize;x++){
				modifiedData[x] = trainData[x].clone();
				if(modifiedData[x][dimensionNum-1] == i){
					modifiedData[x][dimensionNum-1] = 1;
				}
				else {
					modifiedData[x][dimensionNum-1] = 0;
				}
			}
			double[] predictVal = new double[trainDataSize];
			
			for(int j = 0;j < N;j++){
				GradientDecisionTree gdt = gdtTrees.get(j);
				//System.out.println("开始训练i = "+i+",j = "+j);
				gdt.trainDT(modifiedData, dataSet.getAttributeList(), dataSet.getContinuousArrayList());
				for(int x = 0;x < trainDataSize;x++){
					TreeNode node = gdt.getRootNode();
					predictVal[x] = gdt.regressionByDT(modifiedData[x], node);
					modifiedData[x][dimensionNum-1] = modifiedData[x][dimensionNum-1] - step*predictVal[x];
				}
				//System.out.println("结束训练i = "+i+",j = "+j);
			}
			
		}
		
	}
	
	public double classByGradientBoost(double[] testData){
		double[] predictVal = new double[C];
		for(int i = 0;i < C;i++){
			ArrayList<GradientDecisionTree> gdtTrees = gradientDTLists.get(i);
			predictVal[i] = 0;
			for(int j = 0;j < N;j++){
				GradientDecisionTree gdt = gdtTrees.get(j);
				TreeNode node = gdt.getRootNode();
				double val = gdt.regressionByDT(testData, node);
				predictVal[i] += val;
			}
		}
		int max = 0;
		for(int i = 0;i < C;i++){
			if(predictVal[i] > predictVal[max])
				max = i;
		}
		return max;
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
			
			trainGradientBoost(trainData);
			double sum = 0;
			
			for(int i = 0;i < testSize;i++){
				double realTheme = testData[i][dimensionNum-1];
				double calTheme = classByGradientBoost(testData[i]);
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
		
		System.out.println("Gradient Boost平均准确率为: "+averageRatio+",标准差为："+variance);
	}
	
}










