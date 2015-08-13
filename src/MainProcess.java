import java.io.IOException;
import java.util.ArrayList;


public class MainProcess {
	
	public static void main(String[] args) throws IOException{
		
		/*
		DataSet dataSet1 = new DataSet();
		dataSet1.readPost();
		System.out.println("开始测试Random Forest的正确率");
		RandomForest randomForest = new RandomForest(dataSet1);
		randomForest.tenFoldCrossValidation();
		*/
		System.out.println("开始测试AdaBoost的正确率");
		DataSet dataSet2 = new DataSet();
		dataSet2.readPost();
		AdaBoost boost = new AdaBoost(dataSet2);
		boost.tenFoldCrossValidation();
		
		System.out.println("开始测试Gradient Boost的正确率");
		DataSet dataSet3 = new DataSet();
		dataSet3.readPost();
		GradientBoost gradientBoost = new GradientBoost(dataSet3);
		gradientBoost.tenFoldCrossValidation();
		
	}
	
}
