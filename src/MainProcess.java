import java.io.IOException;
import java.util.ArrayList;


public class MainProcess {
	
	public static void main(String[] args) throws IOException{
		
		
		DataSet dataSet = new DataSet();
		
		dataSet.readPost();
		System.out.println("读完数据");
		DecisionTree dt = new DecisionTree(dataSet, 100, 40);
		dt.tenFoldCrossValidation();
		//RandomForest randomForest = new RandomForest(dataSet);
		//randomForest.tenFoldCrossValidation();
		
	}
	
}
