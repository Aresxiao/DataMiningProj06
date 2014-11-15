import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;



public class InfoGain {
	
	double[][] data;
	ArrayList<Integer> attributeList;
	ArrayList<Integer> continuousArrayList;
	HashMap<Integer, Double> splitValMap;
	
	double[][] leftData;
	double[][] rightData;
	public InfoGain(){
		
	}
	
	public InfoGain(double[][] data,ArrayList<Integer> attributeList,ArrayList<Integer> continuous){
		this.data = new double[data.length][data[0].length];
		for(int i = 0;i<this.data.length;i++){
			for(int j = 0;j<this.data[0].length;j++)
				this.data[i][j] = data[i][j];
		}
		this.attributeList = (ArrayList<Integer>) attributeList.clone();
		
		this.continuousArrayList = (ArrayList<Integer>) continuous.clone(); 
		splitValMap = new HashMap<Integer, Double>();
	}
	
	public double calMinVariance(int i){
		double variance = 0;
		if(continuousArrayList.get(i)==0){
			ArrayList<Double> tempAttributeValueList = new ArrayList<Double>();
			for(int x = 0;x < data.length;x++){
				if(!tempAttributeValueList.contains(data[x][i])){
					tempAttributeValueList.add(data[x][i]);
				}
			}
			double[] tempAttriValue = new double[tempAttributeValueList.size()];
			for(int x = 0;x < tempAttributeValueList.size();x++){
				tempAttriValue[x] = tempAttributeValueList.get(x);
			}
			for(int x = 0;x < tempAttriValue.length;x++){
				int min = x;
				for(int j = x;j < tempAttriValue.length;j++){
					if(tempAttriValue[min] > tempAttriValue[j]){
						min = j;
					}
				}
				if(min != x){
					double t = tempAttriValue[min];
					tempAttriValue[min] = tempAttriValue[x];
					tempAttriValue[x] = t;
				}
			}
			double minVariance = 100000;
			double splitVal = 0;
			for(int x = 0;x < (tempAttriValue.length - 1);x++){
				double mid = (tempAttriValue[x]+tempAttriValue[x+1])/2.0;
				ArrayList<Double> lessSet = new ArrayList<Double>();
				ArrayList<Double> biggerSet = new ArrayList<Double>();
				
				for(int y = 0;y<data.length;y++){
					int lastIndex = data[0].length-1;
					double val = data[y][lastIndex];
					if(data[y][i]<mid){
						lessSet.add(val);
					}
					else{
						biggerSet.add(val);
					}
				}
				double biggerSetSize = biggerSet.size();
				double lessSetSize = lessSet.size();
				
				double v = (biggerSetSize/(biggerSetSize+lessSetSize))*getVariance(biggerSet)+
						(lessSetSize/(lessSetSize+biggerSetSize))*getVariance(lessSet);
				if(minVariance>v){
					minVariance = v;
					splitVal = mid;
				}
			}
			variance = minVariance;
			splitValMap.put(i, splitVal);
		}
		else {
			
			HashSet<Double> tempAttriValue = new HashSet<Double>();
			for(int x = 0;x<data.length;x++){
				tempAttriValue.add(data[x][i]);
			}
			Iterator<Double> iterator = tempAttriValue.iterator();
			double minVariance = 100000;
			double splitVal=0;
			while(iterator.hasNext()){
				double attriVal = iterator.next();
				
				ArrayList<Double> isThisAttributeValArrayList = new ArrayList<Double>();
				ArrayList<Double> notThisAttributeValArrayList = new ArrayList<Double>();
				
				int lastIndex = data[0].length - 1;
				for(int x = 0;x<data.length;x++){
					double theme = data[x][lastIndex];
					
					if(data[x][i]==attriVal){
						isThisAttributeValArrayList.add(theme);
					}
					else {
						notThisAttributeValArrayList.add(theme);
					}
				}
				
				double isSize = isThisAttributeValArrayList.size();
				double notSize = notThisAttributeValArrayList.size();
				//
				double giniGain = (isSize/(isSize+notSize))*getVariance(isThisAttributeValArrayList)+
						(notSize/(isSize+notSize))*getVariance(notThisAttributeValArrayList);
				
				if(minVariance > giniGain){
					minVariance = giniGain;
					splitVal = attriVal;
				}
				
			}
			variance = minVariance;
			splitValMap.put(i, splitVal);
		}
		
		return variance;
	}
	
	public double giniIndex(int i){
		double gini = 0;
		if(continuousArrayList.get(i)==0){			//对于连续属性，连续属性对属性的值空间排序，然后找出最小GINI值的分裂点
			ArrayList<Double> tempAttributeValueList = new ArrayList<Double>();
			
			for(int x = 0;x<data.length;x++){
				if(!tempAttributeValueList.contains(data[x][i])){
					tempAttributeValueList.add(data[x][i]);
				}
			}
			
			double[] tempAttriValue = new double[tempAttributeValueList.size()];
			for(int x = 0;x < tempAttributeValueList.size();x++){
				tempAttriValue[x] = tempAttributeValueList.get(x);
			}
			
			for(int x = 0;x<tempAttriValue.length;x++){
				int min = x;
				for(int j = x;j<tempAttriValue.length;j++){
					if(tempAttriValue[min]>tempAttriValue[j])
						min = j;
				}
				if(min!=x){
					double t = tempAttriValue[min];
					tempAttriValue[min] = tempAttriValue[x];
					tempAttriValue[x] = t;
				}
			}
			
			double minGini=10.0;
			double splitVal=0;
			for(int x = 0;x<(tempAttriValue.length-1);x++){
				double mid = (tempAttriValue[x]+tempAttriValue[x+1])/2.0;
				HashMap<Integer, Double> lessSet = new HashMap<Integer,Double>();
				HashMap<Integer, Double> biggerSet = new HashMap<Integer,Double>();
				for(int y = 0;y<data.length;y++){
					int lastIndex = data[0].length-1;
					double theme = data[y][lastIndex];
					if(data[y][i]<mid){
						lessSet.put(y, theme);
					}
					else{
						biggerSet.put(y, theme);
					}
				}
				double biggerSetSize = biggerSet.size();
				double lessSetSize = lessSet.size();
				
				double giniGain = (biggerSetSize/(biggerSetSize+lessSetSize))*getGini(biggerSet)+
						(lessSetSize/(lessSetSize+biggerSetSize))*getGini(lessSet);
				if(minGini>giniGain){
					minGini = giniGain;
					splitVal = mid;
				}
			}
			gini = minGini;
			splitValMap.put(i, splitVal);
		}
		else{			//对于离散属性，把所有属性值都作为一次分裂尝试，找到最佳的分裂方式
			
			HashSet<Double> tempAttriValue = new HashSet<Double>();
			for(int x = 0;x<data.length;x++){
				tempAttriValue.add(data[x][i]);
			}
			Iterator<Double> iterator = tempAttriValue.iterator();
			double miniGini = 10.0;
			double splitVal=0;
			while(iterator.hasNext()){
				double attriVal = iterator.next();
				
				HashMap<Integer, Double> isThisAttributeValHashMap = new HashMap<Integer, Double>();
				HashMap<Integer, Double> notThisAttributeValHashMap = new HashMap<Integer, Double>();
				
				int lastIndex = data[0].length - 1;
				for(int x = 0;x<data.length;x++){
					double theme = data[x][lastIndex];
					
					if(data[x][i]==attriVal){
						isThisAttributeValHashMap.put(x, theme);
					}
					else {
						notThisAttributeValHashMap.put(x, theme);
					}
				}
				
				double isSize = isThisAttributeValHashMap.size();
				double notSize = notThisAttributeValHashMap.size();
				//
				double giniGain = (isSize/(isSize+notSize))*getGini(isThisAttributeValHashMap)+
						(notSize/(isSize+notSize))*getGini(notThisAttributeValHashMap);
				
				if(miniGini > giniGain){
					miniGini = giniGain;
					splitVal = attriVal;
				}
				
			}
			gini = miniGini;
			splitValMap.put(i, splitVal);
		}
		return gini;
	}
	
	
	public double getGini(HashMap<Integer, Double> map){
		double size = map.size();
		HashMap<Double, Double> themeNumMap = new HashMap<Double, Double>();
		Iterator iterator = map.entrySet().iterator();
		while(iterator.hasNext()){
			Map.Entry<Integer, Double> entry = (Entry<Integer, Double>) iterator.next();
			
			double val = entry.getValue();
			if(themeNumMap.containsKey(val)){
				double d = themeNumMap.get(val);
				themeNumMap.put(val, d+1.0);
			}
			else{
				themeNumMap.put(val, 1.0);
			}
		}
		double sum = 0;
		iterator = themeNumMap.entrySet().iterator();
		while(iterator.hasNext()){
			Map.Entry<Integer, Double> entry = (Entry<Integer, Double>) iterator.next();
			double d = entry.getValue();
			sum+=(d/size)*(d/size);
		}
		sum = 1-sum;
		
		return sum;
	}
	
	public double getVariance(ArrayList<Double> list){
		double sum = 0;
		for(int i = 0;i < list.size();i++){
			sum += list.get(i);
		}
		double average = sum/list.size();
		double variance = 0;
		for(int i = 0;i < list.size();i++){
			double d = list.get(i);
			variance += (d-average)*(d-average);
		}
		variance = variance/list.size();
		
		return variance;
	}
	
	public double getSplit(int i){
		return splitValMap.get(i);
	}
	
	public void splitData(int attriIndex){
		
		ArrayList<Integer> leftList = new ArrayList<Integer>();
		ArrayList<Integer> rightList = new ArrayList<Integer>();
		int len = data[0].length;
		double splitVal = splitValMap.get(attriIndex);
		if(continuousArrayList.get(attriIndex)==0){
			for(int i = 0;i<data.length;i++){
				if(data[i][attriIndex]<splitVal){
					leftList.add(i);
				}
				else {
					rightList.add(i);
				}
			}
		}
		else{
			for(int i = 0;i<data.length;i++){
				if(data[i][attriIndex]==splitVal){
					leftList.add(i);
				}
				else {
					rightList.add(i);
				}
			}
		}
		int leftSize = leftList.size();
		int rightSize = rightList.size();
		leftData = new double[leftSize][len];
		rightData = new double[rightSize][len];
		
		for(int i = 0;i<leftSize;i++){
			int row = leftList.get(i);
			for(int j = 0;j<len;j++){
				leftData[i][j] = data[row][j];
			}
		}
		for(int i = 0;i<rightSize;i++){
			int row = rightList.get(i);
			for(int j = 0;j<len;j++){
				rightData[i][j] = data[row][j];
			}
		}
	}
	
	
	
	public double[][] getRightData(){
		
		return rightData;
	}
	public double[][] getLeftData(){
		
		return leftData;
	}
	
	public static double isPure(ArrayList<Double> list) {
		HashSet<Double> set = new HashSet<Double>();
		for(double d:list){
			set.add(d);
		}
		if(set.size()>1) return -1;
		Iterator<Double> iterator = set.iterator();
		return iterator.next();
	}
	
	public static ArrayList<Double> getTarget(double[][] trainData){
		ArrayList<Double> list = new ArrayList<Double>();
		int lastIndex = trainData[0].length-1;
		for(int i = 0;i < trainData.length ;i++){
			list.add(trainData[i][lastIndex]);
		}
		return list;
	}
	public static double setDataSetClass(ArrayList<Double> list){
		HashMap<Double, Double> map = new HashMap<Double, Double>();
		
		for(int i = 0;i < list.size();i++){
			if(map.containsKey(list.get(i))){
				double d = map.get(list.get(i));
				map.put(list.get(i), d+1);
			}
			else {
				map.put(list.get(i), 1.0);
			}
		}
		
		Iterator<Double> iterator = map.keySet().iterator();
		double classId = 0;
		double maxClass=-1;
		while (iterator.hasNext()) {
			double key = iterator.next();
			double val = map.get(key);
			if(maxClass < val){
				maxClass = val;
				classId = key;
			}
		}
		return classId;
	}
	
	public static double setDataSetMeanValue(ArrayList<Double> list){
		double sum = 0;
		for(int i = 0;i < list.size();i++){
			sum += list.get(i);
		}
		double average = sum/list.size();
		return average;
	}
}




