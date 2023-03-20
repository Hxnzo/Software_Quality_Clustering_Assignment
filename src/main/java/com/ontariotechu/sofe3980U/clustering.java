package com.ontariotechu.sofe3980U;

import java.io.File;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.DensityBasedSpatialClustering;
import net.sf.javaml.clustering.Cobweb;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.DistanceMeasure;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.Gamma;
import net.sf.javaml.clustering.evaluation.CIndex;
import net.sf.javaml.distance.EuclideanDistance;


public class clustering {

    public static void main(String[] args) throws Exception 
    {
        //load iris dataset from files
        Dataset data = FileHandler.loadDataset(new File("C:src/resources/iris.data"), 4, ",");
    
        //perform kMeans clustering
        System.out.println("\nKMeans Clusters:\n");
        Clusterer kMeans = new KMeans();
        Dataset[] kMeansClusters = kMeans.cluster(data);

        //display the cluster
        displayClusters(kMeansClusters);
        //calculate and display scores
        displayScores(kMeansClusters);

        System.out.println("\n----------------------------------------------------------------------------------------------------------------\n");
    
        //perform cobweb clustering
        System.out.println("\nCobweb Clusters:\n");
        Clusterer cobweb = new Cobweb();
        Dataset[] cobwebClusters = cobweb.cluster(data);

        //display the cluster
        displayClusters(cobwebClusters);
        //calculate and display scores
        displayScores(cobwebClusters);

        System.out.println("\n----------------------------------------------------------------------------------------------------------------\n");
    
        //perform Density Based Spatial clustering
        System.out.println("\nDensity Based Space Clusters:\n");
        Clusterer dbsc = new DensityBasedSpatialClustering();
        Dataset[] dbscClusters = dbsc.cluster(data);

        //display the cluster
        displayClusters(dbscClusters);
        //calculate and display scores
        displayScores(dbscClusters);

        System.out.println("\n----------------------------------------------------------------------------------------------------------------\n");
    }
    
    //Displays clusters and their instances
    private static void displayClusters(Dataset[] clusters) 
    {
        //iterate through the clusters
        for (int i = 0; i < clusters.length; i++) 
        {
            //display cluster number
            System.out.println("Cluster " + i + ":");

            //iterate through every instance inside the cluster
            for (Instance instance : clusters[i]) 
            {
                System.out.println(instance);
            }
            System.out.println();
        }
    }
    
    //Calculate and display evaluation scores each dataset
    private static void displayScores(Dataset[] clusters) 
    {
        //calculate all the values
        DistanceMeasure calcEucDistance = new EuclideanDistance();
        ClusterEvaluation calcSumOfSqrErr = new SumOfSquaredErrors();
        ClusterEvaluation calcCindex = new CIndex(calcEucDistance);
        ClusterEvaluation calcGamma = new Gamma(calcEucDistance);
        ClusterEvaluation calcAIC = new AICScore();
    
        //display all the values
        System.out.println("SumOfSquaredErrors score of Clusters: " + calcSumOfSqrErr.score(clusters));
        System.out.println("Cindex score of Clusters: " + calcCindex.score(clusters));
        System.out.println("Gamma score of Clusters: " + calcGamma.score(clusters));
        System.out.println("AICScore of Clusters: " + calcAIC.score(clusters));
    }
}